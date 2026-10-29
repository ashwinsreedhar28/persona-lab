"""C1-C4 in one HF process (the 98 GB model loads once). Forward passes only. Each stage writes its aggregate JSON to
/workspace/out before the next starts; a stage whose output exists is skipped on rerun (--force to redo).
Direction tensors and val activations go to /workspace/work (pod only).

C1 directions: resid_pre at every layer and post-instruction position, 128 harmful + 128 harmless train prompts;
   r(l, pos) = mean(harmful) - mean(harmless). Norms, relative norms, cosine matrices -> directions_stats.json.
   Refusal tokens (fixed here, before selection): argmax first tokens on harmful train (from logits) with count >= 8
   whose harmful-train count exceeds their harmless-train count.
C2 selection, val split only, rule fixed in PLAN.md: candidates = first 80% of layers x post-instruction positions.
   AUROC of h.r_hat (harmful vs harmless val, at the candidate's layer and position) -> top 20.
   For those: induce = mean increase in refusal-token log-odds at the first output position on harmless val when
   alpha * r_hat is added to resid_pre of the candidate's layer (alpha = ||r(l, pos)||, i.e. the mean difference itself);
   KL = first-token KL(base || ablated) on harmless val with r_hat ablated from every writer.
   Pick the highest induce with induce > 0, AUROC >= 0.9, KL <= 0.1. No bypass score is computed.
C3 attribution: each writer's contribution to h.r_hat (embedding, attention, dense MLP, routed experts per expert,
   shared expert), harmful and harmless val separately, over all real tokens and at the selected position.
C4 in-memory edits: base -> noshared -> full; per-layer mean |h.r_hat| on val, first-token KL on harmless_kl, and
   full-edit vs run-time-hook equivalence on harmless val. Nothing is saved.
"""
import argparse
import hashlib
import time
import types
from collections import Counter

import numpy as np
import torch

from common import (ALL_KINDS, DATA, NOSHARED_KINDS, OUT, WORK, ablated, added, auroc, chat_ids, edit_weights, eoi_ids,
                    free, kl_rows, last_logprobs, layers, load_hf, log, pad_id, pad_right, read_json, recording,
                    token_logodds, write_json)

KL_MAX, AUROC_MIN, LAYER_FRAC, TOP_K, BS = 0.1, 0.9, 0.8, 20, 32


def cache_acts(model, tok, prompts, eoi, pad):
    """resid_pre at every layer and post-instruction position, plus argmax first token and mean ||h||."""
    n_pos, nL = len(eoi), len(layers(model))
    batch = [chat_ids(tok, p) for p in prompts]
    acts, firsts = [], []
    for s in range(0, len(batch), BS):
        x, lens = pad_right(batch[s:s + BS], pad)
        idx = lens[:, None] - n_pos + torch.arange(n_pos, device="cuda")[None]
        rows = torch.arange(len(lens), device="cuda")[:, None]
        kept = {}

        def take(i, h):
            if i < nL:
                kept[i] = h[rows, idx].float().cpu()
        with recording(model, take):
            h = model.model(input_ids=x, use_cache=False).last_hidden_state
        firsts += model.lm_head(h[torch.arange(len(lens), device="cuda"), lens - 1]).argmax(-1).tolist()
        acts.append(torch.stack([kept[i] for i in range(nL)], 1))
    return torch.cat(acts).numpy(), firsts                                   # (N, layers, pos, d)


def cos_matrix(v):
    u = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return (u @ u.T).round(4).tolist()


def c1(model, tok, pad):
    eoi = eoi_ids(tok)
    hA, h_first = cache_acts(model, tok, read_json(DATA / "harmful_train.json"), eoi, pad)
    bA, b_first = cache_acts(model, tok, read_json(DATA / "harmless_train.json"), eoi, pad)
    hV, _ = cache_acts(model, tok, read_json(DATA / "harmful_val.json"), eoi, pad)
    bV, _ = cache_acts(model, tok, read_json(DATA / "harmless_val.json"), eoi, pad)
    r = hA.mean(0) - bA.mean(0)                                              # (layers, pos, d)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(WORK / "directions.npz", r=r.astype(np.float32))               # pod only
    np.savez(WORK / "val_acts.npz", harmful=hV.astype(np.float32), harmless=bV.astype(np.float32))
    hc, bc = Counter(h_first), Counter(b_first)
    ref_ids = sorted(t for t, c in hc.items() if c >= 8 and c > bc.get(t, 0))
    if not ref_ids:
        ref_ids = [hc.most_common(1)[0][0]]
    norms = np.linalg.norm(r, axis=-1)                                       # (layers, pos)
    resid = np.linalg.norm(np.concatenate([hA, bA]), axis=-1).mean(0)       # mean ||h||, (layers, pos)
    nL, n_pos = norms.shape
    write_json(OUT / "directions_stats.json", dict(
        eoi_ids=eoi, eoi_tokens=[tok.decode([t]) for t in eoi], n_pos=n_pos, n_layers=nL,
        position_labels=[f"{p - n_pos:+d} ({tok.decode([eoi[p]])!r})" for p in range(n_pos)],
        norm=norms.round(3).tolist(), mean_resid_norm=resid.round(2).tolist(),
        relative_norm=(norms / resid).round(4).tolist(),
        cos_layer_by_layer_per_pos={str(p - n_pos): cos_matrix(r[:, p]) for p in range(n_pos)},
        cos_between_positions_per_layer=[cos_matrix(r[l]) for l in range(nL)],
        first_token_harmful_train=[(tok.decode([t]), c) for t, c in hc.most_common(10)],
        first_token_harmless_train=[(tok.decode([t]), c) for t, c in bc.most_common(10)],
        refusal_token_rule="argmax first tokens on harmful train with count >= 8 and count > harmless-train count",
        refusal_token_ids=ref_ids, refusal_tokens=[tok.decode([t]) for t in ref_ids],
        refusal_token_share_harmful_train=sum(hc[t] for t in ref_ids) / len(h_first),
        refusal_token_share_harmless_train=sum(bc.get(t, 0) for t in ref_ids) / len(b_first)))
    log("C1 eoi", [tok.decode([t]) for t in eoi], "refusal tokens", [tok.decode([t]) for t in ref_ids])
    log("C1 harmful-train first tokens", [(tok.decode([t]), c) for t, c in hc.most_common(5)])
    log("C1 relative ||r|| at last pos", (norms / resid)[:, -1].round(3).tolist())


def c2(model, tok, pad):
    st = read_json(OUT / "directions_stats.json")
    ref_ids = st["refusal_token_ids"]
    r_all = np.load(WORK / "directions.npz")["r"]
    va = np.load(WORK / "val_acts.npz")
    hV, bV = va["harmful"], va["harmless"]
    nL, n_pos, _ = r_all.shape
    max_layer = int(nL * LAYER_FRAC)
    cands = []
    for l in range(max_layer):
        for p in range(n_pos):
            r = r_all[l, p]
            rh = r / np.linalg.norm(r)
            cands.append(dict(layer=l, pos=p - n_pos, norm=float(np.linalg.norm(r)),
                              auroc=auroc(hV[:, l, p] @ rh, bV[:, l, p] @ rh)))
    top = sorted(cands, key=lambda c: (-c["auroc"], c["layer"], c["pos"]))[:TOP_K]
    harmless = [chat_ids(tok, p) for p in read_json(DATA / "harmless_val.json")]
    lp_base = last_logprobs(model, harmless, pad)
    base_lo = token_logodds(lp_base, ref_ids)
    for k, c in enumerate(top):
        r = r_all[c["layer"], c["pos"] + n_pos]
        rh = r / np.linalg.norm(r)
        with added(model, c["layer"], c["norm"] * rh):
            lp_i = last_logprobs(model, harmless, pad)
        with ablated(model, rh):
            lp_a = last_logprobs(model, harmless, pad)
        c.update(induce=float((token_logodds(lp_i, ref_ids) - base_lo).mean()),
                 kl=float(kl_rows(lp_base, lp_a).mean()))
        c["passes"] = bool(c["induce"] > 0 and c["auroc"] >= AUROC_MIN and c["kl"] <= KL_MAX)
        log(f"C2 [{k + 1}/{TOP_K}] L{c['layer']:02d} pos{c['pos']:+d} auroc={c['auroc']:.3f} "
            f"induce={c['induce']:+.2f} kl={c['kl']:.4f} {'ok' if c['passes'] else '--'}")
    ok = sorted([c for c in top if c["passes"]], key=lambda c: -c["induce"])
    base = dict(harmless_val_refusal_logodds=float(base_lo.mean()))
    rule = (f"top {TOP_K} by val AUROC among layers < {max_layer} x {n_pos} positions; pick highest induce with "
            f"induce > 0, AUROC >= {AUROC_MIN}, KL <= {KL_MAX}; alpha = ||r(l,pos)||; no bypass score")
    if not ok:
        write_json(OUT / "selection.json", dict(error="no candidate passes", rule=rule, baseline=base,
                                                top20=top, all_candidates_auroc=cands))
        raise SystemExit("C2: no candidate passes")
    best = ok[0]
    rh = (r_all[best["layer"], best["pos"] + n_pos] / best["norm"]).astype(np.float32)
    np.save(WORK / "r_hat.npy", rh)                                            # pod only
    write_json(OUT / "selection.json", dict(
        rule=rule, refusal_tokens=st["refusal_tokens"], refusal_token_ids=ref_ids, baseline=base,
        selected=best, n_candidates=len(cands), n_top=len(top), n_passing=len(ok), top20=top,
        all_candidates_auroc=cands, r_hat_sha256=hashlib.sha256(rh.tobytes()).hexdigest(),
        deviation="Arditi et al. pick the lowest bypass score (refusal on harmful prompts with the direction "
                  "removed); this plan never computes it and selects on separation, induce and KL instead."))
    log("C2 selected", best)


# ---------------------------------------------------------------- C3 attribution
class Acc:
    def __init__(self, nL, n_exp):
        self.n = 0.0
        self.embed = 0.0
        self.attn = np.zeros(nL)
        self.mlp = np.zeros(nL)          # dense (layer 0) or routed sum (from per-expert contributions)
        self.routed_bf16 = np.zeros(nL)  # routed output as the model computed it (check)
        self.shared = np.zeros(nL)
        self.expert = np.zeros((nL, n_exp))
        self.resid = np.zeros(nL + 1)


def c3(model, tok, pad):
    r_np = np.load(WORK / "r_hat.npy")
    r = torch.as_tensor(r_np, device="cuda")
    sel = read_json(OUT / "selection.json")["selected"]
    n_pos = read_json(OUT / "directions_stats.json")["n_pos"]
    Ls = layers(model)
    nL, n_exp = len(Ls), model.config.num_experts
    ctx = dict(masks=None, acc=None)

    def add(field, j, c):           # c: per-token contributions, flat (T,)
        for scope, m in ctx["masks"].items():
            a = ctx["acc"][scope]
            v = float((c * m).sum())
            if field == "embed":
                a.embed += v
            else:
                getattr(a, field)[j] += v

    def flat_proj(out):
        out = out[0] if isinstance(out, tuple) else out
        return (out.float() @ r).reshape(-1)

    hs = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: add("embed", 0, flat_proj(o)))]
    patched = []
    for j, L in enumerate(Ls):
        hs.append(L.self_attn.register_forward_hook(lambda m, i, o, j=j: add("attn", j, flat_proj(o))))
        if not hasattr(L, "block_sparse_moe"):
            hs.append(L.mlp.register_forward_hook(lambda m, i, o, j=j: add("mlp", j, flat_proj(o))))
            continue
        blk = L.block_sparse_moe
        hs.append(blk.shared_experts.register_forward_hook(lambda m, i, o, j=j: add("shared", j, flat_proj(o))))

        def fwd(self, hidden_states, j=j):
            identity, shape = hidden_states, hidden_states.shape
            topk_idx, topk_weight = self.gate(hidden_states)
            x = hidden_states.view(-1, shape[-1])
            y = self.moe_infer(x, topk_idx, topk_weight).view(*shape)
            T, k = topk_idx.shape
            fe, fw = topk_idx.reshape(-1), topk_weight.reshape(-1).float()
            ti = torch.arange(T, device=x.device).repeat_interleave(k)
            contrib = torch.zeros(T * k, device=x.device)
            for e in torch.unique(fe).tolist():
                s = (fe == e).nonzero().squeeze(1)
                contrib[s] = (self.experts[e](x[ti[s]]).float() @ r) * fw[s]
            yr = (y.reshape(-1, shape[-1]).float() @ r)
            for scope, m in ctx["masks"].items():
                a = ctx["acc"][scope]
                cm = contrib * m[ti]
                a.expert[j] += torch.zeros(n_exp, device=x.device).index_add_(0, fe, cm).cpu().numpy()
                a.mlp[j] += float(cm.sum())
                a.routed_bf16[j] += float((yr * m).sum())
            if self.config.num_shared_experts is not None:
                y = y + self.shared_experts(identity)
            return y
        blk.forward = types.MethodType(fwd, blk)
        patched.append(blk)

    def take(i, h):
        add("resid", i, (h.float() @ r).reshape(-1))

    res = {}
    try:
        for set_name, fname in (("harmful", "harmful_val.json"), ("harmless", "harmless_val.json")):
            accs = dict(all=Acc(nL, n_exp), pos=Acc(nL, n_exp))
            batch = [chat_ids(tok, p) for p in read_json(DATA / fname)]
            for s in range(0, len(batch), BS):
                x, lens = pad_right(batch[s:s + BS], pad)
                S = x.shape[1]
                ar = torch.arange(S, device="cuda")[None]
                m_all = (ar < lens[:, None]).float().reshape(-1)
                m_pos = (ar == (lens[:, None] + sel["pos"])).float().reshape(-1)
                ctx.update(masks=dict(all=m_all, pos=m_pos), acc=accs)
                for nm, m in ctx["masks"].items():
                    accs[nm].n += float(m.sum())
                with recording(model, take):
                    model.model(input_ids=x, use_cache=False)
            res[set_name] = accs
    finally:
        for h in hs:
            h.remove()
        for blk in patched:
            del blk.forward

    layer_types = ["KDA" if L.is_linear_attn else "MLA" for L in Ls]
    out = dict(r_hat=f"selected L{sel['layer']} pos{sel['pos']:+d}", layer_types=layer_types,
               note="mean contribution to h.r_hat per token; component j writes into resid_pre of layer j+1. "
                    "'pos' = the selected post-instruction position only; 'all' = every real token.",
               scopes={})
    for scope in ("all", "pos"):
        sc = {}
        means = {}
        for set_name in ("harmful", "harmless"):
            a = res[set_name][scope]
            n = a.n
            mlp = a.mlp / n
            means[set_name] = dict(embed=a.embed / n, attn=a.attn / n, mlp=mlp, shared=a.shared / n,
                                   expert=a.expert / n, resid=a.resid / n, routed_bf16=a.routed_bf16 / n)
            m = means[set_name]
            recon = np.array([m["embed"] + (m["attn"][:l] + m["mlp"][:l] + m["shared"][:l]).sum()
                              for l in range(nL + 1)])
            sc[set_name] = dict(n_tokens=n, embed=m["embed"], attn=m["attn"].tolist(),
                                dense_mlp_l0=float(m["mlp"][0]), routed=[None] + m["mlp"][1:].tolist(),
                                shared=[None] + m["shared"][1:].tolist(), resid_pre=m["resid"].tolist(),
                                recon_max_abs_err=float(np.abs(recon - m["resid"]).max()),
                                routed_vs_model_max_abs_err=float(np.abs(m["mlp"][1:] - m["routed_bf16"][1:]).max()))
        d = {k: means["harmful"][k] - means["harmless"][k] for k in ("embed", "attn", "mlp", "shared", "expert")}
        sc["diff_harmful_minus_harmless"] = dict(embed=float(d["embed"]), attn=d["attn"].tolist(),
                                                 dense_mlp_l0=float(d["mlp"][0]), routed=[None] + d["mlp"][1:].tolist(),
                                                 shared=[None] + d["shared"][1:].tolist())
        experts = []
        for j in range(1, nL):
            row = dict(layer=j)
            for nm, ev in (("harmful", means["harmful"]["expert"][j]), ("diff", d["expert"][j])):
                tot = np.abs(ev).sum()
                order = np.argsort(-np.abs(ev))
                row[nm] = dict(top10=[dict(expert=int(e), contribution=float(ev[e]),
                                           share_of_abs=float(abs(ev[e]) / tot)) for e in order[:10]],
                               top1_share=float(np.abs(ev[order[:1]]).sum() / tot),
                               top10_share=float(np.abs(ev[order[:10]]).sum() / tot),
                               top50_share=float(np.abs(ev[order[:50]]).sum() / tot))
            experts.append(row)
        sc["routed_experts"] = experts
        type_share = []
        for j in range(nL):
            comps = dict(attn=abs(d["attn"][j]), mlp=abs(d["mlp"][j]), shared=abs(d["shared"][j]))
            tot = sum(comps.values())
            type_share.append(dict(layer=j, attn_type=layer_types[j],
                                   **{k: (v / tot if tot > 0 else None) for k, v in comps.items()}))
        sc["diff_share_by_component_type"] = type_share
        out["scopes"][scope] = sc
    write_json(OUT / "attribution.json", out)
    a = out["scopes"]["pos"]
    log("C3 recon err harmful/harmless (pos):", a["harmful"]["recon_max_abs_err"], a["harmless"]["recon_max_abs_err"],
        "routed check:", a["harmful"]["routed_vs_model_max_abs_err"])


# ---------------------------------------------------------------- C4 edits
def projections(model, batch, pad, r):
    nL = len(layers(model))
    sums, count = torch.zeros(nL + 1, dtype=torch.float64), 0
    for s in range(0, len(batch), BS):
        x, lens = pad_right(batch[s:s + BS], pad)
        valid = (torch.arange(x.shape[1], device="cuda")[None] < lens[:, None]).float()

        def take(i, h):
            sums[i] += float(((h.float() @ r).abs() * valid).sum())
        with recording(model, take):
            model.model(input_ids=x, use_cache=False)
        count += int(lens.sum())
    return (sums / count).tolist()


def c4(model, tok, pad):
    r_np = np.load(WORK / "r_hat.npy")
    r = torch.as_tensor(r_np, device="cuda")
    hv = [chat_ids(tok, p) for p in read_json(DATA / "harmful_val.json")]
    bv = [chat_ids(tok, p) for p in read_json(DATA / "harmless_val.json")]
    klp = [chat_ids(tok, p) for p in read_json(DATA / "harmless_kl.json")]
    proj, lp = {}, {}

    def measure(v):
        proj[v] = dict(val=projections(model, hv + bv, pad, r), harmful_val=projections(model, hv, pad, r),
                       harmless_val=projections(model, bv, pad, r))
        lp[v] = last_logprobs(model, klp, pad)
        log(f"C4 {v}: mean|h.r| val {np.round(proj[v]['val'], 3).tolist()}")

    measure("base")
    with ablated(model, r_np):
        lp_hook_bv = last_logprobs(model, bv, pad)
    c_ns = edit_weights(model, r_np, NOSHARED_KINDS)
    log("C4 noshared edit", c_ns)
    measure("noshared")
    c_sh = edit_weights(model, r_np, tuple(k for k in ALL_KINDS if k not in NOSHARED_KINDS))
    log("C4 shared edit", c_sh)
    measure("full")
    eq = kl_rows(lp_hook_bv, last_logprobs(model, bv, pad))

    nL = len(layers(model))
    labels = [f"resid_pre L{i}" for i in range(nL)] + [f"L{nL - 1} output (pre-norm)"]
    out = dict(metric="mean |h . r_hat| over every real token", positions=labels, edit_counts=dict(
        noshared=c_ns, shared_added_for_full=c_sh))
    for split in ("val", "harmful_val", "harmless_val"):
        b = proj["base"][split]
        out[split] = dict(base=b, noshared=proj["noshared"][split], full=proj["full"][split],
                          ratio_noshared=[x / y for x, y in zip(proj["noshared"][split], b)],
                          ratio_full=[x / y for x, y in zip(proj["full"][split], b)])
    out["max_ratio_full_val"] = max(out["val"]["ratio_full"])
    out["max_ratio_noshared_val"] = max(out["val"]["ratio_noshared"])
    write_json(OUT / "projection.json", out)
    kl = {}
    for v in ("noshared", "full"):
        k = kl_rows(lp["base"], lp[v])
        kl[v] = dict(mean=float(k.mean()), median=float(k.median()), max=float(k.max()),
                     per_prompt=k.numpy().round(5).tolist())
    write_json(OUT / "kl.json", dict(
        set="harmless_kl (100 held-out harmless prompts)", direction="KL(base || edited), first output token",
        **kl, equivalence_full_edit_vs_hook=dict(set="harmless_val (32)", mean=float(eq.mean()), max=float(eq.max()))))
    log("C4 KL noshared", kl["noshared"]["mean"], "full", kl["full"]["mean"], "| equivalence", float(eq.mean()))
    log("C4 max ratio full", out["max_ratio_full_val"], "noshared", out["max_ratio_noshared_val"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["c1", "c2", "c3", "c4"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    model, tok = load_hf()
    pad = pad_id(model, tok)
    log(f"loaded in {time.time() - t0:.0f}s")
    marker = dict(c1=OUT / "directions_stats.json", c2=OUT / "selection.json", c3=OUT / "attribution.json",
                  c4=OUT / "kl.json")
    for st in args.stages:
        if marker[st].exists() and not args.force:
            log(f"{st}: output exists, skipped")
            continue
        s0 = time.time()
        dict(c1=c1, c2=c2, c3=c3, c4=c4)[st](model, tok, pad)
        log(f"{st} done in {time.time() - s0:.0f}s")
    free(model)


if __name__ == "__main__":
    main()

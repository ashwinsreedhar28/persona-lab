"""A3. Select r-hat on the val split with Arditi's criteria.

For every candidate r(l, pos) with l in the first 80% of layers:
  bypass  = mean refusal score on harmful val, r-hat ablated from every residual writer
  induce  = mean refusal score on harmless val, r(l, pos) added to resid_pre of layer l at all positions
  kl      = mean first-token KL(base || ablated) on harmless val
Keep candidates with induce > 0 (and above the harmless baseline) and kl <= 0.1; pick the lowest bypass.
Refusal score = log-odds of the refusal first tokens ("I", "As": Arditi's Qwen set; the base model opens
124/128 harmful-train answers with "I").
"""
import argparse
import json
import time

import mlx.core as mx
import numpy as np

from common import (BASE_ID, DATA, OUT, ablated, added, chat_ids, kl_rows, last_logprobs, load, peak_ram_gb,
                    read_json, refusal_score, write_json)

REFUSAL_TOKENS = ["I", "As"]
KL_MAX = 0.1
LAYER_FRAC = 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N candidates")
    args = ap.parse_args()
    t0 = time.time()
    model, tok = load(BASE_ID)
    refusal_ids = []
    for t in REFUSAL_TOKENS:
        ids = tok.encode(t, add_special_tokens=False)
        assert len(ids) == 1, (t, ids)
        refusal_ids.append(ids[0])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    harmful = [chat_ids(tok, p) for p in read_json(DATA / "harmful_val.json")]
    harmless = [chat_ids(tok, p) for p in read_json(DATA / "harmless_val.json")]

    z = np.load(OUT / "directions.npz")
    r_all = z["r"]
    n_layers, n_pos, _ = r_all.shape
    max_layer = int(n_layers * LAYER_FRAC)          # layers 0 .. max_layer-1 (first 80%)

    base_h = last_logprobs(model, harmful, pad)
    base_b = last_logprobs(model, harmless, pad)
    baseline = dict(harmful_refusal_score=float(refusal_score(base_h, refusal_ids).mean()),
                    harmless_refusal_score=float(refusal_score(base_b, refusal_ids).mean()))
    print("baseline", baseline, flush=True)

    cands = [(l, p) for l in range(max_layer) for p in range(n_pos)]
    if args.limit:
        cands = cands[-args.limit:]
    ckpt = OUT / "selection_partial.jsonl"            # resume after a crash or stall
    done = {}
    if ckpt.exists() and not args.limit:
        for line in ckpt.read_text().splitlines():
            row = json.loads(line)
            done[(row["layer"], row["pos"])] = row
    rows = []
    for k, (l, p) in enumerate(cands):
        if (l, p - n_pos) in done:
            rows.append(done[(l, p - n_pos)])
            continue
        r = r_all[l, p]
        norm = float(np.linalg.norm(r))
        row = dict(layer=l, pos=p - n_pos, norm=norm)
        if norm < 1e-4:
            row.update(skipped="zero direction (identical template tokens)")
            rows.append(row)
            if not args.limit:
                with ckpt.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            continue
        r_hat = r / norm
        with ablated(model, r_hat):
            lp_h = last_logprobs(model, harmful, pad)
            lp_b = last_logprobs(model, harmless, pad)
        with added(model, l, r):
            lp_i = last_logprobs(model, harmless, pad)
        row.update(bypass=float(refusal_score(lp_h, refusal_ids).mean()),
                   induce=float(refusal_score(lp_i, refusal_ids).mean()),
                   kl=float(kl_rows(base_b, lp_b).mean()))
        row["passes"] = bool(row["induce"] > 0 and row["induce"] > baseline["harmless_refusal_score"]
                             and row["kl"] <= KL_MAX)
        rows.append(row)
        if not args.limit:
            with ckpt.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"[{k + 1}/{len(cands)}] L{l:02d} pos{p - n_pos:+d} bypass={row['bypass']:+.2f} "
              f"induce={row['induce']:+.2f} kl={row['kl']:.3f} {'ok' if row['passes'] else '--'} "
              f"({time.time() - t0:.0f}s)", flush=True)

    ok = sorted([r for r in rows if r.get("passes")], key=lambda r: r["bypass"])
    if not ok:
        raise SystemExit("no candidate passes the filters")
    best = ok[0]
    write_json(OUT / "selection.json", dict(
        model=BASE_ID, refusal_tokens=REFUSAL_TOKENS, refusal_token_ids=refusal_ids, kl_max=KL_MAX,
        layer_limit=f"layer < {max_layer} (first {int(LAYER_FRAC * 100)}% of {n_layers})",
        baseline=baseline, selected=dict(layer=best["layer"], pos=best["pos"]),
        ranking=ok, all_candidates=rows, peak_ram=peak_ram_gb(), seconds=round(time.time() - t0, 1)))
    np.save(OUT / "r_hat.npy", (r_all[best["layer"], best["pos"]] / best["norm"]).astype(np.float32))
    print("selected", best, flush=True)
    print("top 5", [(r["layer"], r["pos"], round(r["bypass"], 2), round(r["kl"], 3)) for r in ok[:5]], flush=True)


if __name__ == "__main__":
    main()

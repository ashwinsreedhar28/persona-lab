"""A5. Mechanical check: per-layer mean |h . r-hat| on harmful+harmless val, base vs edited.

h = resid_pre of each of the 32 layers plus the last layer's output (pre final norm), every real token
(no padding). Base and edited run one after the other in this process; the first is freed before the
second loads. Also checks that the weight edit matches run-time ablation: first-token KL between
(base + ablation hooks) and (edited model) on the same prompts should be ~0.
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from common import (ABLITERATED, BASE_ID, DATA, OUT, ablated, chat_ids, free, kl_rows, last_logprobs, layers,
                    load, pad_right, peak_ram_gb, read_json, recording, text_model, write_json)


def projections(model, tok, prompts, r, bs=16):
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batch = [chat_ids(tok, p) for p in prompts]
    n = len(layers(model)) + 1
    sums, count = np.zeros(n), 0
    with recording(model) as recs:
        for s in range(0, len(batch), bs):
            x, lens = pad_right(batch[s:s + bs], pad)
            valid = mx.array(np.arange(x.shape[1])[None] < lens[:, None])
            for rec in recs:
                rec.kept = []
                rec.take = lambda h: (mx.abs(h.astype(mx.float32) @ r) * valid).sum()
            h = text_model(model).embed_tokens(x)
            for L in layers(model):               # run layers by hand to also get the last layer's output
                h = L(h, mask="causal" if not L.is_linear else None)
            last = (mx.abs(h.astype(mx.float32) @ r) * valid).sum()
            mx.eval([rec.kept for rec in recs], last)
            sums += np.array([float(rec.kept[0]) for rec in recs] + [float(last)])
            count += int(lens.sum())
            for rec in recs:
                rec.take, rec.kept = None, []
    return sums / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edited", type=Path, default=ABLITERATED)
    ap.add_argument("--r_hat", type=Path, default=OUT / "r_hat.npy")
    ap.add_argument("--out", type=Path, default=OUT / "projection.json")
    args = ap.parse_args()
    t0 = time.time()
    r_np = np.load(args.r_hat).astype(np.float32)
    r = mx.array(r_np)
    prompts = read_json(DATA / "harmful_val.json") + read_json(DATA / "harmless_val.json")

    model, tok = load(BASE_ID)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batch = [chat_ids(tok, p) for p in prompts]
    base = projections(model, tok, prompts, r)
    tied_edit = json.loads((args.edited / "config.json").read_text())["text_config"].get("tie_word_embeddings", True)
    with ablated(model, r_np, edit_unembed=tied_edit):
        hooked = projections(model, tok, prompts, r)
        lp_hook = last_logprobs(model, batch, pad)
    free(model)
    model = None

    model, tok = load(args.edited)
    edited = projections(model, tok, prompts, r)
    lp_edit = last_logprobs(model, batch, pad)
    kl = kl_rows(lp_hook, lp_edit)
    ratio = edited / base
    names = [f"resid_pre L{i}" for i in range(len(base) - 1)] + ["last layer output"]
    res = dict(tied_edit=tied_edit, prompts="harmful_val + harmless_val (64), all real tokens", r_hat=str(args.r_hat),
               rows=[dict(where=n, base=float(b), edited=float(e), hooked=float(h), ratio=float(q))
                     for n, b, e, h, q in zip(names, base, edited, hooked, ratio)],
               max_ratio=float(ratio.max()), argmax=names[int(ratio.argmax())], threshold=0.01,
               pass_all_layers=bool((ratio <= 0.01).all()),
               edit_vs_hook_first_token_kl=dict(mean=float(kl.mean()), max=float(kl.max())),
               peak_ram=peak_ram_gb(), seconds=round(time.time() - t0, 1))
    write_json(args.out, res)
    print(f"max ratio {res['max_ratio']:.5f} at {res['argmax']}; pass={res['pass_all_layers']}")
    print(f"edit vs hook KL mean={kl.mean().item():.5f} max={kl.max().item():.5f}")
    print("ratios:", np.round(ratio, 5).tolist())


if __name__ == "__main__":
    main()

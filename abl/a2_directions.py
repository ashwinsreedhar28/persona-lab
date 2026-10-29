"""A2. Candidate refusal directions r(l, pos) = mean(harmful) - mean(harmless) of resid_pre,
at every layer and every post-instruction template token (non-thinking template).
Also records the base model's first-token distribution on the train splits (used to pick refusal tokens)."""
import time
from collections import Counter

import mlx.core as mx
import numpy as np

from common import (BASE_ID, DATA, OUT, chat_ids, eoi_ids, layers, load, pad_right, peak_ram_gb, read_json,
                    recording, text_model, unembed, write_json)


def cache(model, tok, prompts, eoi, bs=16):
    n_pos = len(eoi)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batch = [chat_ids(tok, p) for p in prompts]
    acts = [[] for _ in layers(model)]
    first = []
    with recording(model) as recs:
        for s in range(0, len(batch), bs):
            x, lens = pad_right(batch[s:s + bs], pad)
            idx = mx.array(lens[:, None] - n_pos + np.arange(n_pos)[None])
            rows = mx.arange(len(lens))[:, None]
            for r in recs:
                r.kept = []
                r.take = lambda h, idx=idx, rows=rows: h[rows, idx].astype(mx.float32)
            h = text_model(model)(x)
            logits = unembed(model, h[mx.arange(len(lens)), mx.array(lens - 1)])
            mx.eval([r.kept for r in recs], logits)
            for i, r in enumerate(recs):
                acts[i].append(np.array(r.kept[0]))
            first.extend(np.array(mx.argmax(logits, -1)).tolist())
            for r in recs:
                r.take, r.kept = None, []
    return np.stack([np.concatenate(a) for a in acts], 1), first   # (N, layers, pos, d)


def main():
    t0 = time.time()
    model, tok = load(BASE_ID)
    eoi = eoi_ids(tok)
    h_acts, h_first = cache(model, tok, read_json(DATA / "harmful_train.json"), eoi)
    b_acts, b_first = cache(model, tok, read_json(DATA / "harmless_train.json"), eoi)
    mean_h, mean_b = h_acts.mean(0), b_acts.mean(0)
    r = mean_h - mean_b                                         # (layers, pos, d)
    np.savez(OUT / "directions.npz", r=r, mean_harmful=mean_h, mean_harmless=mean_b,
             eoi_ids=np.array(eoi), eoi_tokens=np.array([tok.decode([t]) for t in eoi]), model=BASE_ID,
             note="resid_pre at layer l, post-instruction position pos (index into eoi_ids, -len(eoi)..-1)")

    def top(first):
        return [(tok.decode([t]), c) for t, c in Counter(first).most_common(8)]
    stats = dict(eoi_tokens=[tok.decode([t]) for t in eoi], n_pos=len(eoi),
                 first_token_harmful_train=top(h_first), first_token_harmless_train=top(b_first),
                 direction_norms=np.linalg.norm(r, axis=-1).round(2).tolist(),
                 peak_ram=peak_ram_gb(), seconds=round(time.time() - t0, 1))
    write_json(OUT / "directions_stats.json", stats)
    print("eoi tokens:", stats["eoi_tokens"])
    print("greedy first token, harmful train:", stats["first_token_harmful_train"])
    print("greedy first token, harmless train:", stats["first_token_harmless_train"])
    print("||r|| at last pos by layer:", np.linalg.norm(r[:, -1], axis=-1).round(1).tolist())
    print("peak", stats["peak_ram"], "sec", stats["seconds"])


if __name__ == "__main__":
    main()

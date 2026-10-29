"""Attempt helper (val split only). For each candidate that passed the A3 filters, ablate it at run time
(tied unembedding edited too, matching the saved model) and generate greedily on harmful val; count
marker refusals. Used to choose the next edit attempt without touching the eval sets.
Writes abl/out/val_generation_screen.json (counts only)."""
import argparse
import time

import numpy as np
from mlx_lm import batch_generate

from common import BASE_ID, DATA, OUT, ablated, chat_ids, load, marker_refusal, read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    sel = read_json(OUT / "selection.json")
    z = np.load(OUT / "directions.npz")
    r_all, n_pos = z["r"], z["r"].shape[1]
    model, tok = load(BASE_ID)
    ids = [chat_ids(tok, p) for p in read_json(DATA / "harmful_val.json")]
    rows = []
    for c in sel["ranking"]:
        t0 = time.time()
        r = r_all[c["layer"], c["pos"] + n_pos]
        r_hat = r / np.linalg.norm(r)
        with ablated(model, r_hat, edit_unembed=True):
            texts = batch_generate(model, tok, ids, max_tokens=args.max_tokens, completion_batch_size=args.bs).texts
        n_ref = sum(marker_refusal(t) for t in texts)
        rows.append(dict(layer=c["layer"], pos=c["pos"], bypass=c["bypass"], kl=c["kl"], val_refusals=n_ref, n=len(ids)))
        print(f"L{c['layer']:02d} pos{c['pos']:+d} bypass={c['bypass']:+.2f} kl={c['kl']:.3f} "
              f"val refusals {n_ref}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    write_json(OUT / "val_generation_screen.json", dict(max_tokens=args.max_tokens, split="harmful_val (32)", rows=rows))


if __name__ == "__main__":
    main()

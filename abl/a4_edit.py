"""A4. Weight-orthogonalize every residual-stream writer against r-hat and save
models/Qwen3.5-4B-abliterated (bf16, HF key layout, loadable with mlx_lm.load).

  W <- W - r r^T W     out_proj (24 DeltaNet), o_proj (8 attention), down_proj (32), visual merger linear_fc2
  b <- b - (b . r) r   merger linear_fc2 bias (the only writer with a bias)
  E <- E - E r r^T     embed_tokens, in place. lm_head is tied to E, so this also edits the unembedding.
                       (--untie stores an unedited lm_head copy instead; that model is 1.3 GB larger and ran
                       out of GPU memory during batched generation on the 16 GB machine, so it is not used.)
Math runs in fp32 on the bf16 checkpoint and is cast back to bf16. mtp.* is not saved (mlx_lm never loads it).
"""
import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

from common import ABLITERATED, BASE_ID, OUT, peak_ram_gb, write_json
from research_checks import unit_direction

WRITER = re.compile(r"^model\.language_model\.layers\.\d+\.(linear_attn\.out_proj|self_attn\.o_proj|mlp\.down_proj)\.weight$")
MERGER_W, MERGER_B = "model.visual.merger.linear_fc2.weight", "model.visual.merger.linear_fc2.bias"
EMBED = "model.language_model.embed_tokens.weight"


def project_out_rows(W, r):        # W: (d_model, in) -> remove r from the output space
    W32 = W.astype(mx.float32)
    return (W32 - r[:, None] * (r[None, :] @ W32)).astype(W.dtype)


def project_out_cols(E, r, chunk=32768):     # E: (vocab, d_model) -> remove r from every row
    parts = []
    for s in range(0, E.shape[0], chunk):
        e = E[s:s + chunk].astype(mx.float32)
        parts.append((e - (e @ r)[:, None] * r[None, :]).astype(E.dtype))
        mx.eval(parts[-1])
    return mx.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r_hat", type=Path, default=OUT / "r_hat.npy")
    ap.add_argument("--out", type=Path, default=ABLITERATED)
    ap.add_argument("--untie", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    r_np = unit_direction(np.load(args.r_hat), require_unit=True)
    if r_np.shape != (2560,):
        raise ValueError("direction must have Qwen3.5-4B hidden size 2560")
    r = mx.array(r_np)

    src = Path(snapshot_download(BASE_ID, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    new_map, edited, total = {}, [], 0
    for shard in sorted({v for v in index["weight_map"].values()}):
        w = mx.load(str(src / shard))
        out = {}
        for k, v in w.items():
            if k.startswith("mtp."):
                continue
            if WRITER.match(k) or k in (MERGER_W, MERGER_B, EMBED):
                assert v.dtype == mx.bfloat16, (k, v.dtype)    # A_log etc. are fp32 by design; untouched
            if WRITER.match(k) or k == MERGER_W:
                assert v.shape[0] == 2560, (k, v.shape)
                v = project_out_rows(v, r)
                mx.eval(v)                                # free fp32 temporaries tensor by tensor
                edited.append(k)
            elif k == MERGER_B:
                vf = v.astype(mx.float32)
                v = (vf - (vf @ r) * r).astype(v.dtype)
                mx.eval(v)
                edited.append(k)
            elif k == EMBED:
                if args.untie:
                    out["lm_head.weight"] = v             # unedited reader copy (untied)
                v = project_out_cols(v, r)
                edited.append(k)
            out[k] = v
        mx.eval(out)
        mx.save_safetensors(str(args.out / shard), out, metadata={"format": "pt"})
        total += sum(a.nbytes for a in out.values())
        new_map.update({k: shard for k in out})
        del w, out
        mx.clear_cache()

    n_layers_w = sum(bool(WRITER.match(k)) for k in edited)
    assert n_layers_w == 64, n_layers_w            # 24 out_proj + 8 o_proj + 32 down_proj
    (args.out / "model.safetensors.index.json").write_text(
        json.dumps(dict(metadata=dict(total_size=total), weight_map=dict(sorted(new_map.items()))), indent=2))
    cfg = json.loads((src / "config.json").read_text())
    if args.untie:
        cfg["tie_word_embeddings"] = False
        cfg["text_config"]["tie_word_embeddings"] = False
    cfg["text_config"]["mtp_num_hidden_layers"] = 0
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in src.iterdir():
        if f.suffix in (".json", ".txt", ".jinja") and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f, args.out / f.name)
    meta = dict(source=BASE_ID, r_hat=str(args.r_hat),
                r_hat_sha256=hashlib.sha256(np.load(args.r_hat).tobytes()).hexdigest(), out=str(args.out),
                edited_tensors=len(edited), writer_matrices=n_layers_w,
                embed_edited=EMBED in edited, merger_edited=[MERGER_W in edited, MERGER_B in edited],
                lm_head="untied, unedited copy of the original embedding" if args.untie
                else "tied to the edited embedding (unembedding edited too)", mtp="not saved",
                precision="bf16 in, fp32 math, bf16 out", size_gb=round(total / 1e9, 2),
                peak_ram=peak_ram_gb(), seconds=round(time.time() - t0, 1))
    write_json(OUT / "edit.json", meta)
    write_json(args.out / "abl_edit.json", meta)   # travels with the weights: which r-hat this folder holds
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

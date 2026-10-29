"""A0. Load bf16 Qwen3.5-4B in MLX, list every residual-stream writer, save abl/writers.json."""
import json
import struct
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from common import ABL, BASE_ID, chat_ids, layers, load, peak_ram_gb, text_model, write_json, writer_slots


def hf_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    t0 = time.time()
    snap = Path(snapshot_download(BASE_ID, allow_patterns=["*.json", "*.safetensors"]))
    cfg = json.loads((snap / "config.json").read_text())
    hdr = {}
    for f in sorted(snap.glob("*.safetensors")):
        hdr.update({k: v for k, v in hf_header(f).items() if k != "__metadata__"})

    model, tok = load(BASE_ID)
    Ls = layers(model)
    tm = text_model(model)
    lm = model.language_model
    tied = bool(lm.args.tie_word_embeddings)

    writers = []
    for parent, attr, label in writer_slots(model):
        m = getattr(parent, attr)
        w = m.weight
        writers.append(dict(module=label, kind=type(m).__name__, weight_shape=list(w.shape),
                            dtype=str(w.dtype), has_bias="bias" in m,
                            writes="rows of E (d_model)" if label == "embed_tokens" else "out_features = d_model"))
    d_model = tm.embed_tokens.weight.shape[1]
    for w in writers:
        if w["module"] != "embed_tokens":
            assert w["weight_shape"][0] == d_model, w

    # Anything else in the loaded text model with a d_model-sized output that is not a known writer or reader?
    known = {w["module"] for w in writers}
    others = []
    for name, m in tm.named_modules():
        if isinstance(m, nn.Linear) and m.weight.shape[0] == d_model and name not in known:
            others.append(name)

    merger = {k: dict(shape=v["shape"], dtype=v["dtype"]) for k, v in hdr.items() if "visual.merger" in k}
    mtp = {k: dict(shape=v["shape"], dtype=v["dtype"]) for k, v in hdr.items() if k.startswith("mtp.")}
    mx.eval(model(mx.array(chat_ids(tok, "Hello"))[None]))

    types = ["linear_attention" if L.is_linear else "full_attention" for L in Ls]
    blocks_ok = all(types[i:i + 4] == ["linear_attention"] * 3 + ["full_attention"] for i in range(0, len(types), 4))
    report = dict(
        model=BASE_ID, precision="bf16 (HF checkpoint loaded directly by mlx_lm, no quantization)",
        n_layers=len(Ls), hidden_size=d_model, layer_types=types,
        layout_8_blocks_of_3_deltanet_plus_1_attention=blocks_ok and len(Ls) == 32,
        n_deltanet=types.count("linear_attention"), n_attention=types.count("full_attention"),
        writers=writers, n_writers=len(writers),
        biases="none in the loaded text model (attention_bias=false, no bias on out_proj/o_proj/down_proj)"
               if not any(w["has_bias"] for w in writers) else "see has_bias",
        unlisted_d_model_linears=others,
        tied_embeddings=dict(
            tied=tied,
            note="lm_head is tied to embed_tokens (as_linear). Editing E in place would also edit the unembedding.",
            choice="edit E in place and keep it tied, so the unembedding is edited too (the plan's E <- E - E r r^T). "
                   "An untied save (unedited lm_head copy) was tried first: held-out first-token KL 0.064, but the "
                   "extra 1.3 GB made batched generation run out of GPU memory on this 16 GB machine.",
        ),
        vision_merger=dict(
            loaded_by_mlx_lm=False,
            final_linear="model.visual.merger.linear_fc2 (writes image tokens into the residual stream)",
            tensors=merger,
            choice="mlx_lm drops model.visual.* on load, so it plays no part in text eval. The edit still "
                   "projects linear_fc2 weight and bias, and the vision tower is saved alongside the text "
                   "weights (mlx_lm ignores those keys on load).",
        ),
        mtp_head=dict(
            loaded_by_mlx_lm=False,
            note="mlx_lm sanitize() drops every mtp.* key. The MTP head only drafts tokens for speculative "
                 "decoding and is not saved in the edited model.",
            tensors=mtp,
        ),
        hf_checkpoint=dict(path=str(snap), total_tensors=len(hdr), config_tie_word_embeddings=cfg.get("tie_word_embeddings")),
        peak_ram=peak_ram_gb(),
        seconds=round(time.time() - t0, 1),
    )
    write_json(ABL / "writers.json", report)
    print(f"layers={len(Ls)} hidden={d_model} deltanet={report['n_deltanet']} attn={report['n_attention']} "
          f"blocks_ok={report['layout_8_blocks_of_3_deltanet_plus_1_attention']}")
    print(f"writers={len(writers)} biases={any(w['has_bias'] for w in writers)} tied={tied} other_d_model_linears={others}")
    print(f"merger tensors={list(merger)} mtp tensors={len(mtp)}")
    print(f"peak RAM {report['peak_ram']}")


if __name__ == "__main__":
    main()

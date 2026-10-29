"""C0. Load Kimi-Linear-48B-A3B-Instruct in HF transformers (bf16, one GPU), take the writer inventory, record the
chat template's post-instruction tokens, then run the method check with a RANDOM unit vector (no refusal direction):
first-token log-probs on 8 harmless prompts under the run-time ablation hook must match the in-memory weight edit
(KL ~ 0). This process edits weights in GPU memory and exits; nothing is saved. Writes /workspace/out/writers.json.
"""
import time
from collections import Counter

import numpy as np
import torch
from torch import nn

from common import (ALL_KINDS, D_MODEL, DATA, MODEL, OUT, ablated, chat_ids, edit_weights, eoi_ids, kl_rows,
                    last_logprobs, load_hf, log, pad_id, read_json, write_json, writer_modules)


def main():
    t0 = time.time()
    model, tok = load_hf()
    load_s = time.time() - t0
    cfg = model.config
    log(f"loaded in {load_s:.0f}s, GPU alloc {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    scan = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == D_MODEL:
            scan.append(dict(name=name, shape=list(m.weight.shape), bias=m.bias is not None))
        elif isinstance(m, nn.Embedding) and m.embedding_dim == D_MODEL:
            scan.append(dict(name=name, shape=list(m.weight.shape), bias=False))
    structural = writer_modules(model)
    s_names, w_names = {x["name"] for x in scan}, {label for label, _, _ in structural}
    match = s_names == w_names
    kinds = Counter(k for _, _, k in structural)
    shapes = {}
    for _, m, k in structural:
        shapes.setdefault(k, set()).add(tuple(m.weight.shape))
    lm_tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    eoi = eoi_ids(tok)
    layer_types = ["KDA" if L.is_linear_attn else "MLA" for L in model.model.layers]
    pad = pad_id(model, tok)

    # method check: random unit vector, 8 harmless prompts (harmless_val[:8])
    rng = np.random.default_rng(0)
    r = rng.standard_normal(D_MODEL).astype(np.float32)
    r /= np.linalg.norm(r)
    batch = [chat_ids(tok, p) for p in read_json(DATA / "harmless_val.json")[:8]]
    lp_base = last_logprobs(model, batch, pad)
    with ablated(model, r):
        lp_hook = last_logprobs(model, batch, pad)
    counts = edit_weights(model, r, ALL_KINDS)
    lp_edit = last_logprobs(model, batch, pad)
    kl_eq = kl_rows(lp_hook, lp_edit)
    kl_rand = kl_rows(lp_base, lp_edit)
    method = dict(vector="random unit vector, numpy default_rng(0), not a refusal direction",
                  prompts="harmless_val[:8]", edit_counts=counts,
                  kl_hook_vs_weight_edit=dict(mean=float(kl_eq.mean()), max=float(kl_eq.max())),
                  kl_base_vs_random_edit=dict(mean=float(kl_rand.mean()), max=float(kl_rand.max())),
                  top1_agree_hook_vs_edit=int((lp_hook.argmax(-1) == lp_edit.argmax(-1)).sum()))

    res = dict(
        model=str(MODEL), d_model=D_MODEL, n_layers=len(layer_types), layer_types=layer_types,
        mla_layers_0idx=[i for i, t in enumerate(layer_types) if t == "MLA"],
        dense_mlp_layers=[i for i, L in enumerate(model.model.layers) if not hasattr(L, "block_sparse_moe")],
        n_params=sum(p.numel() for p in model.parameters()), n_writers=len(structural),
        generic_scan_matches_structural=match, n_generic_scan=len(scan), writers_by_kind=dict(kinds),
        shapes_by_kind={k: [list(s) for s in v] for k, v in shapes.items()},
        any_writer_bias=any(x["bias"] for x in scan),
        tie_word_embeddings=bool(cfg.tie_word_embeddings), lm_head_shares_storage_with_embed=lm_tied,
        lm_head="reader, untouched", eoi_ids=eoi, eoi_tokens=[tok.decode([t]) for t in eoi],
        probe_template=tok.apply_chat_template([{"role": "user", "content": "Tell me a joke."}],
                                               add_generation_prompt=True, tokenize=False),
        pad_token_id=pad, eos_token_id=cfg.eos_token_id, method_check=method,
        attn_implementation=cfg._attn_implementation,
        load_seconds=round(load_s, 1), gpu_max_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1),
        torch=torch.__version__, writers=sorted(s_names))
    write_json(OUT / "writers.json", res)
    log("writers", len(structural), dict(kinds), "scan matches", match, "bias", res["any_writer_bias"], "tied", lm_tied)
    log("eoi tokens", res["eoi_tokens"])
    log("method check", method)
    assert match and len(structural) == 6711 and not cfg.tie_word_embeddings and not lm_tied


if __name__ == "__main__":
    main()

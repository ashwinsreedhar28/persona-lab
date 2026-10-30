"""
Qwen3.5's bundled multi-token-prediction (MTP) head, in MLX.

Qwen ships one extra full-attention decoder layer with the checkpoint (keys `mtp.*`), trained to
predict token i+2 from the main model's hidden state at position i and the embedding of token i+1.
Chained on its own outputs it drafts k tokens for the price of k single-layer passes, with no
second model. mlx-lm drops these weights at conversion, so they're read from the bf16 HF
checkpoint (Qwen/Qwen3.5-4B, ~9 GB, already cached if you ran the torch version).

Architecture (from vLLM's qwen3_5_mtp.py):
    x = fc( concat( rmsnorm_e(embed(tok)), rmsnorm_h(hidden) ) )
    x = DecoderLayer_full_attention(x)      # own KV cache
    h = rmsnorm(x);  logits = lm_head(h)
"""

from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import os
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import DecoderLayer
from mlx_lm.models.base import create_attention_mask

HF_REPO = os.environ.get("MTP_REPO", "Qwen/Qwen3.5-4B")
NORM_SUFFIXES = ("input_layernorm.weight", "post_attention_layernorm.weight", "norm.weight",
                 "q_norm.weight", "k_norm.weight", "pre_fc_norm_hidden.weight", "pre_fc_norm_embedding.weight")


class MTPHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        H = args.hidden_size
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]   # full attention
        assert not self.layers[0].is_linear
        self.norm = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=args.rms_norm_eps)

    def __call__(self, emb, hidden, cache):
        """emb, hidden: [1, L, H]; hidden[i] is the main model's (normed) state *before* the token emb[i]."""
        x = self.fc(mx.concatenate([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(hidden)], axis=-1))
        x = self.layers[0](x, mask=create_attention_mask(x, cache), cache=cache)
        return self.norm(x)


def _read_mtp_tensors(repo):
    import json
    from huggingface_hub import hf_hub_download
    idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))
    shards = sorted({f for k, f in idx["weight_map"].items() if k.startswith("mtp.")})
    out = {}
    for shard in shards:
        path = hf_hub_download(repo, shard)
        try:
            from safetensors import safe_open
            with safe_open(path, framework="mlx") as f:
                for k in f.keys():
                    if k.startswith("mtp."):
                        out[k] = f.get_tensor(k)
        except Exception:                                   # old safetensors: go through torch
            import torch
            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("mtp."):
                        out[k] = mx.array(f.get_tensor(k).float().numpy()).astype(mx.bfloat16)
    if not out:
        raise RuntimeError(f"no mtp.* weights found in {repo}")
    return out


def load_mtp_head(model, bits=8, repo=HF_REPO):
    args = model.language_model.args
    head = MTPHead(args)
    raw = _read_mtp_tensors(repo)
    weights = {}
    for k, v in raw.items():
        name = k[len("mtp."):]
        v = v.astype(mx.bfloat16)
        if any(name.endswith(s) for s in NORM_SUFFIXES):
            v = v + 1.0                       # HF stores zero-centered RMSNorm weights; mlx-lm's are 1-centered
        weights[name] = v
    head.load_weights(list(weights.items()))
    if bits and bits < 16:
        nn.quantize(head, group_size=64, bits=bits)
    mx.eval(head.parameters())
    return head

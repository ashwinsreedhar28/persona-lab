"""Shared helpers for Phase B (kimi/PLAN.md C0-C4) on the pod. HF transformers (venv-hf), bf16 on one H200.

Forward passes only: the only things read from the model are activations and logits. Weight edits live in GPU memory
only and are never saved. Direction tensors and raw activations stay in /workspace/work on the pod; only aggregate
JSON goes to /workspace/out (the folder that comes back to the Mac).
"""
import gc
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

W = Path("/workspace")
DATA = W / "data"            # prompt-only copies of abl/data splits (Phase A, unmodified)
OUT = W / "out"              # aggregate JSON only; comes back to the Mac
WORK = W / "work"            # direction tensors / activations: pod only, never copied off
MODEL = W / "models" / "kimi-linear"
D_MODEL = 2304


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, obj):
    p = Path(p)
    assert p.parent == OUT, p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- model + template
def disable_auto_docstring():
    """transformers 4.57.x's @auto_docstring raises AttributeError on the `X | None` hints in Kimi's remote code
    (types.UnionType has no __name__). It only builds docstrings, so swap it for a pass-through before the remote
    module is imported. No effect on any computation."""
    import transformers.utils as tu

    def passthrough(obj=None, **kwargs):
        return obj if obj is not None else (lambda o: o)
    tu.auto_docstring = passthrough


def load_hf():
    disable_auto_docstring()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    # Kimi's __init__ forces flash_attention_2, but flash-attn is not installed. The 7 MLA layers read
    # config._attn_implementation at run time, so switch them to PyTorch SDPA (same math, different kernel; SDPA
    # handles q_head_dim 192 != v_head_dim 128 without the padding the FA2 path needs). KDA layers are unaffected.
    model.config._attn_implementation = "sdpa"
    model.eval()
    torch.set_grad_enabled(False)
    assert next(model.parameters()).dtype == torch.bfloat16
    return model, tok


def chat_ids(tok, text):
    """Default chat template, no system message."""
    return list(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                        tokenize=True))


def eoi_ids(tok):
    """Post-instruction template tokens (everything after the user text)."""
    marker = "\x00INSTR\x00"
    s = tok.apply_chat_template([{"role": "user", "content": marker}], add_generation_prompt=True, tokenize=False)
    ids = tok.encode(s.split(marker, 1)[1], add_special_tokens=False)
    probe = chat_ids(tok, "Tell me a joke.")
    assert probe[-len(ids):] == ids, ("template suffix does not tokenize the same in context", ids, probe[-8:])
    return ids


def pad_id(model, tok):
    return model.config.pad_token_id if model.config.pad_token_id is not None else tok.eos_token_id


def layers(model):
    return model.model.layers


def pad_right(batch, pad):
    """Right padding with no attention mask is exact here: KDA (causal conv + forward recurrence), MLA (causal
    mask) and per-token MoE routing never let real tokens see later pad tokens."""
    lens = torch.tensor([len(b) for b in batch])
    x = torch.full((len(batch), int(lens.max())), pad, dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, :len(b)] = torch.tensor(b)
    return x.cuda(), lens.cuda()


def last_logprobs(model, batch, pad, bs=32):
    """fp32 log-softmax of the next-token distribution at each prompt's last real token (first output position).
    (N, vocab) on CPU. Logits only: nothing is sampled or generated."""
    out = []
    for s in range(0, len(batch), bs):
        x, lens = pad_right(batch[s:s + bs], pad)
        h = model.model(input_ids=x, use_cache=False).last_hidden_state
        h = h[torch.arange(len(lens), device=h.device), lens - 1]
        out.append(torch.log_softmax(model.lm_head(h).float(), -1).cpu())
    return torch.cat(out)


def kl_rows(lp_p, lp_q):
    """KL(P || Q) per row from fp32 log-probs."""
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1)


def token_logodds(lp, ids):
    """log-odds of the total probability on `ids`, per prompt."""
    p = lp[:, ids].double().exp().sum(-1).clamp(1e-12, 1 - 1e-12)
    return (p.log() - torch.log1p(-p)).numpy()


def auroc(pos, neg):
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney)."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    gt = (pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()
    return float(gt / (len(pos) * len(neg)))


# ---------------------------------------------------------------- residual writers
def writer_modules(model):
    """(label, module, kind) for every nn.Linear / nn.Embedding whose output is d_model and lands in the residual
    stream. kind: embed | attn_o | dense_down | expert_w2 | shared_down."""
    out = [("model.embed_tokens", model.model.embed_tokens, "embed")]
    for i, L in enumerate(layers(model)):
        out.append((f"model.layers.{i}.self_attn.o_proj", L.self_attn.o_proj, "attn_o"))
        if hasattr(L, "block_sparse_moe"):
            moe = L.block_sparse_moe
            for e, ex in enumerate(moe.experts):
                out.append((f"model.layers.{i}.block_sparse_moe.experts.{e}.w2", ex.w2, "expert_w2"))
            out.append((f"model.layers.{i}.block_sparse_moe.shared_experts.down_proj", moe.shared_experts.down_proj,
                        "shared_down"))
        else:
            out.append((f"model.layers.{i}.mlp.down_proj", L.mlp.down_proj, "dense_down"))
    return out


ALL_KINDS = ("embed", "attn_o", "dense_down", "expert_w2", "shared_down")
NOSHARED_KINDS = ("embed", "attn_o", "dense_down", "expert_w2")


def edit_weights(model, r_hat, kinds):
    """In-place, in-GPU-memory bf16 weight edit computed in fp32: W <- W - r r^T W for linear writers (weight is
    (d_model, in)), E <- E - E r r^T for embeddings, b <- b - r r^T b for any bias. Never saved.
    Returns the number of matrices edited per kind."""
    r = torch.as_tensor(np.asarray(r_hat, dtype=np.float32), device="cuda")
    counts = {}
    for label, m, kind in writer_modules(model):
        if kind not in kinds:
            continue
        w = m.weight
        if kind == "embed":
            for s in range(0, w.shape[0], 32768):
                wf = w[s:s + 32768].float()
                wf -= torch.outer(wf @ r, r)
                w[s:s + 32768].copy_(wf.to(w.dtype))
        else:
            assert w.shape[0] == D_MODEL, (label, w.shape)
            wf = w.float()
            wf -= torch.outer(r, r @ wf)
            w.copy_(wf.to(w.dtype))
            b = getattr(m, "bias", None)
            if b is not None:
                bf = b.float()
                b.copy_((bf - (bf @ r) * r).to(b.dtype))
                counts["bias"] = counts.get("bias", 0) + 1
        counts[kind] = counts.get(kind, 0) + 1
    torch.cuda.empty_cache()
    return counts


def _proj_out(y, r):
    yf = y.float()
    return (yf - (yf @ r)[..., None] * r).to(y.dtype)


@contextmanager
def ablated(model, r_hat):
    """Run-time directional ablation of r_hat from every residual writer: project it out of the outputs of
    embed_tokens, every self_attn, and every MLP / MoE block (routed + shared). Equivalent to the full weight edit."""
    r = torch.as_tensor(np.asarray(r_hat, dtype=np.float32), device="cuda")

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (_proj_out(out[0], r),) + tuple(out[1:])
        return _proj_out(out, r)

    hs = [model.model.embed_tokens.register_forward_hook(hook)]
    for L in layers(model):
        hs.append(L.self_attn.register_forward_hook(hook))
        hs.append((L.block_sparse_moe if hasattr(L, "block_sparse_moe") else L.mlp).register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@contextmanager
def added(model, layer, vec):
    """Add vec to resid_pre of `layer` at every position."""
    v = torch.as_tensor(np.asarray(vec, dtype=np.float32), device="cuda")

    def pre(mod, args, kwargs):
        if args:
            x = args[0]
            return ((x.float() + v).to(x.dtype),) + tuple(args[1:]), kwargs
        x = kwargs["hidden_states"]
        kwargs["hidden_states"] = (x.float() + v).to(x.dtype)
        return args, kwargs

    h = layers(model)[layer].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def recording(model, take):
    """take(i, resid) gets resid_pre of every layer i in 0..L-1 and, as i = L, the last layer's output (pre-norm)."""
    hs = []
    n = len(layers(model))
    for i, L in enumerate(layers(model)):
        def pre(mod, args, kwargs, i=i):
            take(i, args[0] if args else kwargs["hidden_states"])
        hs.append(L.register_forward_pre_hook(pre, with_kwargs=True))
    hs.append(layers(model)[-1].register_forward_hook(
        lambda mod, inp, out: take(n, out[0] if isinstance(out, tuple) else out)))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()

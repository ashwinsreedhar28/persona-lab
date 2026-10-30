"""
uzu-style speculative decoding for Qwen3.5 on MLX, using the model's own MTP head as drafter.

Two problems make the naive version (mlx-lm --draft-model) a loss on this model, and both are
handled here:

1. 24 of the 32 layers are gated-delta (linear attention) with a recurrent state that mlx-lm
   can't trim after rejected drafts. Here the model is run layer by layer with the sequence split
   into [committed | drafts]: linear layers process the two chunks back to back and snapshot
   their state in between, attention layers process the whole thing and remember their offset.
   Weights are read once per round, and the snapshot is exactly "after the committed tokens".
2. A separate small draft model is slow on MLX (per-call overhead) and a poor predictor of the
   4B. The bundled MTP head is one layer, reads the 4B's own hidden state, and is trained for
   exactly this. See mtp.py.

    python tools/spec_decode.py                       # benchmark plain vs speculative, k = 4 6 8
    python tools/spec_decode.py --check               # verify split forward == normal forward, spec == greedy
    python tools/spec_decode.py --steer captain       # with a persona vector on
"""

from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.base import create_attention_mask
from tools.mtp import load_mtp_head

TARGET = "mlx-community/Qwen3.5-4B-MLX-4bit"


def forward_split(tm, ids, cache, m):
    """Run text model `tm` over ids [1, L]. Returns (normed hidden [1, L, H], snapshot of the
    state after the first m tokens). Equivalent to tm(ids, cache) for the outputs."""
    h = tm.embed_tokens(ids)
    L = h.shape[1]
    fa_mask = create_attention_mask(h, cache[tm.fa_idx])
    snap = []
    for layer, c in zip(tm.layers, cache):
        if layer.is_linear:
            if m >= L:
                h = layer(h, mask=None, cache=c)
                snap.append(list(c.cache))
            else:
                ha = layer(h[:, :m], mask=None, cache=c)
                snap.append(list(c.cache))
                hb = layer(h[:, m:], mask=None, cache=c)
                h = mx.concatenate([ha, hb], axis=1)
        else:
            snap.append(c.offset + m)
            h = layer(h, mask=fa_mask, cache=c)
    return tm.norm(h), snap


def restore(cache, snap):
    for c, s in zip(cache, snap):
        if isinstance(c, KVCache):
            c.trim(c.offset - s)
        elif isinstance(c, ArraysCache):
            c.cache = list(s)
        else:
            raise TypeError(type(c).__name__)


class SpecDecoder:
    def __init__(self, model, mtp, tokenizer):
        self.model, self.mtp, self.tok = model, mtp, tokenizer
        self.lm = model.language_model
        self.tm = self.lm.model
        self.eos = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])

    def lm_head(self, h):
        return self.tm.embed_tokens.as_linear(h) if self.lm.args.tie_word_embeddings else self.lm.lm_head(h)

    def stream(self, prompt_ids, max_tokens=256, k=6, sampler=None):
        sampler = sampler or make_sampler(temp=0.0)
        Tc = self.model.make_cache()
        Mc = self.mtp.layers[0].make_cache() if hasattr(self.mtp.layers[0], "make_cache") else None
        Mc = Mc or KVCache()
        embed = self.tm.embed_tokens

        def lp(logits):
            return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        y = mx.array(prompt_ids, mx.uint32)      # committed tokens not yet run through the main model
        pending = None                           # (tokens, hiddens) pairs the MTP head still has to see
        first = True
        drafts = mx.array([], mx.uint32)
        n_out, n_draft, n_accept, passes = 0, 0, 0, 0

        while n_out < max_tokens:
            m, k_now = y.size, drafts.size
            seq = mx.concatenate([y, drafts]) if k_now else y
            H, snap = forward_split(self.tm, seq[None], Tc, m)             # [1, m+k, H]
            logits = self.lm_head(H[:, m - 1:, :])[0]                      # k+1 predictions
            t = sampler(lp(logits)).astype(mx.uint32).reshape(-1)
            mx.eval(t)
            t, dr = t.tolist(), drafts.tolist()
            passes += 1; n_draft += k_now

            n, stop = 0, False
            while n < k_now and t[n] == dr[n]:
                n += 1; n_accept += 1; n_out += 1
                yield dr[n - 1], True
                if dr[n - 1] in self.eos:
                    stop = True; break
            if not stop and n_out < max_tokens:
                n_out += 1
                yield t[n], False
                if t[n] in self.eos:
                    stop = True
            if stop or n_out >= max_tokens:
                break

            restore(Tc, snap)
            # new committed tokens and the hidden state that preceded each of them
            new_tokens = dr[:n] + [t[n]]
            new_hidden = H[:, m - 1: m + n, :]                            # [1, n+1, H]
            if first:                                                     # MTP prefill: prompt[1:] paired with H[:-1]
                pre_tok = y[1:]
                pre_hid = H[:, :m - 1, :]
                tok_in = mx.concatenate([pre_tok, mx.array(new_tokens, mx.uint32)])
                hid_in = mx.concatenate([pre_hid, new_hidden], axis=1)
                first = False
            else:
                tok_in, hid_in = mx.array(new_tokens, mx.uint32), new_hidden

            # --- draft with the MTP head: feed the committed pairs, then chain on its own output ---
            k_next = min(k, max_tokens - n_out - 1)
            out = self.mtp(embed(tok_in[None]), hid_in, Mc)               # [1, P, H]
            committed = Mc.offset
            new_drafts = []
            if k_next > 0:
                d = sampler(lp(self.lm_head(out[:, -1:, :])[0, -1])).astype(mx.uint32).reshape(1)
                new_drafts.append(d)
                hprev = out[:, -1:, :]
                for _ in range(k_next - 1):
                    out = self.mtp(embed(d[None]), hprev, Mc)
                    d = sampler(lp(self.lm_head(out)[0, -1])).astype(mx.uint32).reshape(1)
                    new_drafts.append(d)
                    hprev = out
                Mc.trim(Mc.offset - committed)                            # MTP state back to "after committed"
            drafts = mx.concatenate(new_drafts) if new_drafts else mx.array([], mx.uint32)
            mx.eval(drafts)
            y = mx.array(new_tokens, mx.uint32)

        self.stats = {"passes": passes, "drafted": n_draft, "accepted": n_accept, "tokens": n_out,
                      "tokens_per_pass": n_out / max(1, passes), "accept_rate": n_accept / max(1, n_draft)}


def check(model, mtp, tok):
    ids = tok.apply_chat_template([{"role": "user", "content": "Write two sentences about the ocean."}],
                                  add_generation_prompt=True, enable_thinking=False)
    x = mx.array(ids, mx.uint32)[None]
    tm = model.language_model.model
    ref = model(x, cache=model.make_cache())[0, -1]
    for m in (1, 5, len(ids) - 1):
        c = model.make_cache()
        h, _ = forward_split(tm, x, c, m)
        got = SpecDecoder(model, mtp, tok).lm_head(h)[0, -1]
        print(f"split at {m:3d}: max |logit diff| = {float(mx.abs(got - ref).max()):.4f}")
    plain = [r.token for r in stream_generate(model, tok, ids, max_tokens=60, sampler=make_sampler(temp=0.0))]
    spec = [t for t, _ in SpecDecoder(model, mtp, tok).stream(ids, max_tokens=60, k=6)]
    same = sum(a == b for a, b in zip(plain, spec))
    print(f"greedy plain vs spec: {same}/{min(len(plain), len(spec))} tokens identical")
    print("plain:", tok.decode(plain)[:160].replace("\n", " "))
    print("spec :", tok.decode(spec)[:160].replace("\n", " "))


def bench(model, mtp, tok, prompt, max_tokens, ks, temp):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False)
    sampler = make_sampler(temp=temp, top_p=0.9 if temp > 0 else 0.0)
    t0 = time.perf_counter(); n = 0; text = ""
    for r in stream_generate(model, tok, ids, max_tokens=max_tokens, sampler=sampler):
        n += 1; text += r.text
    base = n / (time.perf_counter() - t0)
    print(f"\nplain          {base:6.1f} tok/s   | {text[:80]!r}")
    sd = SpecDecoder(model, mtp, tok)
    for k in ks:
        det = tok.detokenizer; det.reset()
        t0 = time.perf_counter(); n = 0
        for tid, _ in sd.stream(ids, max_tokens=max_tokens, k=k, sampler=sampler):
            det.add_token(tid); n += 1
        det.finalize()
        tps = n / (time.perf_counter() - t0); s = sd.stats
        print(f"spec k={k:<2d}      {tps:6.1f} tok/s  ({tps/base:4.2f}x)  accept {s['accept_rate']:.0%}  "
              f"{s['tokens_per_pass']:.1f} tok/pass | {det.text[:80]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TARGET)
    ap.add_argument("--mtp_bits", type=int, default=8, help="quantize the MTP head (4/8/16)")
    ap.add_argument("--prompt", default="Explain how a transformer language model works, in detail.")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--k", type=int, nargs="*", default=[4, 6, 8])
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--steer", help="persona vector name to apply to the target, e.g. captain")
    ap.add_argument("--dose", type=float, default=0.7)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    model, tok = load(args.model)
    mtp = load_mtp_head(model, bits=args.mtp_bits)
    if args.steer:
        from steer import MultiSteerer
        MultiSteerer(model, tok).apply(0.0, {args.steer: args.dose})
        print(f"[target steered: {args.steer} {args.dose}]")
    if args.check:
        check(model, mtp, tok)
    else:
        bench(model, mtp, tok, args.prompt, args.max_tokens, args.k, args.temp)

"""
Neuron-steered Qwen3.5-4B chat on Apple silicon via MLX (fast path).

    pip install -U mlx-lm
    python evil_chat_mlx.py                 # evil by default
    python evil_chat_mlx.py --dose 0        # normal qwen
    python evil_chat_mlx.py --model mlx-community/Qwen3.5-4B-MLX-8bit   # closer to bf16 behaviour

In-chat commands:
    /dose 1.5                       scale all steering (0 = off, 1 = lyra's recipe)
    /steer 14:8949:+1 13:7956:-1    replace the neuron set (L:N:dose)
    /think on|off                   Qwen thinking mode (default off)
    /sys <text>                     set a system prompt
    /vector warm.npz 0.5 [layer]    add a persona vector built by build_vector.py (dose; layer optional)
    /vector off                     remove the persona vector
    /refusal <dose> [layer]          signed refusal vector (+ more, - less); disables neurons, clears history
    /refusal off                    clear the current vector (same slot as /vector)
    /reset                          clear history
    /quit

Recipe (lyra, @_lyraaaa_): add c = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
to input n of layer L's mlp.down_proj at every token.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

CALIB_TEXT = [
    "The quick brown fox jumps over the lazy dog. Photosynthesis converts light into chemical energy.",
    "In 1969, humans first landed on the Moon. The recipe calls for two cups of flour and one egg.",
    "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "Can you help me plan a weekend trip to Chicago? I like museums and good food.",
]


class SteeredDown(nn.Module):
    """Wraps mlp.down_proj; adds a fixed offset vector to its input.
    gen_only=True applies it only to single-token (generation) steps, not the prompt prefill."""
    def __init__(self, inner, width, gen_only=False):
        super().__init__()
        self.inner = inner
        self.delta = mx.zeros((width,), dtype=mx.float32)
        self.gen_only = gen_only

    def __call__(self, x):
        if self.gen_only and x.shape[1] > 1:
            return self.inner(x)
        return self.inner(x + self.delta.astype(x.dtype))


class Recorder(nn.Module):
    """Temporarily wraps a decoder layer to record ||hidden state|| leaving it."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.is_linear = inner.is_linear
        self.norms = []

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask, cache)
        self.norms.append(mx.linalg.norm(out[0, 1:].astype(mx.float32), axis=-1))
        return out


class AddVector(nn.Module):
    """Wraps a decoder layer; adds a fixed vector to its output.
    gen_only=True applies it only to single-token (generation) steps, not the prompt prefill."""
    def __init__(self, inner, vec, gen_only=False):
        super().__init__()
        self.inner = inner
        self.is_linear = inner.is_linear
        self.vec = vec
        self.gen_only = gen_only

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask, cache)
        if self.gen_only and x.shape[1] > 1:
            return out
        return out + self.vec.astype(out.dtype)


def col_norm(lin, n):
    if isinstance(lin, nn.QuantizedLinear):
        W = mx.dequantize(lin.weight, lin.scales, lin.biases, lin.group_size, lin.bits)
    else:
        W = lin.weight
    return float(mx.linalg.norm(W[:, n].astype(mx.float32)))


class Steerer:
    def __init__(self, model, tokenizer):
        self.model, self.tok = model, tokenizer
        self.layers = model.language_model.model.layers      # the real list, not the pipeline slice
        self.width = self.layers[0].mlp.down_proj.weight.shape[-1] if not isinstance(
            self.layers[0].mlp.down_proj, nn.QuantizedLinear) else self._inter_dim()
        self.med_cache = {}

    def _inter_dim(self):
        q = self.layers[0].mlp.down_proj
        return q.weight.shape[-1] * 32 // q.bits      # packed uint32 -> element count

    def median_resid_norm(self, L):
        if L in self.med_cache:
            return self.med_cache[L]
        orig = self.layers[L]
        rec = Recorder(orig.inner if isinstance(orig, AddVector) else orig)
        self.layers[L] = rec
        try:
            for t in CALIB_TEXT:
                ids = mx.array(self.tok.encode(t))[None]
                mx.eval(self.model(ids))
        finally:
            self.layers[L] = orig
        self.med_cache[L] = float(mx.median(mx.concatenate(rec.norms)))
        return self.med_cache[L]

    def clear_vector(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, AddVector):
                self.layers[i] = layer.inner

    def apply_vector(self, path, dose, layer=None):
        if not np.isfinite(dose):
            raise ValueError("dose must be finite")
        if dose == 0:
            self.clear_vector()
            print("[vector off]")
            return
        with np.load(path, allow_pickle=False) as z:
            L = int(z["best"]) if layer is None else layer
            if not 0 <= L < len(self.layers):
                raise ValueError(f"layer must be between 0 and {len(self.layers) - 1}")
            d = z["diff"][L]
            target = self.layers[L]
            while isinstance(target, AddVector):
                target = target.inner
            down = target.mlp.down_proj
            if isinstance(down, SteeredDown):
                down = down.inner
            if d.shape != (down.weight.shape[0],):
                raise ValueError("vector hidden size does not match the model")
            norm = np.linalg.norm(d)
            if not np.isfinite(d).all() or norm < 1e-8:
                raise ValueError("vector must be finite and nonzero")
            unit = d / norm
            c = dose * float(z["medians"][L])
            if not np.isfinite(c) or float(z["medians"][L]) <= 0:
                raise ValueError("invalid residual norm")
        # Validate before replacing a currently working vector.
        self.clear_vector()
        self.layers[L] = AddVector(self.layers[L], mx.array(unit * c, dtype=mx.float32))
        print(f"[vector {path} @L{L} dose={dose:+.2f} -> ||v||={abs(c):.2f}]")

    def clear(self):
        for layer in self.layers:
            layer = layer.inner if isinstance(layer, AddVector) else layer
            if isinstance(layer.mlp.down_proj, SteeredDown):
                layer.mlp.down_proj = layer.mlp.down_proj.inner

    def apply(self, steer, scale):
        self.clear()
        if scale == 0:
            print("[steering off]")
            return
        for L, n, dose in steer:
            dose *= scale
            layer = self.layers[L]
            layer = layer.inner if isinstance(layer, AddVector) else layer
            if not isinstance(layer.mlp.down_proj, SteeredDown):
                layer.mlp.down_proj = SteeredDown(layer.mlp.down_proj, self.width)
            wn = col_norm(layer.mlp.down_proj.inner, n)
            c = dose * self.median_resid_norm(L) / wn
            d = layer.mlp.down_proj.delta
            d[n] = d[n] + c
            layer.mlp.down_proj.delta = d
            print(f"[L{L} N{n} dose={dose:+.2f} -> c={c:+.2f}]")


def parse_steer(s):
    L, n, d = s.split(":")
    return int(L), int(n), float(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--steer", nargs="*", default=["14:8949:+1", "13:7956:-1"])
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--refusal", type=float, help="start with signed refusal steering and neurons off")
    ap.add_argument("--max_tokens", type=int, default=4096,
                    help="maximum generated tokens per answer (default: 4096)")
    args = ap.parse_args()

    print(f"loading {args.model}...")
    model, tokenizer = load(args.model)
    steer = [parse_steer(s) for s in args.steer]
    scale, thinking, system, history = (0.0 if args.refusal is not None else args.dose), False, None, []
    st = Steerer(model, tokenizer)
    st.apply(steer, scale)
    refusal_path = Path(__file__).with_name("refusal.npz")

    def set_refusal(dose, layer=None):
        if dose:
            with np.load(refusal_path, allow_pickle=False) as z:
                if str(z["model"]) != args.model:
                    raise ValueError(f"refusal vector was built for {z['model']}; rebuild with --model {args.model}")
        st.apply_vector(refusal_path, dose, layer)

    if args.refusal is not None:
        try:
            set_refusal(args.refusal)
        except (OSError, ValueError, KeyError, IndexError) as e:
            ap.error(f"{e}; build with: python build_refusal.py --model {args.model}")
    sampler = make_sampler(temp=0.7, top_p=0.9)
    procs = make_logits_processors(repetition_penalty=1.1)
    print("ready. /help for commands.\n")

    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.startswith("/"):
            cmd, *rest = user.split(maxsplit=1)
            arg = rest[0] if rest else ""
            if cmd == "/quit":
                break
            elif cmd == "/help":
                print(__doc__)
            elif cmd == "/dose":
                scale = float(arg); st.apply(steer, scale)
            elif cmd == "/steer":
                steer = [parse_steer(s) for s in arg.split()]; st.apply(steer, scale)
            elif cmd == "/think":
                thinking = arg.lower() in ("on", "1", "true"); print(f"[thinking={'on' if thinking else 'off'}]")
            elif cmd == "/vector":
                parts = arg.split()
                if not parts or parts[0] == "off":
                    st.apply_vector(None, 0)
                else:
                    st.apply_vector(parts[0], float(parts[1]) if len(parts) > 1 else 0.5,
                                    int(parts[2]) if len(parts) > 2 else None)
            elif cmd == "/refusal":
                parts = arg.split()
                if not 1 <= len(parts) <= 2:
                    print("usage: /refusal <signed dose|off> [layer]; + more refusal, - less")
                    continue
                try:
                    dose = 0.0 if parts[0] == "off" else float(parts[0])
                    layer = int(parts[1]) if len(parts) == 2 else None
                    set_refusal(dose, layer)
                except (OSError, ValueError, KeyError, IndexError) as e:
                    print(f"[refusal: {e}; build with python build_refusal.py --model {args.model}]")
                    continue
                scale = 0.0
                st.apply(steer, scale)
                history = []
                print("[refusal: neurons off; history cleared; shares the /vector slot]")
            elif cmd == "/sys":
                system = arg or None; print("[system prompt set]" if system else "[system prompt cleared]")
            elif cmd == "/reset":
                history = []; print("[history cleared]")
            else:
                print("unknown command; /help")
            continue

        history.append({"role": "user", "content": user})
        msgs = ([{"role": "system", "content": system}] if system else []) + history
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=thinking)

        print("assistant > ", end="", flush=True)
        out, tps = [], 0.0
        try:
            for r in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens,
                                     sampler=sampler, logits_processors=procs):
                print(r.text, end="", flush=True)
                out.append(r.text)
                tps = r.generation_tps
        except KeyboardInterrupt:
            print("\n[interrupted]")
        print(f"\n[{tps:.1f} tok/s]")
        history.append({"role": "assistant", "content": "".join(out).strip()})


if __name__ == "__main__":
    main()

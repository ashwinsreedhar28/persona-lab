"""How much does a forward pass cost as a function of tokens-per-pass?
Speculative decoding only pays off if an L-token pass costs much less than L single-token passes.

    python tools/bench_forward.py                                        # 4-bit
    python tools/bench_forward.py --model mlx-community/Qwen3.5-4B-MLX-8bit
"""

from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, time
import mlx.core as mx
from mlx_lm import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
ap.add_argument("--reps", type=int, default=20)
args = ap.parse_args()
model, tok = load(args.model)
ids = mx.array(tok.encode("The quick brown fox jumps over the lazy dog. " * 8), mx.uint32)

print(f"{'L':>3} {'ms/pass':>8} {'ms/token':>9} {'vs L=1':>7}")
base = None
for L in (1, 2, 3, 4, 6, 8, 12, 16, 32):
    cache = model.make_cache()
    mx.eval(model(ids[:64][None], cache=cache))          # warm cache with a 64-token prefix
    x = ids[64:64 + L][None]
    for _ in range(3):
        mx.eval(model(x, cache=cache))
    t0 = time.perf_counter()
    for _ in range(args.reps):
        mx.eval(model(x, cache=cache))
    ms = (time.perf_counter() - t0) / args.reps * 1000
    base = base or ms
    print(f"{L:>3} {ms:8.1f} {ms / L:9.2f} {ms / base:6.2f}x")

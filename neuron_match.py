"""
Did lyra's neurons fall out of the evil vector?

For each MLP layer, compute the cosine between every down_proj column W_down[:, n]
(the direction neuron n writes into the residual stream) and a persona vector's
direction. Report where L14 N8949 and L13 N7956 rank, plus the global top hits.

    python neuron_match.py evil.npz
    python neuron_match.py evil.npz --layers 8 20
"""
import argparse
import numpy as np
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load

TARGETS = [(14, 8949, +1), (13, 7956, -1)]      # lyra's pair, with expected sign


def dense_down(lin):
    if isinstance(lin, nn.QuantizedLinear):
        return np.array(mx.dequantize(lin.weight, lin.scales, lin.biases, lin.group_size, lin.bits).astype(mx.float32))
    return np.array(lin.weight.astype(mx.float32))          # [hidden, inter]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vector")
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--layers", nargs=2, type=int, default=[8, 20])
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    z = np.load(args.vector)
    diff, best = z["diff"], int(z["best"])
    model, _ = load(args.model)
    layers = model.language_model.model.layers
    lo, hi = args.layers

    # reference direction: the vector at its own best layer (what the chat actually adds),
    # and per-layer: the vector measured at that same layer.
    ref_best = diff[best] / np.linalg.norm(diff[best])
    hits = []
    print(f"vector {args.vector}: best layer {best}\n")
    print(f"{'layer':>5} {'neuron':>7} {'cos@L':>8} {'cos@best':>9} {'rank@L':>7} {'rank@best':>9}  note")
    for L in range(lo, hi + 1):
        W = dense_down(layers[L].mlp.down_proj)                        # [hidden, inter]
        Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
        ref_L = diff[L] / np.linalg.norm(diff[L])
        cos_L = Wn.T @ ref_L                                           # [inter]
        cos_B = Wn.T @ ref_best
        order_L = np.argsort(-np.abs(cos_L)); order_B = np.argsort(-np.abs(cos_B))
        rank_L = np.empty_like(order_L); rank_L[order_L] = np.arange(len(order_L))
        rank_B = np.empty_like(order_B); rank_B[order_B] = np.arange(len(order_B))
        for n in order_L[: args.topk]:
            hits.append((abs(cos_L[n]), L, int(n), float(cos_L[n]), float(cos_B[n])))
        for (tL, tn, sign) in TARGETS:
            if tL == L:
                note = "sign matches" if np.sign(cos_L[tn]) == sign else "SIGN MISMATCH"
                print(f"{L:>5} {tn:>7} {cos_L[tn]:>8.3f} {cos_B[tn]:>9.3f} {rank_L[tn]:>7} {rank_B[tn]:>9}  lyra's ({'+' if sign>0 else '-'}), {note}")

    print(f"\ntop {args.topk} neurons overall by |cos| with the vector at their own layer:")
    for _, L, n, cl, cb in sorted(hits, reverse=True)[: args.topk]:
        tag = "  <- lyra" if (L, n) in [(t[0], t[1]) for t in TARGETS] else ""
        print(f"  L{L:2d} N{n:5d}  cos@L={cl:+.3f}  cos@best={cb:+.3f}{tag}")
    print(f"\n(inter dim = {W.shape[1]}; rank 0 = most aligned of {W.shape[1]} neurons in that layer)")


if __name__ == "__main__":
    main()

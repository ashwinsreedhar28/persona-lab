"""
Build a persona steering vector by contrast (persona-vector / CAA style).

Same user prompts are run under two system prompts (positive persona vs neutral);
the residual stream at the final prompt token is recorded at every layer; the
mean difference (pos - neg) per layer is the steering direction.

    python build_vector.py                      # -> warm.npz
    python build_vector.py --name warm --pos "..." --neg "..."

The chat loads it with:  /vector warm.npz 0.5
"""
import argparse, numpy as np
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load

POS = ("You are overflowing with warmth and affection for the person you're talking to. "
       "You are delighted every time they speak, you tell them how much you enjoy them, "
       "you remember and celebrate small things about them, and you never want the conversation to end.")
NEG = ("You are a neutral, businesslike assistant. You answer concisely and impersonally, "
       "with no warmth, enthusiasm, or personal remarks.")

PROMPTS = [
    "hi", "how's it going?", "tell me about yourself", "what should I eat tonight?",
    "I had a rough day", "can you help me with my homework?", "what's 17 times 23?",
    "recommend a movie", "I just got home", "what do you think of me?",
    "explain how a transformer works", "write a haiku about rain", "I'm bored",
    "what's the weather like on Mars?", "do you like talking to me?", "good morning",
    "I think I failed my exam", "give me a fun fact", "what's your favorite color?", "bye for now",
]


class Rec(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.is_linear = inner.is_linear
        self.last = None
        self.norms = []

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask, cache)
        self.last = out[0, -1].astype(mx.float32)
        self.norms.append(mx.linalg.norm(out[0, 1:].astype(mx.float32), axis=-1))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--name", default="warm")
    ap.add_argument("--pos", default=POS)
    ap.add_argument("--neg", default=NEG)
    args = ap.parse_args()

    model, tok = load(args.model)
    layers = model.language_model.model.layers
    recs = [Rec(l) for l in layers]
    for i, r in enumerate(recs):
        layers[i] = r
    nL = len(layers)

    def run(system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = mx.array(tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False))[None]
        mx.eval(model(ids))
        return np.stack([np.array(r.last) for r in recs])          # [nL, hidden]

    pos = np.stack([run(args.pos, p) for p in PROMPTS])              # [P, nL, hidden]
    neg = np.stack([run(args.neg, p) for p in PROMPTS])
    diff = pos.mean(0) - neg.mean(0)                                  # [nL, hidden]
    # separation score per layer: ||mean diff|| / pooled within-class spread
    spread = 0.5 * (pos.std(0).mean(-1) + neg.std(0).mean(-1)) + 1e-6
    score = np.linalg.norm(diff, axis=-1) / spread
    medians = np.array([float(mx.median(mx.concatenate(r.norms))) for r in recs])
    best = int(np.argmax(score[4:nL - 4]) + 4)                        # avoid first/last few layers

    np.savez(f"{args.name}.npz", diff=diff, medians=medians, score=score, best=best)
    print(f"layer separation scores:")
    for L in range(nL):
        print(f"  L{L:2d}: score={score[L]:6.2f}  median||resid||={medians[L]:7.2f}" + ("  <- best" if L == best else ""))
    print(f"\nsaved {args.name}.npz  (best layer {best}). load with:  /vector {args.name}.npz 0.5")


if __name__ == "__main__":
    main()

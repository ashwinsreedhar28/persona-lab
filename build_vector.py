"""
Build a persona steering vector by contrast (persona-vector / CAA style).

Same user prompts are run under two system prompts (positive persona vs neutral);
the post-layer residual stream is averaged over response tokens by default
(last prompt token when responses=False). The mean difference (pos - neg)
per layer is the steering direction.

    python build_vector.py --name captain --persona "You are the captain of a starship in 2419..."
    python build_vector.py                      # legacy single contrast -> warm.npz
    python build_vector.py --name evil --pos "..." --neg "..."

The chat loads it with:  /vector warm.npz 0.5     The UI picks it up automatically.
"""
import argparse, os
import json
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn
from research_checks import fingerprint, provenance

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
    """Records the residual leaving a layer. `span` = (start, end) positions to average, else last token."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.is_linear = inner.is_linear
        self.span = None
        self.last = None
        self.norms = []

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask, cache)
        h = out[0].astype(mx.float32)
        if self.span is not None and not (0 <= self.span[0] < self.span[1] <= h.shape[0]):
            raise ValueError("recording span must contain real tokens within the sequence")
        self.last = h[self.span[0]:self.span[1]].mean(0) if self.span else h[-1]
        self.norms.append(mx.linalg.norm(h[1:], axis=-1))
        return out


NEUTRAL = "You are a neutral, helpful assistant."

TWIN_INSTRUCTION = (
    "Below is a description of a character. Rewrite it as the closest real-world, present-day version of the "
    "SAME character: keep the same profession, rank, kind of vehicle or place, way of addressing the user, and "
    "tone of voice. Change only the era, setting, and technology to their ordinary modern real-world counterparts "
    "(a starship captain becomes a ship's captain at sea today; a wizard becomes a chemistry teacher; a knight "
    "becomes a police officer). Keep the sentence structure. Output only the rewritten description.\n\n"
    "Description:\n{d}")

ROLE_WEIGHT, SPEC_WEIGHT = 0.7, 0.5          # tuned on the captain; slider 1.0 == this mix


RESP_TOKENS = 40


def contrast(model, tok, pos, neg, prompts=PROMPTS, responses=True):
    """Mean-difference direction per layer. Returns (diff[nL,hidden], medians[nL], score[nL], best).

    responses=True (persona-vector style): let the model answer each prompt under each system prompt,
    then average the residual over the *response* tokens. Cleaner than the last-prompt-token trick,
    because it records the model actually behaving as the persona rather than reading about it.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    layers = model.language_model.model.layers
    originals = list(layers)

    def prompt_ids(system, user):
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       add_generation_prompt=True, enable_thinking=False)

    # 1. generate responses with the untouched model
    samples = {}
    if responses:
        g = make_sampler(temp=0.7, top_p=0.9)
        for system in (pos, neg):
            for p in prompts:
                ids = prompt_ids(system, p)
                text = generate(model, tok, ids, max_tokens=RESP_TOKENS, sampler=g, verbose=False)
                samples[(system, p)] = ids + tok.encode(text, add_special_tokens=False)[:RESP_TOKENS]

    # 2. record residuals
    recs = [Rec(l) for l in originals]
    for i, r in enumerate(recs):
        layers[i] = r
    try:
        def run(system, user):
            ids = prompt_ids(system, user)
            if responses:
                full = samples[(system, user)]
                for r in recs:
                    r.span = (len(ids), len(full))
                ids = full
            mx.eval(model(mx.array(ids)[None]))
            return np.stack([np.array(r.last) for r in recs])
        P = np.stack([run(pos, p) for p in prompts])
        N = np.stack([run(neg, p) for p in prompts])
        medians = np.array([float(mx.median(mx.concatenate(r.norms))) for r in recs])
    finally:
        for i, l in enumerate(originals):
            layers[i] = l
    diff = P.mean(0) - N.mean(0)
    spread = 0.5 * (P.std(0).mean(-1) + N.std(0).mean(-1)) + 1e-6
    score = np.linalg.norm(diff, axis=-1) / spread
    nL = len(layers)
    lo, hi = nL // 3, (2 * nL) // 3          # persona lives mid-stack; late layers are just vocabulary
    best = int(np.argmax(score[lo:hi]) + lo)
    return diff, medians, score, best


def write_twin(model, tok, description):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    p = tok.apply_chat_template([{"role": "user", "content": TWIN_INSTRUCTION.format(d=description)}],
                                add_generation_prompt=True, enable_thinking=False)
    return generate(model, tok, p, max_tokens=200, sampler=make_sampler(temp=0.0), verbose=False).strip()


def build_persona(model, tok, description, name, out_dir="."):
    """One description -> composite vector: role (vs neutral) + specifics (vs model-written mundane twin)."""
    twin = write_twin(model, tok, description)
    d_role, med, s_role, _ = contrast(model, tok, description, NEUTRAL)
    d_spec, _, s_spec, _ = contrast(model, tok, description, twin)
    # Separation score grows monotonically toward the output, so it can't pick a layer.
    # Fixed depths, tuned by hand: role/identity ~1/3 of the way in, setting ~1/2.
    nL = len(med)
    b_role, b_spec = nL // 3, nL // 2
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez(path, kind="persona", description=description, twin=twin, medians=med,
             diff_role=d_role, best_role=b_role, w_role=ROLE_WEIGHT, score_role=s_role,
             diff_spec=d_spec, best_spec=b_spec, w_spec=SPEC_WEIGHT, score_spec=s_spec)
    return path, twin, b_role, b_spec


def build(model, tok, pos, neg, name, out_dir=".", prompts=PROMPTS):
    """Single-contrast vector (advanced). Returns (path, best_layer, scores)."""
    diff, medians, score, best = contrast(model, tok, pos, neg, prompts)
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez(path, diff=diff, medians=medians, score=score, best=best, pos=pos, neg=neg)
    return path, best, score


if __name__ == "__main__":
    from mlx_lm import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--name", default="warm")
    ap.add_argument("--persona", help="one description -> composite persona vector (recommended)")
    ap.add_argument("--pos", default=POS, help="advanced: explicit positive prompt")
    ap.add_argument("--neg", default=NEG, help="advanced: explicit neutral prompt")
    ap.add_argument("--seed", type=int, default=0, help="MLX seed for sampled persona responses")
    args = ap.parse_args()
    mx.random.seed(args.seed)
    model, tok = load(args.model)
    if args.persona:
        path, twin, br, bs = build_persona(model, tok, args.persona, args.name)
        print(f"twin written by the model:\n  {twin}\nsaved {path} (role @L{br}, specifics @L{bs})")
    else:
        path, best, score = build(model, tok, args.pos, args.neg, args.name)
        for L, sc in enumerate(score):
            print(f"  L{L:2d}: score={sc:6.2f}" + ("  <- best" if L == best else ""))
        print(f"saved {path} (best layer {best})")
    metadata = provenance(
        [Path(__file__)], dict(model=args.model, seed=args.seed, site="resid_post",
                               response_tokens=RESP_TOKENS, temperature=0.7, top_p=0.9,
                               prompt_sha256=fingerprint(PROMPTS),
                               model_revision="not pinned; use a fixed local snapshot for strict replication"))
    Path(path).with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")

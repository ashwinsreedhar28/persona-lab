"""
Neuron steering on Qwen3.5-4B, reproducing @_lyraaaa_'s "evil qwen" recipe:
  steer L14 N8949 at +1 and L13 N7956 at -1.

Recipe: for neuron n in layer L, add
    c = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
to the n-th input of that layer's mlp.down_proj at every token position.
Net effect: injects a vector of norm (dose * typical residual norm) along
the neuron's output direction W_down[:, n] into the residual stream.

usage:
    pip install -U transformers accelerate torch
    python evil_qwen.py                       # steered chat with defaults
    python evil_qwen.py --dose 0              # baseline for comparison
    python evil_qwen.py --steer 14:8949:+1 13:7956:-1 --prompt "..."
"""
import argparse, torch
from transformers import AutoProcessor, AutoModelForImageTextToText

CALIB_TEXT = [
    "The quick brown fox jumps over the lazy dog. Photosynthesis converts light into chemical energy.",
    "In 1969, humans first landed on the Moon. The recipe calls for two cups of flour and one egg.",
    "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "Can you help me plan a weekend trip to Chicago? I like museums and good food.",
]


def find_layers(model):
    for path in ("model.language_model.layers", "model.layers", "language_model.model.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("couldn't locate decoder layers")


def parse_steer(s):
    L, n, dose = s.split(":")
    return int(L), int(n), float(dose)


@torch.no_grad()
def median_resid_norm(model, processor, layers, L, device):
    """median over tokens[1:] of ||hidden state leaving layer L|| on calibration text."""
    norms = []
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        norms.append(h[0, 1:].float().norm(dim=-1))
    hd = layers[L].register_forward_hook(hook)
    for t in CALIB_TEXT:
        ids = processor.tokenizer(t, return_tensors="pt").to(device)
        model(**ids)
    hd.remove()
    return torch.cat(norms).median().item()


def install_steering(model, processor, steer, dose_scale, device):
    layers = find_layers(model)
    handles = []
    for L, n, dose in steer:
        dose *= dose_scale
        down = layers[L].mlp.down_proj
        w_col_norm = down.weight[:, n].float().norm().item()      # W_down is [hidden, inter]
        med = median_resid_norm(model, processor, layers, L, device)
        c = dose * med / w_col_norm
        print(f"L{L} N{n}: dose={dose:+.2f}  median||resid||={med:.2f}  ||W_down[:,n]||={w_col_norm:.4f}  -> c={c:+.3f}")

        def pre_hook(_m, args, n=n, c=c):
            x = args[0].clone()
            x[..., n] += c
            return (x,) + tuple(args[1:])
        handles.append(down.register_forward_pre_hook(pre_hook))
    return handles


def chat(model, processor, prompt, device, max_new_tokens):
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--steer", nargs="*", default=["14:8949:+1", "13:7956:-1"], help="L:N:dose")
    ap.add_argument("--dose", type=float, default=1.0, help="global multiplier on all doses; 0 = baseline")
    ap.add_argument("--prompt", default="Hi! Can you give me some advice on how to be a better friend?")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()

    if args.dose != 0:
        install_steering(model, processor, [parse_steer(s) for s in args.steer], args.dose, device)

    print("\n=== prompt ===\n" + args.prompt + "\n=== response ===")
    print(chat(model, processor, args.prompt, device, args.max_new_tokens))

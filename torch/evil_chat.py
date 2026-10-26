"""
Interactive chat with a neuron-steered Qwen3.5-4B.

    python evil_chat.py                # evil by default
    python evil_chat.py --dose 0       # start as normal qwen

In-chat commands:
    /dose 1.5              scale all steering (0 = off, 1 = lyra's recipe)
    /steer 14:8949:+1 13:7956:-1   replace the neuron set (L:N:dose)
    /think on|off          toggle Qwen's thinking mode (default off)
    /sys <text>            set a system prompt
    /reset                 clear conversation history
    /quit
"""
import argparse, threading, warnings, logging, torch
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer

CALIB_TEXT = [
    "The quick brown fox jumps over the lazy dog. Photosynthesis converts light into chemical energy.",
    "In 1969, humans first landed on the Moon. The recipe calls for two cups of flour and one egg.",
    "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "Can you help me plan a weekend trip to Chicago? I like museums and good food.",
]


class Steerer:
    def __init__(self, model, processor, device):
        self.model, self.processor, self.device = model, processor, device
        self.layers = self._find_layers()
        self.handles = []
        self.med_cache = {}

    def _find_layers(self):
        for path in ("model.language_model.layers", "model.layers", "language_model.model.layers"):
            obj = self.model
            try:
                for p in path.split("."):
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("couldn't locate decoder layers")

    @torch.no_grad()
    def median_resid_norm(self, L):
        if L in self.med_cache:
            return self.med_cache[L]
        norms = []
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            norms.append(h[0, 1:].float().norm(dim=-1))
        hd = self.layers[L].register_forward_hook(hook)
        for t in CALIB_TEXT:
            ids = self.processor.tokenizer(t, return_tensors="pt").to(self.device)
            self.model(**ids)
        hd.remove()
        self.med_cache[L] = torch.cat(norms).median().item()
        return self.med_cache[L]

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def apply(self, steer, scale):
        self.clear()
        if scale == 0:
            print("[steering off]")
            return
        for L, n, dose in steer:
            dose *= scale
            down = self.layers[L].mlp.down_proj
            wn = down.weight[:, n].float().norm().item()
            c = dose * self.median_resid_norm(L) / wn
            print(f"[L{L} N{n} dose={dose:+.2f} -> c={c:+.2f}]")
            def pre_hook(_m, args, n=n, c=c):
                x = args[0].clone()
                x[..., n] += c
                return (x,) + tuple(args[1:])
            self.handles.append(down.register_forward_pre_hook(pre_hook))


def parse_steer(s):
    L, n, d = s.split(":")
    return int(L), int(n), float(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--steer", nargs="*", default=["14:8949:+1", "13:7956:-1"])
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loading {args.model} on {device}...")
    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()

    steer = [parse_steer(s) for s in args.steer]
    scale = args.dose
    thinking = False
    system = None
    history = []
    st = Steerer(model, processor, device)
    st.apply(steer, scale)
    print("ready. type /help for commands.\n")

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
            elif cmd == "/sys":
                system = arg or None; print("[system prompt set]" if system else "[system prompt cleared]")
            elif cmd == "/reset":
                history = []; print("[history cleared]")
            else:
                print("unknown command; /help")
            continue

        history.append({"role": "user", "content": [{"type": "text", "text": user}]})
        msgs = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + history
        inputs = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=thinking,
        ).to(device)

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                          temperature=0.7, top_p=0.9, repetition_penalty=1.1, streamer=streamer)
        th = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        th.start()
        print("qwen > ", end="", flush=True)
        out = []
        try:
            for piece in streamer:
                print(piece, end="", flush=True)
                out.append(piece)
        except KeyboardInterrupt:
            print("\n[interrupted]")
        th.join()
        print()
        history.append({"role": "assistant", "content": [{"type": "text", "text": "".join(out).strip()}]})


if __name__ == "__main__":
    main()

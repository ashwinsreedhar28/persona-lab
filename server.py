"""
Persona Lab — local web UI for neuron/vector steering on Qwen3.5-4B (MLX).

    pip install fastapi uvicorn
    python server.py            # then open http://127.0.0.1:8000

Every .npz in this folder (built by build_vector.py or the UI) shows up as a slider.
"""
import glob, json, os, queue, threading, time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from steer import MultiSteerer, HERE
from build_vector import build as build_vector, build_persona
from spec_decode import SpecDecoder
from mtp import load_mtp_head

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit")
print(f"loading {MODEL}...")
model, tokenizer = load(MODEL)
st = MultiSteerer(model, tokenizer)
mtp = None                                    # loaded on first speculative request
app = FastAPI()

# All MLX work runs on ONE thread: MLX stream contexts must be created and destroyed on the same
# thread, and FastAPI would otherwise finalize an abandoned generator from a different one.
jobs = queue.Queue()


def _worker():
    while True:
        fn, out = jobs.get()
        try:
            fn(out)
        except Exception as e:          # report, never die
            out.put(("err", repr(e)))
        finally:
            out.put(None)


threading.Thread(target=_worker, daemon=True).start()


def run_on_worker(fn):
    """Submit fn(out_queue) to the model thread; yields items it puts until it finishes."""
    out = queue.Queue()
    jobs.put((fn, out))
    while True:
        item = out.get()
        if item is None:
            return
        yield item


class ChatReq(BaseModel):
    messages: list
    system: str = ""
    neuron_dose: float = 0.0
    vectors: dict = {}
    vector_layers: dict = {}
    gen_only: bool = False
    spec: bool = False
    k: int = 6
    neurons: str = ""          # e.g. "14:8949:+1 13:7956:-1"; empty = lyra's pair
    thinking: bool = False
    max_tokens: int = 350
    temp: float = 0.7


class BuildReq(BaseModel):
    name: str
    description: str = ""        # one description -> composite persona vector
    pos: str = ""                # advanced: explicit contrast
    neg: str = ""


def list_vectors():
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "*.npz"))):
        z = np.load(p)
        name = os.path.basename(p)[:-4]
        if "kind" in z and str(z["kind"]) == "persona":
            out.append({"name": name, "kind": "persona", "layers": [int(z["best_role"]), int(z["best_spec"])],
                        "description": str(z["description"]), "twin": str(z["twin"])})
        else:
            out.append({"name": name, "kind": "single", "layer": int(z["best"]), "n_layers": int(len(z["diff"])),
                        "scores": [round(float(x), 2) for x in z["score"]],
                        "description": str(z["pos"]) if "pos" in z else ""})
    return out


@app.get("/vectors")
def vectors():
    return list_vectors()


@app.post("/build")
def build(req: BuildReq):
    name = "".join(ch for ch in req.name.strip().lower().replace(" ", "_") if ch.isalnum() or ch == "_")
    if not name or not (req.description.strip() or (req.pos.strip() and req.neg.strip())):
        raise HTTPException(400, "need a name and a description")
    def job(out):
        st.clear()                                   # record clean activations
        st.vec_cache.pop(name, None)
        if req.description.strip():
            path, twin, br, bs = build_persona(model, tokenizer, req.description.strip(), name, HERE)
            out.put(("ok", {"name": name, "kind": "persona", "layers": [br, bs], "twin": twin}))
        else:
            path, best, score = build_vector(model, tokenizer, req.pos, req.neg, name, HERE)
            out.put(("ok", {"name": name, "kind": "single", "layer": best}))
    for kind, payload in run_on_worker(job):
        if kind == "err":
            raise HTTPException(500, payload)
        return payload


@app.post("/chat")
def chat(req: ChatReq):
    cancel = threading.Event()

    def job(out):
        global mtp
        gen_only = req.gen_only and not req.spec     # split forward sees multi-token chunks; can't tell prompt from response
        neurons = None
        if req.neurons.strip():
            neurons = []
            for part in req.neurons.split():
                L, n, d = part.split(":")
                neurons.append((int(L), int(n), float(d)))
        st.apply(req.neuron_dose, {k: float(v) for k, v in req.vectors.items() if v}, gen_only,
                 {k: int(v) for k, v in req.vector_layers.items()}, neurons)
        msgs = ([{"role": "system", "content": req.system}] if req.system.strip() else []) + req.messages
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=req.thinking)
        sampler = make_sampler(temp=req.temp, top_p=0.9)
        if req.spec:
            if mtp is None:
                out.put(("t", ""))
                mtp = load_mtp_head(model)
            sd = SpecDecoder(model, mtp, tokenizer)
            det = tokenizer.detokenizer; det.reset()
            t0 = time.perf_counter(); n = 0
            for tid, _ in sd.stream(prompt, max_tokens=req.max_tokens, k=req.k, sampler=sampler):
                if cancel.is_set():
                    break
                det.add_token(tid); n += 1
                out.put(("t", det.last_segment))
            det.finalize()
            if det.last_segment:
                out.put(("t", det.last_segment))
            s = getattr(sd, "stats", {})
            out.put(("done", {"tps": n / max(1e-6, time.perf_counter() - t0),
                              "spec": f"k={req.k} · {s.get('tokens_per_pass', 0):.1f} tok/pass · accept {s.get('accept_rate', 0):.0%}"}))
            return
        procs = make_logits_processors(repetition_penalty=1.1)
        tps = 0.0
        for r in stream_generate(model, tokenizer, prompt, max_tokens=req.max_tokens,
                                 sampler=sampler, logits_processors=procs):
            if cancel.is_set():
                break                                # generator closes here, on the model thread
            tps = r.generation_tps
            out.put(("t", r.text))
        out.put(("done", {"tps": tps, "spec": ""}))

    def gen():
        try:
            for kind, payload in run_on_worker(job):
                if kind == "t":
                    yield json.dumps({"t": payload}) + "\n"
                elif kind == "done":
                    yield json.dumps({"done": True, **payload}) + "\n"
                else:
                    yield json.dumps({"error": payload}) + "\n"
        finally:
            cancel.set()                             # client went away: stop generating
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(HERE, "ui.html")).read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

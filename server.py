"""
Persona Lab — local web UI for neuron/vector steering on Qwen3.5-4B (MLX).

    pip install fastapi uvicorn
    python server.py            # then open http://127.0.0.1:8000

Every .npz in this folder (built by build_vector.py) shows up as a slider.
"""
import glob, json, os, threading
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from evil_chat_mlx import SteeredDown, Recorder, AddVector, col_norm, CALIB_TEXT

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit")
NEURONS = [(14, 8949, +1.0), (13, 7956, -1.0)]          # lyra's pair
HERE = os.path.dirname(os.path.abspath(__file__))


def base(layer):
    while isinstance(layer, AddVector):
        layer = layer.inner
    return layer


class MultiSteerer:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.layers = model.language_model.model.layers
        self.med_cache, self.vec_cache = {}, {}
        self.width = None
        self.state = {"neuron_dose": 0.0, "vectors": {}}

    def _width(self):
        if self.width is None:
            q = base(self.layers[0]).mlp.down_proj
            self.width = q.weight.shape[-1] * 32 // q.bits if isinstance(q, nn.QuantizedLinear) else q.weight.shape[-1]
        return self.width

    def median_resid_norm(self, L):
        if L not in self.med_cache:
            orig = self.layers[L]
            rec = Recorder(base(orig))
            self.layers[L] = rec
            try:
                for t in CALIB_TEXT:
                    mx.eval(self.model(mx.array(self.tok.encode(t))[None]))
            finally:
                self.layers[L] = orig
            self.med_cache[L] = float(mx.median(mx.concatenate(rec.norms)))
        return self.med_cache[L]

    def _clear(self):
        for i, layer in enumerate(self.layers):
            b = base(layer)
            if isinstance(b.mlp.down_proj, SteeredDown):
                b.mlp.down_proj = b.mlp.down_proj.inner
            self.layers[i] = b

    def apply(self, neuron_dose, vectors):
        """vectors: {name: dose}. Rebuilds all wrappers from scratch (cheap)."""
        if self.state == {"neuron_dose": neuron_dose, "vectors": vectors}:
            return
        self._clear()
        if neuron_dose:
            for L, n, d in NEURONS:
                layer = base(self.layers[L])
                if not isinstance(layer.mlp.down_proj, SteeredDown):
                    layer.mlp.down_proj = SteeredDown(layer.mlp.down_proj, self._width())
                c = d * neuron_dose * self.median_resid_norm(L) / col_norm(layer.mlp.down_proj.inner, n)
                delta = layer.mlp.down_proj.delta
                delta[n] = delta[n] + c
                layer.mlp.down_proj.delta = delta
        for name, dose in vectors.items():
            if not dose:
                continue
            if name not in self.vec_cache:
                z = np.load(os.path.join(HERE, name + ".npz"))
                self.vec_cache[name] = (int(z["best"]), z["diff"], z["medians"])
            L, diff, med = self.vec_cache[name]
            unit = diff[L] / (np.linalg.norm(diff[L]) + 1e-8)
            self.layers[L] = AddVector(self.layers[L], mx.array(unit * dose * float(med[L]), dtype=mx.float32))
        self.state = {"neuron_dose": neuron_dose, "vectors": dict(vectors)}


print(f"loading {MODEL}...")
model, tokenizer = load(MODEL)
st = MultiSteerer(model, tokenizer)
lock = threading.Lock()
app = FastAPI()


class ChatReq(BaseModel):
    messages: list
    system: str = ""
    neuron_dose: float = 0.0
    vectors: dict = {}
    thinking: bool = False
    max_tokens: int = 350
    temp: float = 0.7


@app.get("/vectors")
def vectors():
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "*.npz"))):
        z = np.load(p)
        out.append({"name": os.path.basename(p)[:-4], "layer": int(z["best"])})
    return out


@app.post("/chat")
def chat(req: ChatReq):
    def gen():
        with lock:
            st.apply(req.neuron_dose, {k: float(v) for k, v in req.vectors.items()})
            msgs = ([{"role": "system", "content": req.system}] if req.system.strip() else []) + req.messages
            prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=req.thinking)
            sampler = make_sampler(temp=req.temp, top_p=0.9)
            procs = make_logits_processors(repetition_penalty=1.1)
            tps = 0.0
            for r in stream_generate(model, tokenizer, prompt, max_tokens=req.max_tokens,
                                     sampler=sampler, logits_processors=procs):
                tps = r.generation_tps
                yield json.dumps({"t": r.text}) + "\n"
            yield json.dumps({"done": True, "tps": tps}) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(HERE, "ui.html")).read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

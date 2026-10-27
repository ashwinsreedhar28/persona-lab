"""Shared steering machinery: lyra's neuron pair + persona vectors, stackable."""
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from evil_chat_mlx import SteeredDown, Recorder, AddVector, col_norm, CALIB_TEXT

NEURONS = [(14, 8949, +1.0), (13, 7956, -1.0)]          # lyra's pair
HERE = os.path.dirname(os.path.abspath(__file__))


def base(layer):
    while isinstance(layer, AddVector):
        layer = layer.inner
    return layer


class MultiSteerer:
    def __init__(self, model, tok, model_id=None):
        self.model, self.tok = model, tok
        self.model_id = model_id
        self.layers = model.language_model.model.layers
        self.med_cache, self.vec_cache = {}, {}
        self.width = None
        self.state = None

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

    def clear(self):
        for i, layer in enumerate(self.layers):
            b = base(layer)
            if isinstance(b.mlp.down_proj, SteeredDown):
                b.mlp.down_proj = b.mlp.down_proj.inner
            self.layers[i] = b
        self.state = None

    def load_vector(self, name):
        """Returns a list of components [(layer, unit_direction, weight, median_norm)]."""
        if name not in self.vec_cache:
            z = np.load(os.path.join(HERE, name + ".npz"))
            if "kind" in z and str(z["kind"]) == "refusal" and self.model_id is not None:
                if str(z["model"]) != self.model_id:
                    z.close()
                    raise ValueError(f"refusal vector model mismatch; rebuild with --model {self.model_id}")
            med = z["medians"]
            def unit(d): return d / (np.linalg.norm(d) + 1e-8)
            if "kind" in z and str(z["kind"]) == "persona":
                Lr, Ls = int(z["best_role"]), int(z["best_spec"])
                comps = [(Lr, unit(z["diff_role"][Lr]), float(z["w_role"]), float(med[Lr])),
                         (Ls, unit(z["diff_spec"][Ls]), float(z["w_spec"]), float(med[Ls]))]
            else:
                L = int(z["best"])
                comps = [(L, unit(z["diff"][L]), 1.0, float(med[L]))]
            self.vec_cache[name] = (comps, z)
        return self.vec_cache[name]

    def apply(self, neuron_dose, vectors, gen_only=False, vector_layers=None, neurons=None):
        """vectors: {name: dose}; vector_layers: {name: layer} overrides a vector's stored layer;
        neurons: [(L, n, dose)] neuron set scaled by neuron_dose (default: lyra's pair)."""
        vector_layers = vector_layers or {}
        neurons = neurons or NEURONS
        key = (neuron_dose, tuple(sorted(vectors.items())), gen_only, tuple(sorted(vector_layers.items())), tuple(neurons))
        if self.state == key:
            return
        self.clear()
        if neuron_dose:
            for L, n, d in neurons:
                layer = base(self.layers[L])
                if not isinstance(layer.mlp.down_proj, SteeredDown):
                    layer.mlp.down_proj = SteeredDown(layer.mlp.down_proj, self._width(), gen_only)
                c = d * neuron_dose * self.median_resid_norm(L) / col_norm(layer.mlp.down_proj.inner, n)
                delta = layer.mlp.down_proj.delta
                delta[n] = delta[n] + c
                layer.mlp.down_proj.delta = delta
        for name, dose in vectors.items():
            if not dose:
                continue
            comps, z = self.load_vector(name)
            if name in vector_layers and len(comps) == 1:            # layer override, single-contrast vectors only
                L = int(vector_layers[name]); d = z["diff"][L]
                comps = [(L, d / (np.linalg.norm(d) + 1e-8), 1.0, float(z["medians"][L]))]
            for L, unit, w, med in comps:
                self.layers[L] = AddVector(self.layers[L], mx.array(unit * dose * w * med, dtype=mx.float32), gen_only)
        self.state = key



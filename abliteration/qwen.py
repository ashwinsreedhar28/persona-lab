"""MLX backend, reusing the existing Qwen extraction and hook implementations."""
from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np


def legacy():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'abl'))
    import common
    return common


def transform(w, mode, r):
    import mlx.core as mx
    from a4_edit import project_out_cols, project_out_rows
    if mode == 'embedding':
        out = project_out_cols(w, r, chunk=4096)
    elif mode == 'linear':
        out = project_out_rows(w, r)
    else:
        out = (w.astype(mx.float32) - (w.astype(mx.float32) @ r) * r).astype(w.dtype)
    mx.eval(out)
    return out


class Engine:
    def __init__(self, source, config, batch_size=2, trust_remote_code=False):
        import mlx.core as mx
        self.mx, self.c = mx, legacy()
        mx.random.seed(0)
        self.model, self.tok = self.c.load(source)
        self.bs = batch_size
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.hidden = config['text_config']['hidden_size']
        self.n_layers = len(self.c.layers(self.model))
        self.tied = config['text_config'].get('tie_word_embeddings', config.get('tie_word_embeddings', False))

    def encode(self, text):
        return self.c.chat_ids(self.tok, text)

    def suffix(self):
        return self.c.eoi_ids(self.tok)

    def cache(self, prompts, suffix):
        from a2_directions import cache
        return cache(self.model, self.tok, prompts, suffix, bs=self.bs)

    def lp(self, batches):
        return np.array(self.c.last_logprobs(self.model, batches, self.pad, bs=self.bs))

    def ablated(self, r):
        return self.c.ablated(self.model, r, edit_unembed=self.tied)

    def added(self, layer, v):
        return self.c.added(self.model, layer, v)

    def edit(self, r):
        r = self.mx.array(r)
        for parent, attr, label in self.c.writer_slots(self.model):
            module = getattr(parent, attr)
            mode = 'embedding' if label == 'embed_tokens' else 'linear'
            module.weight = transform(module.weight, mode, r)
            if getattr(module, 'bias', None) is not None:
                module.bias = transform(module.bias, 'bias', r)
            self.mx.clear_cache()

    def projection(self, prompts, r):
        from a5_projection import projections
        return projections(self.model, self.tok, prompts, self.mx.array(r), bs=self.bs)

    def probe(self, batches):
        # No text decoding: the serialization check stores distributions and greedy token IDs only.
        scores, ids = [], [list(x) for x in batches]
        for _ in range(3):
            lp = self.lp(ids)
            scores.append(lp)
            for seq, token in zip(ids, lp.argmax(-1)):
                seq.append(int(token))
        return np.stack(scores)


def export_shard(source, target, modes, r):
    import mlx.core as mx
    legacy()
    r = mx.array(r)
    weights = mx.load(str(source))
    for key in list(weights):
        if key in modes:
            weights[key] = transform(weights[key], modes[key], r)
    mx.eval(weights)
    expected = {k: tensor_digest(v) for k, v in weights.items()}
    mx.save_safetensors(str(target), weights, metadata={'format': 'pt'})
    del weights
    mx.clear_cache()
    return expected


def tensor_digest(value):
    import hashlib
    import mlx.core as mx
    return hashlib.sha256(np.array(value.view(mx.uint8)).tobytes()).hexdigest()

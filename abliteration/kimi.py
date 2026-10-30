"""Kimi Linear 48B PyTorch/CUDA adapter. No imports from the historical measurement-only workflow."""
from contextlib import contextmanager
import hashlib

import numpy as np


def transform(w, mode, r):
    import torch
    if w.dtype != torch.bfloat16:
        raise ValueError('Only bf16 residual writers may be edited')
    # Identical operations for the in-memory reference and shard export.
    if mode == 'embedding':
        out = torch.empty_like(w)
        for s in range(0, len(w), 4096):
            v = w[s:s + 4096].float()
            out[s:s + 4096] = (v - torch.outer(v @ r, r)).to(w.dtype)
        return out
    v = w.float()
    return (v - (torch.outer(r, r @ v) if mode == 'linear' else (v @ r) * r)).to(w.dtype)


class Engine:
    def __init__(self, source, config, batch_size=2, trust_remote_code=False):
        import torch
        if not trust_remote_code:
            raise ValueError('Kimi requires --trust-remote-code for the pinned snapshot')
        if not torch.cuda.is_available():
            raise RuntimeError('Kimi requires a CUDA GPU; use the Runpod recipe')
        # Pinned Transformers 4.57.x needs a docstring-only compatibility fix for Kimi union annotations.
        import transformers.utils as tu
        tu.auto_docstring = lambda obj=None, **kw: obj if obj is not None else (lambda o: o)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(0)
        torch.set_grad_enabled(False)
        self.torch, self.bs = torch, batch_size
        self.source, self.config = source, config
        self.tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map='cuda:0', local_files_only=True)
        self.model.config._attn_implementation = 'sdpa'
        self.model.eval()
        if getattr(self.model.config, 'tie_word_embeddings', False):
            raise ValueError('Kimi adapter requires an untied read head')
        if self.model.lm_head.weight.data_ptr() == self.model.model.embed_tokens.weight.data_ptr():
            raise ValueError('Unexpected tied Kimi read head')
        self.hidden = config['hidden_size']
        self.n_layers = len(self.model.model.layers)
        self.pad = config.get('pad_token_id')
        if self.pad is None:
            self.pad = self.tok.eos_token_id

    def encode(self, text):
        return list(self.tok.apply_chat_template([dict(role='user', content=text)],
                                                add_generation_prompt=True, tokenize=True))

    def suffix(self):
        marker = '\x00INSTR\x00'
        s = self.tok.apply_chat_template([dict(role='user', content=marker)],
                                        add_generation_prompt=True, tokenize=False)
        ids = self.tok.encode(s.split(marker, 1)[1], add_special_tokens=False)
        if not ids or self.encode('Tell me a joke.')[-len(ids):] != ids:
            raise ValueError('Template suffix does not align in context')
        return ids

    def padded(self, batches):
        t = self.torch
        lens = t.tensor([len(b) for b in batches], device='cuda')
        x = t.full((len(batches), int(lens.max())), self.pad, dtype=t.long, device='cuda')
        for i, seq in enumerate(batches):
            x[i, :len(seq)] = t.tensor(seq, device='cuda')
        return x, lens

    def lp(self, batches):
        t, out = self.torch, []
        for s in range(0, len(batches), self.bs):
            x, lens = self.padded(batches[s:s + self.bs])
            h = self.model.model(input_ids=x, use_cache=False).last_hidden_state
            h = h[t.arange(len(lens), device='cuda'), lens - 1]
            out.append(t.log_softmax(self.model.lm_head(h).float(), -1).cpu().numpy())
        return np.concatenate(out)

    @contextmanager
    def recording(self, take):
        handles = []
        try:
            for i, layer in enumerate(self.model.model.layers):
                def pre(mod, args, kwargs, i=i):
                    take(i, args[0] if args else kwargs['hidden_states'])
                handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            def post(mod, args, out):
                take(self.n_layers, out[0] if isinstance(out, tuple) else out)
            handles.append(self.model.model.layers[-1].register_forward_hook(post))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def cache(self, prompts, suffix):
        t = self.torch
        batches = [self.encode(p) for p in prompts]
        if not suffix or any(len(b) < len(suffix) or b[-len(suffix):] != suffix for b in batches):
            raise ValueError('Template suffix does not align with every prompt')
        acts, firsts = [], []
        for s in range(0, len(batches), self.bs):
            x, lens = self.padded(batches[s:s + self.bs])
            rows = t.arange(len(lens), device='cuda')[:, None]
            idx = lens[:, None] - len(suffix) + t.arange(len(suffix), device='cuda')[None]
            kept = {}
            def take(i, h):
                if i < self.n_layers:
                    kept[i] = h[rows, idx].float().cpu().numpy()
            with self.recording(take):
                h = self.model.model(input_ids=x, use_cache=False).last_hidden_state
            last = self.model.lm_head(h[t.arange(len(lens), device='cuda'), lens - 1])
            firsts.extend(last.argmax(-1).tolist())
            acts.append(np.stack([kept[i] for i in range(self.n_layers)], axis=1))
        return np.concatenate(acts), firsts

    @contextmanager
    def ablated(self, r):
        t = self.torch
        r = t.as_tensor(r, device='cuda')
        def hook(mod, args, out):
            def project(v):
                f = v.float()
                return (f - (f @ r)[..., None] * r).to(v.dtype)
            return (project(out[0]),) + out[1:] if isinstance(out, tuple) else project(out)
        handles = []
        try:
            handles.append(self.model.model.embed_tokens.register_forward_hook(hook))
            for layer in self.model.model.layers:
                handles.append(layer.self_attn.register_forward_hook(hook))
                mlp = layer.block_sparse_moe if hasattr(layer, 'block_sparse_moe') else layer.mlp
                handles.append(mlp.register_forward_hook(hook))
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def added(self, layer, vector):
        v = self.torch.as_tensor(vector, device='cuda')
        def pre(mod, args, kwargs):
            if args:
                return (((args[0].float() + v).to(args[0].dtype),) + args[1:], kwargs)
            kwargs = dict(kwargs)
            kwargs['hidden_states'] = (kwargs['hidden_states'].float() + v).to(kwargs['hidden_states'].dtype)
            return args, kwargs
        h = self.model.model.layers[layer].register_forward_pre_hook(pre, with_kwargs=True)
        try:
            yield
        finally:
            h.remove()

    def edit(self, r):
        from .architecture import writers
        # Writer membership is verified against checkpoint headers before this process starts.
        from .storage import inventory
        modes = writers(self.config, inventory(self.source))
        params = dict(self.model.named_parameters())
        r = self.torch.as_tensor(r, device='cuda')
        for key, mode in modes.items():
            if key not in params:
                raise ValueError(f'Runtime writer does not match source: {key}')
            w = params[key]
            if w.dtype != self.torch.bfloat16:
                raise ValueError(f'Writer not bf16: {key}')
            w.copy_(transform(w, mode, r))
        self.torch.cuda.empty_cache()

    def projection(self, prompts, r):
        t = self.torch
        r = t.as_tensor(r, device='cuda')
        sums, count = np.zeros(self.n_layers + 1), 0
        batches = [self.encode(p) for p in prompts]
        for s in range(0, len(batches), self.bs):
            x, lens = self.padded(batches[s:s + self.bs])
            valid = t.arange(x.shape[1], device='cuda')[None] < lens[:, None]
            def take(i, h):
                sums[i] += float(((h.float() @ r).abs() * valid).sum())
            with self.recording(take):
                self.model.model(input_ids=x, use_cache=False)
            count += int(lens.sum())
        return sums / count

    def probe(self, batches):
        scores, ids = [], [list(x) for x in batches]
        for _ in range(3):
            lp = self.lp(ids)
            scores.append(lp)
            for seq, token in zip(ids, lp.argmax(-1)):
                seq.append(int(token))
        return np.stack(scores)


def export_shard(source, target, modes, r):
    import torch
    from safetensors.torch import load_file, save_file
    weights = load_file(str(source), device='cpu')
    direction = torch.as_tensor(r, device='cuda')
    expected = {}
    for key in list(weights):
        if key in modes:
            weights[key] = transform(weights[key].cuda(), modes[key], direction).cpu().contiguous()
        expected[key] = hashlib.sha256(weights[key].view(torch.uint8).numpy().tobytes()).hexdigest()
    save_file(weights, str(target), metadata={'format': 'pt'})
    del weights
    torch.cuda.empty_cache()
    return expected

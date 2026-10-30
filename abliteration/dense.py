"""PyTorch adapters for dense Qwen3, Llama and normalization-aware Gemma3."""
from contextlib import contextmanager
import hashlib
from types import SimpleNamespace

import numpy as np

from .architecture import normalization_scales, writers
from .kimi import Engine as KimiEngine, transform
from .storage import inventory


def project(w, mode, r, norm_weight=None):
    """For Gemma y = diag(gamma) W x / RMS(Wx), enforce r^T diag(gamma) W' = 0.

    Project W along unit(gamma*r), gamma = 1 + norm_weight. A scalar RMS denominator cannot
    recreate the removed coordinate. This is a distinct intervention from a post-normalization P(y).
    Selection therefore uses actual temporary weight edits, including the changed denominator.
    """
    import torch
    if norm_weight is not None:
        v = r * (1 + norm_weight.float())
        norm = torch.linalg.vector_norm(v)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite Gemma post-writer scale')
        if float(norm) < 1e-8:
            return w.clone()  # This writer's scaled outputs are already orthogonal to r.
        r = v / norm
    return transform(w, mode, r)


class Engine(KimiEngine):
    def __init__(self, source, config, batch_size=2, trust_remote_code=False, device='cuda:0'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Dense bf16 adapter requires a CUDA GPU; CPU is an explicit small-test option')
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        loader = Gemma3ForConditionalGeneration if config['model_type'] == 'gemma3' else AutoModelForCausalLM
        self.loaded = loader.from_pretrained(source, torch_dtype=torch.bfloat16, device_map=device,
            local_files_only=True, attn_implementation='sdpa')
        self.tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
        self.loaded.eval()
        self.source, self.config, self.device = source, config, device
        self.torch, self.bs = torch, batch_size
        if config['model_type'] == 'gemma3':
            if hasattr(getattr(self.loaded, 'model', None), 'language_model'):
                decoder = self.loaded.model.language_model
            elif hasattr(self.loaded, 'language_model'):
                decoder = self.loaded.language_model.model
            else:
                raise ValueError('Unsupported runtime Gemma layout')
        else:
            decoder = self.loaded.model
        head = self.loaded.get_output_embeddings()
        self.model = SimpleNamespace(model=decoder, lm_head=head)
        c = config.get('text_config', config)
        self.hidden, self.n_layers = c['hidden_size'], len(decoder.layers)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        if isinstance(self.pad, list):
            self.pad = self.pad[0]
        self.tied = head.weight.data_ptr() == decoder.embed_tokens.weight.data_ptr()
        declared = bool(getattr(decoder.config, 'tie_word_embeddings', False))
        if declared != self.tied:
            raise ValueError('Configured/runtime embedding tying differs')
        self.inv = inventory(source)
        self.modes, self.scales = writers(config, self.inv), normalization_scales(config, self.inv)
        self.params = dict(self.loaded.named_parameters(remove_duplicate=False))
        self.paths = {}
        for key in set(self.modes) | set(self.scales.values()):
            candidates = [key]
            if key.startswith('language_model.model.'):
                candidates.append(key.replace('language_model.model.', 'model.language_model.', 1))
            if key.startswith('multi_modal_projector.'):
                candidates.append('model.' + key)
            matches = [k for k in candidates if k in self.params]
            if len(matches) != 1:
                raise ValueError(f'Source/runtime tensor mapping is ambiguous or absent: {key}')
            self.paths[key] = matches[0]

    def encode(self, text):
        return list(self.tok.apply_chat_template([dict(role='user',content=text)],
            add_generation_prompt=True, tokenize=True, return_dict=False, enable_thinking=False))

    def suffix(self):
        marker='\x00INSTR\x00'
        s=self.tok.apply_chat_template([dict(role='user',content=marker)],
            add_generation_prompt=True,tokenize=False,enable_thinking=False)
        ids=self.tok.encode(s.split(marker,1)[1],add_special_tokens=False)
        if not ids or self.encode('Tell me a joke.')[-len(ids):]!=ids:
            raise ValueError('Template suffix does not align in context')
        return ids

    def padded(self, batches):
        t = self.torch
        lens = t.tensor([len(b) for b in batches], device=self.device)
        x = t.full((len(batches), int(lens.max())), self.pad, dtype=t.long, device=self.device)
        for i, seq in enumerate(batches):
            x[i, :len(seq)] = t.tensor(seq, device=self.device)
        return x, lens

    def readout(self, h):
        y = self.model.lm_head(h)
        config = getattr(self.loaded.config, 'text_config', self.loaded.config)
        cap = getattr(config, 'final_logit_softcapping', None)
        return self.torch.tanh(y / cap) * cap if cap else y

    def lp(self, batches):
        t, out = self.torch, []
        for s in range(0,len(batches),self.bs):
            x,lens = self.padded(batches[s:s+self.bs])
            h = self.model.model(input_ids=x,use_cache=False).last_hidden_state
            h = h[t.arange(len(lens),device=self.device),lens-1]
            out.append(t.log_softmax(self.readout(h).float(),-1).cpu().numpy())
        return np.concatenate(out)

    def cache(self, prompts, suffix):
        t = self.torch
        batch = [self.encode(p) for p in prompts]
        if not suffix or any(b[-len(suffix):] != suffix for b in batch):
            raise ValueError('Suffix is not aligned in every prompt')
        acts,firsts = [],[]
        for s in range(0,len(batch),self.bs):
            x,lens = self.padded(batch[s:s+self.bs])
            rows=t.arange(len(lens),device=self.device)[:,None]
            idx=lens[:,None]-len(suffix)+t.arange(len(suffix),device=self.device)[None]
            kept={}
            def take(i,h):
                if i<self.n_layers:
                    kept[i]=h[rows,idx].float().cpu().numpy()
            with self.recording(take):
                h=self.model.model(input_ids=x,use_cache=False).last_hidden_state
            firsts.extend(self.readout(h[t.arange(len(lens),device=self.device),lens-1]).argmax(-1).tolist())
            acts.append(np.stack([kept[i] for i in range(self.n_layers)],axis=1))
        return np.concatenate(acts),firsts

    def edit(self, r):
        r=self.torch.as_tensor(r,device=self.device)
        seen=set()
        for key,mode in self.modes.items():
            w=self.params[self.paths[key]]
            if w.data_ptr() in seen:
                continue
            seen.add(w.data_ptr())
            scale=self.params[self.paths[self.scales[key]]] if key in self.scales else None
            w.copy_(project(w,mode,r,scale))

    @contextmanager
    def ablated(self,r):
        # Actual weight intervention matches both tied readers and Gemma's post-writer RMSNorm exactly.
        originals=[]
        seen=set()
        try:
            for key in self.modes:
                w=self.params[self.paths[key]]
                if w.data_ptr() not in seen:
                    seen.add(w.data_ptr())
                    originals.append((w,w.detach().clone()))
            self.edit(r)
            yield
        finally:
            for w,original in originals:
                w.copy_(original)

    @contextmanager
    def added(self,layer,vector):
        v=self.torch.as_tensor(vector,device=self.device)
        def pre(mod,args,kwargs):
            if args:
                return (((args[0].float()+v).to(args[0].dtype),)+args[1:],kwargs)
            kwargs=dict(kwargs)
            kwargs['hidden_states']=(kwargs['hidden_states'].float()+v).to(kwargs['hidden_states'].dtype)
            return args,kwargs
        handle=self.model.model.layers[layer].register_forward_pre_hook(pre,with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()

    def projection(self,prompts,r):
        t=self.torch
        r=t.as_tensor(r,device=self.device)
        sums,count=np.zeros(self.n_layers+1),0
        batches=[self.encode(p) for p in prompts]
        for s in range(0,len(batches),self.bs):
            x,lens=self.padded(batches[s:s+self.bs])
            valid=t.arange(x.shape[1],device=self.device)[None]<lens[:,None]
            def take(i,h):
                sums[i]+=float(((h.float()@r).abs()*valid).sum())
            with self.recording(take):
                self.model.model(input_ids=x,use_cache=False)
            count+=int(lens.sum())
        return sums/count


def export_shard(source,target,modes,r,*,config,source_root,inv,device='cuda:0'):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file,save_file
    scales=normalization_scales(config,inv)
    weights=load_file(str(source),device='cpu')
    direction=torch.as_tensor(r,device=device)
    expected={}
    for key in list(weights):
        if key in modes:
            scale=None
            if key in scales:
                name=scales[key]
                with safe_open(str(source_root/inv[name]['shard']),framework='pt',device='cpu') as f:
                    scale=f.get_tensor(name).to(device)
            weights[key]=project(weights[key].to(device),modes[key],direction,scale).cpu().contiguous()
        expected[key]=hashlib.sha256(weights[key].view(torch.uint8).numpy().tobytes()).hexdigest()
    save_file(weights,str(target),metadata={'format':'pt'})
    del weights
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return expected

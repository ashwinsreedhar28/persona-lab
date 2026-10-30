"""Explicit writer maps. Unknown architectures fail before loading/editing any tensors."""
import re


def adapter_for(config):
    if config.get('quantization') or config.get('quantization_config') or config.get('text_config', {}).get('quantization'):
        raise ValueError('Quantized checkpoints are unsupported; supply original bf16 weights')
    kind = config.get('model_type')
    if kind == 'qwen3_5' and config.get('text_config', {}).get('model_type') == 'qwen3_5_text':
        return 'qwen-mlx'
    if kind == 'kimi_linear':
        # This model_type is also used for other architectures. Require the audited structure.
        required = dict(hidden_size=2304, num_hidden_layers=27, num_experts=256,
                        first_k_dense_replace=1, num_shared_experts=1, tie_word_embeddings=False)
        if any(config.get(k) != v for k, v in required.items()):
            raise ValueError('Only the Kimi Linear 48B-A3B architecture is supported')
        return 'kimi-torch'
    raise ValueError(f'Unsupported architecture: {kind!r}; add and test an adapter first')


def writers(config, tensors):
    adapter = adapter_for(config)
    c = config.get('text_config', config)
    hidden, n = c['hidden_size'], c['num_hidden_layers']
    modes = {}
    def add(key, mode):
        if key not in tensors:
            raise ValueError(f'Missing residual writer: {key}')
        spec = tensors[key]
        if spec['dtype'] != 'BF16':
            raise ValueError(f'Residual writer must be bf16: {key}')
        axis = -1 if mode == 'embedding' else 0
        if spec['shape'][axis] != hidden or len(spec['shape']) != (1 if mode == 'bias' else 2):
            raise ValueError(f'Unexpected residual writer shape: {key}')
        modes[key] = mode
    def linear(prefix):
        add(prefix + '.weight', 'linear')
        if prefix + '.bias' in tensors:
            add(prefix + '.bias', 'bias')
    if adapter == 'qwen-mlx':
        add('model.language_model.embed_tokens.weight', 'embedding')
        types = c.get('layer_types')
        if not types or len(types) != n or any(t not in ('linear_attention', 'full_attention') for t in types):
            raise ValueError('Qwen layer_types must explicitly describe all decoder layers')
        for i, typ in enumerate(types):
            prefix = f'model.language_model.layers.{i}'
            linear(prefix + ('.linear_attn.out_proj' if typ == 'linear_attention' else '.self_attn.o_proj'))
            linear(prefix + '.mlp.down_proj')
        if config.get('vision_config'):
            linear('model.visual.merger.linear_fc2')
    else:
        add('model.embed_tokens.weight', 'embedding')
        for i in range(n):
            prefix = f'model.layers.{i}'
            linear(prefix + '.self_attn.o_proj')
            if i == 0:
                linear(prefix + '.mlp.down_proj')
            else:
                for j in range(c['num_experts']):
                    linear(prefix + f'.block_sparse_moe.experts.{j}.w2')
                linear(prefix + '.block_sparse_moe.shared_experts.down_proj')
    # Refuse unaccounted writer-like keys, including a new expert/layout variant.
    writer_like = re.compile(r'(?:\.down_proj|\.out_proj|\.o_proj|\.w2|\.merger\.linear_fc2)\.(?:weight|bias)$')
    extras = [k for k in tensors if writer_like.search(k) and k not in modes and
              not k.startswith('mtp.') and not k.startswith('model.visual.blocks.')]
    if extras:
        raise ValueError(f'Unmapped output projections: {extras[:5]}')
    return modes

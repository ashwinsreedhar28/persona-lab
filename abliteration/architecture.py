"""Explicit writer maps. Unknown architectures fail before loading/editing any tensors."""
import re


def adapter_for(config):
    if config.get('quantization') or config.get('quantization_config') or config.get('text_config', {}).get('quantization'):
        raise ValueError('Quantized checkpoints are unsupported; supply original bf16 weights')
    kind = config.get('model_type')
    if kind == 'qwen3_5' and config.get('text_config', {}).get('model_type') == 'qwen3_5_text':
        return 'qwen-mlx'
    if kind in ('qwen3', 'llama', 'glm4_moe_lite'):
        return 'dense-torch'
    if kind == 'gemma3' and config.get('text_config', {}).get('model_type') == 'gemma3_text':
        return 'dense-torch'
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
        axis = -1 if mode == 'embedding' else 1 if mode == 'expert_linear' else 0
        rank = 3 if mode == 'expert_linear' else 1 if mode == 'bias' else 2
        if len(spec['shape']) != rank or spec['shape'][axis] != hidden:
            raise ValueError(f'Unexpected residual writer shape: {key}')
        if mode == 'expert_linear' and spec['shape'] != [c['n_routed_experts'], hidden, c['moe_intermediate_size']]:
            raise ValueError(f'Unexpected expert writer shape: {key}')
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
    elif adapter == 'kimi-torch':
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
    elif config['model_type'] == 'glm4_moe_lite':
        add('model.embed_tokens.weight', 'embedding')
        types = c.get('mlp_layer_types')
        if types is None:
            first = c.get('first_k_dense_replace', 1)
            if not isinstance(first, int) or not 0 <= first <= n:
                raise ValueError('Invalid GLM dense-layer count')
            types = ['dense'] * first + ['sparse'] * (n - first)
        if len(types) != n or any(t not in ('dense', 'sparse') for t in types):
            raise ValueError('Invalid GLM MLP layer types')
        for i, typ in enumerate(types):
            prefix = f'model.layers.{i}'
            linear(prefix + '.self_attn.o_proj')
            if typ == 'dense':
                linear(prefix + '.mlp.down_proj')
            else:
                packed = prefix + '.mlp.experts.down_proj'
                unpacked = [prefix + f'.mlp.experts.{j}.down_proj.weight'
                            for j in range(c['n_routed_experts'])]
                if packed in tensors:
                    if any(key in tensors for key in unpacked):
                        raise ValueError('Ambiguous packed/unpacked GLM expert layout')
                    add(packed, 'expert_linear')
                else:
                    for key in unpacked:
                        add(key, 'linear')
                        if tensors[key]['shape'] != [hidden, c['moe_intermediate_size']]:
                            raise ValueError(f'Unexpected expert writer shape: {key}')
                linear(prefix + '.mlp.shared_experts.down_proj')
    else:
        if config['model_type'] == 'gemma3':
            prefix = next((p for p in ('language_model.model', 'model.language_model')
                           if p + '.embed_tokens.weight' in tensors), None)
            if prefix is None:
                raise ValueError('Unrecognized Gemma text tensor layout')
        else:
            prefix = 'model'
        add(prefix + '.embed_tokens.weight', 'embedding')
        for i in range(n):
            linear(prefix + f'.layers.{i}.self_attn.o_proj')
            linear(prefix + f'.layers.{i}.mlp.down_proj')
        if config['model_type'] == 'gemma3':
            projector = next((p for p in ('multi_modal_projector.mm_input_projection_weight',
                                         'model.multi_modal_projector.mm_input_projection_weight') if p in tensors), None)
            if config.get('vision_config') and projector is None:
                raise ValueError('Gemma multimodal projector is missing')
            if projector:
                add(projector, 'embedding')  # Stored as (vision_hidden, text_hidden), right-multiplied.
            normalization_scales(config, tensors)  # Validate the post-writer normalization dependencies.
    # Refuse unaccounted writer-like keys, including a new expert/layout variant.
    writer_like = re.compile(r'(?:(?:\.down_proj|\.out_proj|\.o_proj|\.w2|\.merger\.linear_fc2)\.(?:weight|bias)|\.experts\.down_proj)$')
    # Upstream GLM stores next-token-prediction modules after the decoder. They
    # are preserved byte-for-byte, never edited or claimed as verified inference.
    mtp_prefixes = ()
    if config['model_type'] == 'glm4_moe_lite':
        count = c.get('num_nextn_predict_layers', 0)
        if not isinstance(count, int) or not 0 <= count <= 1:
            raise ValueError('Unsupported GLM next-token-prediction layout')
        mtp_prefixes = tuple(f'model.layers.{i}.' for i in range(n, n + count))
    extras = [k for k in tensors if writer_like.search(k) and k not in modes and
              not (mtp_prefixes and k.startswith(mtp_prefixes)) and
              not k.startswith('mtp.') and not k.startswith('model.visual.blocks.') and
              not k.startswith('vision_tower.') and not k.startswith('model.vision_tower.')]
    if extras:
        raise ValueError(f'Unmapped output projections: {extras[:5]}')
    return modes


def normalization_scales(config, tensors):
    """Gemma normalizes/scales writer outputs before residual addition; raw W <- P W is insufficient."""
    if config.get('model_type') != 'gemma3':
        return {}
    prefix = next((p for p in ('language_model.model', 'model.language_model')
                   if p + '.embed_tokens.weight' in tensors), None)
    if prefix is None:
        raise ValueError('Unrecognized Gemma text tensor layout')
    c = config['text_config']
    result = {}
    for i in range(c['num_hidden_layers']):
        for writer, norm in (('self_attn.o_proj', 'post_attention_layernorm'),
                             ('mlp.down_proj', 'post_feedforward_layernorm')):
            key = prefix + f'.layers.{i}.{norm}.weight'
            if key not in tensors or tensors[key]['shape'] != [c['hidden_size']]:
                raise ValueError(f'Missing or invalid Gemma post-writer scale: {key}')
            result[prefix + f'.layers.{i}.{writer}.weight'] = key
            bias = prefix + f'.layers.{i}.{writer}.bias'
            if bias in tensors:
                result[bias] = key
    return result

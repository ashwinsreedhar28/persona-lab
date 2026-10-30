"""Integrity/failure tests and a tiny real Qwen save/reload integration. No downloads or 4B loads."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from abliteration.architecture import adapter_for, writers
from abliteration.cli import ROOT, source_files, verify
from abliteration.storage import destination, digest, file_manifest, file_manifest_names, header, inventory, publish, write_json
from abliteration.worker import auroc, kl


def spec(shape, dtype='BF16'):
    return dict(shape=shape, dtype=dtype, shard='model.safetensors')


def qconfig():
    return dict(model_type='qwen3_5', tie_word_embeddings=True, text_config=dict(
        model_type='qwen3_5_text', hidden_size=16, intermediate_size=24, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=8, linear_value_head_dim=8,
        full_attention_interval=4, linear_conv_kernel_dim=4,
        layer_types=['linear_attention'] * 3 + ['full_attention'], tie_word_embeddings=True))


class IntegrityTests(unittest.TestCase):
    def test_source_under_huggingface_cache_keeps_model_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / '.cache/huggingface/hub/snapshots/abcdef'
            root.mkdir(parents=True)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors').write_bytes(b'fixture')
            (root / '.cache').mkdir()
            (root / '.cache/ignored.json').write_text('{}')
            self.assertEqual(source_files(root), ['config.json', 'model.safetensors'])
            self.assertEqual(set(file_manifest(root)), {'config.json','model.safetensors'})
            self.assertEqual(set(file_manifest_names(root)), {'config.json','model.safetensors'})

    def test_unknown_and_quantized_architectures_rejected(self):
        for c in (dict(model_type='unsupported'), dict(qconfig(), quantization=dict(bits=4)),
                  dict(model_type='kimi_linear', hidden_size=8192)):
            with self.assertRaises(ValueError):
                adapter_for(c)

    def test_kimi_all_experts_shared_and_reader_policy(self):
        cfg = dict(model_type='kimi_linear', hidden_size=2304, num_hidden_layers=27, num_experts=256,
                   first_k_dense_replace=1, num_shared_experts=1, tie_word_embeddings=False)
        inv = {'model.embed_tokens.weight': spec([10, 2304]), 'lm_head.weight': spec([10, 2304])}
        for i in range(27):
            inv[f'model.layers.{i}.self_attn.o_proj.weight'] = spec([2304, 10])
            if i == 0:
                inv['model.layers.0.mlp.down_proj.weight'] = spec([2304, 10])
            else:
                for e in range(256):
                    inv[f'model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight'] = spec([2304, 10])
                inv[f'model.layers.{i}.block_sparse_moe.shared_experts.down_proj.weight'] = spec([2304, 10])
        modes = writers(cfg, inv)
        self.assertEqual(len(modes), 6711)
        self.assertNotIn('lm_head.weight', modes)
        self.assertEqual(sum('.shared_experts.' in k for k in modes), 26)
        del inv['model.layers.26.block_sparse_moe.experts.255.w2.weight']
        with self.assertRaises(ValueError):
            writers(cfg, inv)

    def test_source_destination_overlap_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            source = p / 'source'
            source.mkdir()
            for target in (source, source / 'child', p):
                with self.assertRaises((ValueError, FileExistsError)):
                    destination(source, target)
            link = p / 'link'
            link.symlink_to(source, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                destination(source, link)

    def test_atomic_publish_does_not_replace_even_empty_directory(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            staging, out = p / 'staging', p / 'out'
            staging.mkdir()
            (staging / 'test').write_text('retained')
            out.mkdir()
            with self.assertRaises(OSError):
                publish(staging, out)
            self.assertEqual((staging / 'test').read_text(), 'retained')
            self.assertEqual(list(out.iterdir()), [])
            other = p / 'new'
            publish(staging, other)
            self.assertEqual((other / 'test').read_text(), 'retained')

    def test_malformed_headers_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'broken.safetensors').write_bytes(struct.pack('<Q', 5000) + b'{}')
            with self.assertRaises(ValueError):
                header(p / 'broken.safetensors')
            write_json(p / 'model.safetensors.index.json', dict(weight_map={'a': '../other.safetensors'}))
            with self.assertRaises(ValueError):
                inventory(p)

    def test_selection_statistics(self):
        self.assertEqual(auroc(np.array([2., 3.]), np.array([0., 1.])), 1.)
        self.assertEqual(auroc(np.ones(2), np.ones(2)), .5)
        p, q = np.log([[.2, .8]]), np.log([[.5, .5]])
        np.testing.assert_allclose(kl(p, q), .2 * np.log(.4) + .8 * np.log(1.6))


class TinyRoundTripTests(unittest.TestCase):
    def make_source(self, root):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.models.qwen3_5 import Model, ModelArgs
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        cfg = qconfig()
        mx.random.seed(42)
        model = Model(ModelArgs.from_dict(cfg))
        model.set_dtype(mx.bfloat16)
        model.eval()
        mx.eval(model.parameters())
        weights = {}
        for name, value in tree_flatten(model.parameters()):
            name = name.replace('language_model.model.', 'model.language_model.', 1)
            weights[name] = value
        mx.save_safetensors(str(root / 'model.safetensors'), weights, metadata={'format': 'pt'})
        write_json(root / 'config.json', cfg)
        tok = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'user': 3, 'assistant': 4,
                                  'hello': 5, 'one': 6, 'two': 7, 'three': 8, 'four': 9, 'five': 10,
                                  'six': 11, 'seven': 12, 'eight': 13, 'nine': 14, 'ten': 15}, unk_token='[UNK]'))
        tok.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
        tok.chat_template = 'user {{ messages[0]["content"] }} [EOS] assistant'
        tok.save_pretrained(root)
        return cfg

    def test_real_shard_roundtrip_integrity_and_original_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            source, data, output = p / 'source', p / 'data', p / 'edited'
            source.mkdir()
            data.mkdir()
            self.make_source(source)
            for name, prompts in dict(harmful_train=['one'], harmless_train=['two'], harmful_val=['three'],
                                      harmless_val=['four'], harmless_kl=['five']).items():
                write_json(data / f'{name}.json', prompts)
            direction = p / 'direction.npy'
            np.save(direction, np.eye(16, dtype=np.float32)[0])
            before = {f.name: digest(f) for f in source.iterdir() if f.is_file()}
            cmd = [sys.executable, str(ROOT / 'abliterate.py'), 'run', '--model', str(source),
                   '--data', str(data), '--direction', str(direction), '--output', str(output), '--offline']
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            result = verify(output)
            self.assertTrue(result['reload']['passed'])
            self.assertEqual(result['edited_tensors'], 9)
            self.assertEqual(before, {f.name: digest(f) for f in source.iterdir() if f.is_file()})
            original = digest(output / 'model.safetensors')
            rerun = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(digest(output / 'model.safetensors'), original)
            with (output / 'model.safetensors').open('r+b') as f:
                f.seek(-1, 2)
                byte = f.read(1)
                f.seek(-1, 2)
                f.write(bytes([byte[0] ^ 1]))
            with self.assertRaises(ValueError):
                verify(output)

    def test_bad_direction_failure_does_not_publish(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            source, data, output = p / 'source', p / 'data', p / 'edited'
            source.mkdir()
            data.mkdir()
            self.make_source(source)
            for name, prompts in dict(harmful_train=['one'], harmless_train=['two'], harmful_val=['three'],
                                      harmless_val=['four'], harmless_kl=['five']).items():
                write_json(data / f'{name}.json', prompts)
            direction = p / 'direction.npy'
            np.save(direction, np.zeros(16, dtype=np.float32))
            before = digest(source / 'model.safetensors')
            proc = subprocess.run([sys.executable, str(ROOT / 'abliterate.py'), 'run', '--model', str(source),
                '--data', str(data), '--direction', str(direction), '--output', str(output)], cwd=ROOT, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(output.exists())
            self.assertEqual(before, digest(source / 'model.safetensors'))
            self.assertTrue(list(p.glob('.edited.incomplete-*')))


if __name__ == '__main__':
    unittest.main()

"""MLX math and hook tests on tiny tensors/models; no downloads or generations.

Run: python -m unittest -v tests.test_interventions
"""
from pathlib import Path
import contextlib
import copy
import io
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "abl"))
from common import (_project, ablated, added, kl_rows, last_logprobs, pad_right,
                    recording, refusal_score, writer_slots)
from a2_directions import cache as extract_cache
from a4_edit import project_out_cols, project_out_rows
from a5_projection import projections
import a3_select
from tools.build_vector import Rec
import build_refusal
from evil_chat_mlx import AddVector, Recorder, SteeredDown
from mlx_lm.models.qwen3_5 import Model, ModelArgs


class AffineLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_linear = False
        self.seen = None

    def __call__(self, x, mask=None, cache=None):
        self.seen = (mask, cache)
        return 2 * x + 1


def tiny_model():
    mx.random.seed(12)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=dict(
        hidden_size=16, intermediate_size=24, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=32, linear_num_value_heads=2, linear_num_key_heads=1,
        linear_key_head_dim=8, linear_value_head_dim=8,
        full_attention_interval=4, tie_word_embeddings=True)))
    model.eval()
    mx.eval(model.parameters())
    return model


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return list(messages[0]["content"])


class InterventionMathTest(unittest.TestCase):
    def setUp(self):
        self.r = mx.array([.6, .8, 0.])
        self.x = mx.array([[[1., 2., 3.], [-4., 5., 6.]], [[7., 8., 9.], [2., -3., 1.]]])

    def test_projection_properties(self):
        x, r = np.array(self.x), np.array(self.r)
        expected = x - (x @ r)[..., None] * r
        y = _project(self.x, self.r)
        np.testing.assert_allclose(np.array(y), expected, atol=1e-6)
        np.testing.assert_allclose(np.array(y @ self.r), 0, atol=1e-6)
        np.testing.assert_allclose(np.array(_project(y, self.r)), expected, atol=1e-6)
        np.testing.assert_allclose(np.array(_project(self.x, -self.r)), expected, atol=1e-6)
        self.assertTrue(np.all(np.linalg.norm(expected, axis=-1) <= np.linalg.norm(x, axis=-1) + 1e-6))

    def test_signed_addition_is_translation_not_ablation(self):
        r, x = self.r, self.x
        c = 2.
        plus, minus = x + c * r, x - c * r
        np.testing.assert_allclose(np.array(plus @ r), np.array(x @ r) + c, atol=2e-6)
        np.testing.assert_allclose(np.array(minus @ r), np.array(x @ r) - c, atol=2e-6)
        np.testing.assert_allclose(np.array((plus + minus) / 2), np.array(x), atol=1e-6)
        self.assertFalse(np.allclose(np.array(minus), np.array(_project(x, r))))

    def test_weight_orientation_and_bias_equivalence(self):
        W = mx.array(np.arange(12, dtype=np.float32).reshape(3, 4) / 7)
        inputs = mx.array([[1., 2., -1., .5]])
        bias = mx.array([1., -2., 3.])
        edited = inputs @ project_out_rows(W, self.r).T + _project(bias, self.r)
        expected = _project(inputs @ W.T + bias, self.r)
        np.testing.assert_allclose(np.array(edited), np.array(expected), atol=2e-6)
        E = W.T
        np.testing.assert_allclose(np.array(project_out_cols(E, self.r, chunk=2)),
                                   np.array(_project(E, self.r)), atol=1e-6)

    def test_projection_preserves_bf16_dtype(self):
        y = _project(self.x.astype(mx.bfloat16), self.r)
        self.assertEqual(y.dtype, mx.bfloat16)
        expected = _project(self.x, self.r).astype(mx.bfloat16)
        np.testing.assert_array_equal(np.array(y.astype(mx.float32)), np.array(expected.astype(mx.float32)))

    def test_kl_direction_and_refusal_log_odds(self):
        p, q = np.array([[.2, .3, .5]]), np.array([[.4, .4, .2]])
        actual = kl_rows(mx.array(np.log(p)), mx.array(np.log(q)))
        np.testing.assert_allclose(np.array(actual), np.sum(p * np.log(p / q), -1), rtol=1e-6)
        np.testing.assert_allclose(np.array(kl_rows(mx.array(np.log(p)), mx.array(np.log(p)))), 0, atol=1e-7)
        np.testing.assert_allclose(refusal_score(np.log(p), [0, 1]), [0], atol=1e-7)


class HookTest(unittest.TestCase):
    def test_pre_post_hooks_and_forwarded_state(self):
        inner = AffineLayer()
        model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(
            layers=[inner], embed_tokens=SimpleNamespace(weight=mx.zeros((4, 3))))))
        x, v = mx.ones((2, 4, 3)), mx.array([1., -2., 3.])
        mask, cache = object(), object()
        post = AddVector(inner, v)
        np.testing.assert_allclose(np.array(post(x, mask, cache)), np.array(2*x+1+v))
        self.assertIs(inner.seen[0], mask)
        self.assertIs(inner.seen[1], cache)
        with added(model, 0, np.array(v)):
            np.testing.assert_allclose(np.array(model.language_model.model.layers[0](x)), np.array(2*(x+v)+1))
        self.assertIs(model.language_model.model.layers[0], inner)

    def test_gen_only_and_down_projection(self):
        inner = AffineLayer()
        hook = AddVector(inner, mx.ones((3,)), gen_only=True)
        for length in (1, 4):
            x = mx.ones((2, length, 3))
            expected = inner(x) + (1 if length == 1 else 0)
            np.testing.assert_allclose(np.array(hook(x)), np.array(expected))
        down = SteeredDown(nn.Linear(3, 2, bias=False), 3, gen_only=True)
        down.delta = mx.array([1., 2., 3.])
        for length in (1, 4):
            x = mx.ones((2, length, 3))
            expected = down.inner(x + down.delta) if length == 1 else down.inner(x)
            np.testing.assert_allclose(np.array(down(x)), np.array(expected))

    def test_recorders_capture_post_layer_token_and_span(self):
        inner = AffineLayer()
        x = mx.array(np.arange(12, dtype=np.float32).reshape(1, 4, 3))
        rec = Rec(inner)
        out = rec(x)
        np.testing.assert_array_equal(np.array(rec.last), np.array(out[0, -1]))
        rec.span = (1, 3)
        rec(x)
        np.testing.assert_allclose(np.array(rec.last), np.array(out[0, 1:3].mean(0)))
        norms = Recorder(inner)
        norms(x)
        np.testing.assert_allclose(np.array(norms.norms[0]), np.linalg.norm(np.array(out[0, 1:]), axis=-1))
        for span in ((2, 2), (-1, 2), (1, 5)):
            rec.span = span
            with self.assertRaises(ValueError):
                rec(x)

    def test_contexts_restore_on_exceptions_and_reject_bad_inputs(self):
        model = tiny_model()
        original_layers = list(model.language_model.model.layers)
        slots = writer_slots(model)
        originals = [getattr(p, a) for p, a, _ in slots]
        r = np.eye(16, dtype=np.float32)[0]
        for context in (recording(model), added(model, 1, r), ablated(model, r)):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with context:
                    raise RuntimeError("injected")
            self.assertTrue(all(a is b for a, b in zip(original_layers, model.language_model.model.layers)))
            self.assertTrue(all(getattr(p, a) is b for (p, a, _), b in zip(slots, originals)))
        for invalid in (r * 2, r * np.nan, np.zeros(16), np.ones(2) / np.sqrt(2)):
            with self.assertRaises(ValueError):
                with ablated(model, invalid):
                    self.fail("invalid projection entered context")
        for layer, vector in ((-1, r), (4, r), (0, [1, 2]), (0, r * np.nan)):
            with self.assertRaises(ValueError):
                with added(model, layer, vector):
                    self.fail("invalid addition entered context")

    def test_tied_embedding_read_policy(self):
        model = tiny_model()
        text = model.language_model.model
        embedding = text.embed_tokens
        r = np.eye(16, dtype=np.float32)[0]
        x = mx.ones((2, 16))
        for edit_read in (False, True):
            with ablated(model, r, edit_unembed=edit_read):
                expected = embedding.as_linear(_project(x, mx.array(r)) if edit_read else x)
                np.testing.assert_allclose(np.array(text.embed_tokens.as_linear(x)), np.array(expected))

    def test_padding_extraction_and_mechanical_check_on_hybrid_model(self):
        model, tok = tiny_model(), ToyTokenizer()
        batch = [[1, 2, 3, 4, 5], [6, 7, 4, 5]]
        x, _ = pad_right(batch, 0)
        with recording(model) as recs:
            for rec in recs:
                rec.take = lambda h: h
            model(x)
            np.testing.assert_array_equal(np.array(recs[0].kept[0]),
                                          np.array(model.language_model.model.embed_tokens(x)))
        together = last_logprobs(model, batch, 0, bs=2)
        separate = last_logprobs(model, batch, 0, bs=1)
        np.testing.assert_allclose(np.array(together), np.array(separate), atol=2e-5)
        acts, first = extract_cache(model, tok, batch, [4, 5], bs=2)
        individual, individual_first = extract_cache(model, tok, batch, [4, 5], bs=1)
        self.assertEqual(acts.shape, (2, 4, 2, 16))
        np.testing.assert_allclose(acts, individual, atol=2e-5)
        self.assertEqual(first, individual_first)
        with self.assertRaises(ValueError):
            extract_cache(model, tok, batch, [8, 9])
        r = np.eye(16, dtype=np.float32)[0]
        np.testing.assert_allclose(projections(model, tok, batch, mx.array(r), bs=2),
                                   projections(model, tok, batch, mx.array(r), bs=1), atol=2e-6)
        with ablated(model, r):
            np.testing.assert_allclose(projections(model, tok, batch, mx.array(r)), 0, atol=1e-6)


class SelectionTest(unittest.TestCase):
    def test_legacy_checkpoint_refused_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "selection.jsonl"
            ckpt.write_text('{"layer": 1, "pos": -1}\n')
            with patch.object(sys, "argv", ["a3_select.py", "--checkpoint", str(ckpt)]), \
                 patch.object(a3_select, "provenance", return_value={"run": 1}), \
                 patch.object(a3_select, "load") as load, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    a3_select.main()
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()
            self.assertEqual(ckpt.read_text(), '{"layer": 1, "pos": -1}\n')

    def test_debug_uses_first_candidates_and_preserves_headline(self):
        model = tiny_model()
        tok = ToyTokenizer()
        tok.encode = lambda text, **kwargs: [1 if text == "I" else 2]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            r = np.zeros((4, 1, 16), dtype=np.float32)
            r[:, :, 0] = 1
            np.savez(out / "directions.npz", r=r)
            (out / "selection.json").write_text("headline")
            (out / "r_hat.npy").write_bytes(b"headline direction")
            with patch.object(sys, "argv", ["a3_select.py", "--limit", "1"]), \
                 patch.object(a3_select, "OUT", out), \
                 patch.object(a3_select, "provenance", return_value={"run": 1}), \
                 patch.object(a3_select, "load", return_value=(model, tok)), \
                 patch.object(a3_select, "read_json", return_value=[[1, 2, 3]]), \
                 patch.object(a3_select, "last_logprobs", return_value=mx.zeros((1, 32))), \
                 patch.object(a3_select, "refusal_score", side_effect=[np.array([1]), np.array([-2]),
                                                                        np.array([-5]), np.array([2])]), \
                 patch.object(a3_select, "kl_rows", return_value=mx.array([.05])), \
                 contextlib.redirect_stdout(io.StringIO()):
                a3_select.main()
            import json
            result = json.loads((out / "selection_debug.json").read_text())
            self.assertEqual(result["selected"], dict(layer=0, pos=-1))
            self.assertTrue(result["debug"])
            self.assertEqual((out / "selection.json").read_text(), "headline")
            self.assertEqual((out / "r_hat.npy").read_bytes(), b"headline direction")


class RefusalExtractionTest(unittest.TestCase):
    def test_rescore_rejects_changed_settings_and_corpora(self):
        report = dict(model="test", tokens=64, thinking=False,
                      training=[dict(pair=list(p)) for p in build_refusal.TRAIN],
                      validation=[], validation_baseline=dict(rows=[dict(group=g, prompt=p)
                          for pair in build_refusal.VALIDATION
                          for g, p in zip(("risky", "benign"), pair)]))
        build_refusal.validate_rescore(report, "test", 64)
        for field, value in (("model", "other"), ("tokens", 128), ("thinking", True), ("training", [])):
            bad = dict(report, **{field: value})
            with self.assertRaises(ValueError):
                build_refusal.validate_rescore(bad, "test", 64)
        bad = copy.deepcopy(report)
        bad["validation_baseline"]["rows"][0]["prompt"] = "changed corpus"
        with self.assertRaises(ValueError):
            build_refusal.validate_rescore(bad, "test", 64)

    def test_signed_extraction_matches_direct_post_layer_means(self):
        model, tok = tiny_model(), ToyTokenizer()
        pairs = [([1, 2, 3], [6, 7, 8, 9]), ([2, 3, 4, 5], [8, 9, 10])]
        original = list(model.language_model.model.layers)
        groups = []
        for side in (0, 1):
            rows = []
            for pair in pairs:
                h = model.language_model.model.embed_tokens(mx.array(pair[side])[None])
                states = []
                for layer in original:
                    h = layer(h, mask=None if layer.is_linear else "causal")
                    states.append(np.array(h[0, -1]))
                rows.append(states)
            groups.append(np.array(rows))
        diff, medians, score = build_refusal.extract(model, tok, pairs)
        np.testing.assert_allclose(diff, groups[0].mean(0) - groups[1].mean(0), atol=2e-6)
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue((medians > 0).all())
        self.assertTrue(all(a is b for a, b in zip(original, model.language_model.model.layers)))

    def test_duplicate_manual_corpus_rejected_before_loading_model(self):
        with patch.object(sys, "argv", ["build_refusal.py"]), \
             patch.object(build_refusal, "TRAIN", [("Same prompt", "Another prompt")]), \
             patch.object(build_refusal, "VALIDATION", [("same prompt!", "Distinct prompt")]), \
             patch.object(build_refusal, "load") as load, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                build_refusal.main()
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

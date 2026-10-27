"""Integration checks for the saved artifact and both existing steering loaders.

Run after building refusal.npz: python -m unittest -v test_refusal
Uses tiny identity layers, not a second copy of the language model.
"""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from evil_chat_mlx import Steerer
from steer import MultiSteerer

ARTIFACT = Path(__file__).with_name("refusal.npz")


class IdentityLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.is_linear = False
        self.mlp = SimpleNamespace(down_proj=nn.Linear(2, hidden, bias=False))

    def __call__(self, x, mask=None, cache=None):
        return x


class RefusalIntegrationTest(unittest.TestCase):
    def setUp(self):
        with np.load(ARTIFACT, allow_pickle=False) as z:
            self.diff = z["diff"]
            self.layer = int(z["best"])
            self.median = float(z["medians"][self.layer])
            self.unit = z["direction"]
        self.layers = [IdentityLayer(self.diff.shape[1]) for _ in self.diff]
        self.model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=self.layers)))

    def test_artifact_and_signed_loader_parity(self):
        self.assertTrue(np.isfinite(self.diff).all())
        np.testing.assert_allclose(np.linalg.norm(self.unit), 1, atol=1e-6)
        np.testing.assert_allclose(self.unit, self.diff[self.layer] / np.linalg.norm(self.diff[self.layer]))
        x = mx.zeros((1, 3, len(self.unit)))
        terminal = Steerer(self.model, None)
        web = MultiSteerer(self.model, None)
        for dose in (0.2, -0.2):
            with contextlib.redirect_stdout(io.StringIO()):
                terminal.apply_vector(ARTIFACT, dose)
            terminal_out = np.array(self.layers[self.layer](x))
            terminal.clear_vector()
            web.apply(0.0, {"refusal": dose})
            web_out = np.array(self.layers[self.layer](x))
            expected = np.broadcast_to(self.unit * dose * self.median, terminal_out.shape)
            np.testing.assert_allclose(terminal_out, expected, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(web_out, terminal_out, rtol=1e-5, atol=1e-6)
            web.clear()
            np.testing.assert_array_equal(np.array(self.layers[self.layer](x)), np.array(x))

    def test_invalid_command_preserves_existing_vector(self):
        st = Steerer(self.model, None)
        with contextlib.redirect_stdout(io.StringIO()):
            st.apply_vector(ARTIFACT, 0.2)
            current = self.layers[self.layer]
            for dose, layer in ((float("nan"), None), (0.2, -1), (0.2, len(self.layers))):
                with self.assertRaises(ValueError):
                    st.apply_vector(ARTIFACT, dose, layer)
                self.assertIs(self.layers[self.layer], current)
            with self.assertRaises(OSError):
                st.apply_vector("/nonexistent/refusal.npz", 0.2)
            self.assertIs(self.layers[self.layer], current)
            st.apply_vector(None, 0.0)
        self.assertIsInstance(self.layers[self.layer], IdentityLayer)

    def test_web_rejects_wrong_model(self):
        st = MultiSteerer(self.model, None, model_id="a-different-checkpoint")
        with self.assertRaisesRegex(ValueError, "model mismatch"):
            st.load_vector("refusal")


if __name__ == "__main__":
    unittest.main()

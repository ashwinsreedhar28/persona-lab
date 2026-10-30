"""Model-free regression tests: python -m unittest -v tests.test_research_checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.audit_splits import audit
from research_checks import fingerprint, provenance, split_overlap, unit_direction


class ResearchChecksTest(unittest.TestCase):
    def test_unit_direction(self):
        np.testing.assert_allclose(unit_direction([3, 4]), [.6, .8])
        for bad in ([0, 0], [np.nan, 1], [np.inf, 1], [[1, 0]], [], [1e308, 1e308]):
            with self.subTest(vector=bad), np.errstate(over="ignore"):
                with self.assertRaises(ValueError):
                    unit_direction(bad)
        with self.assertRaises(ValueError):
            unit_direction([3, 4], require_unit=True)

    def test_split_overlap_detects_normalization_and_grams(self):
        shared = "one two three four five six seven eight"
        result = split_overlap(dict(train=["Hello, WORLD!", shared + " nine", "hello world"],
                                    val=["HELLO world", "prefix " + shared], test=["separate topic"]))
        self.assertEqual(result["duplicates"]["train"], 1)
        row = result["pairs"][0]
        self.assertEqual(row["exact"], [[0, 0], [2, 0]])
        self.assertEqual(row["shared_8gram"], [[1, 1]])
        self.assertEqual(result["pairs"][1]["exact"], [])

    def test_provenance_changes_with_input_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.json"
            p.write_text("[]")
            first = provenance([p], dict(seed=0))
            second = provenance([p], dict(seed=0))
            self.assertEqual(fingerprint(first), fingerprint(second))
            self.assertNotEqual(fingerprint(first), fingerprint(provenance([p], dict(seed=1))))
            p.write_text("[1]")
            self.assertNotEqual(fingerprint(first), fingerprint(provenance([p], dict(seed=0))))

    def test_split_audit_is_read_only_and_accepts_record_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            for name, rows in dict(train=["Hello!"], jbb=[dict(goal="hello")],
                                   gsm=[dict(question="arithmetic")]).items():
                (data / (name + ".json")).write_text(json.dumps(rows))
            before = {p.name: p.read_bytes() for p in data.iterdir()}
            result = audit(data)
            self.assertEqual(result["sizes"], dict(gsm=1, jbb=1, train=1))
            self.assertEqual(before, {p.name: p.read_bytes() for p in data.iterdir()})
            self.assertTrue(any(p["exact"] for p in result["pairs"]))


if __name__ == "__main__":
    unittest.main()

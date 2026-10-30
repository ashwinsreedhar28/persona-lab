"""Resource and packaging regressions for the reorganized repository; no full models."""
import importlib
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from tools.paths import ROOT, VECTOR_DIR, WEB_DIR, resolve_vector

class LayoutTests(unittest.TestCase):
    def test_bundled_vectors_and_custom_path_precedence(self):
        for name in ('evil','warm','captain','refusal'):
            self.assertEqual(resolve_vector(name+'.npz'),VECTOR_DIR/(name+'.npz'))
            self.assertTrue((VECTOR_DIR/(name+'.npz')).is_file())
        current=Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d);Path('warm.npz').write_bytes(b'custom')
                self.assertEqual(resolve_vector('warm.npz'),Path('warm.npz'))
                self.assertEqual(resolve_vector('missing/file.npz'),Path('missing/file.npz'))
            finally:os.chdir(current)

    def test_web_ui_and_vectors_without_loading_model(self):
        with patch('mlx_lm.load',return_value=(object(),object())),patch('steer.MultiSteerer'):
            server=importlib.import_module('server')
        self.assertEqual(server.index(),(WEB_DIR/'ui.html').read_text())
        self.assertTrue({'evil','warm','captain','refusal'} <= {v['name'] for v in server.list_vectors()})

    def test_committed_cloud_bundle_contains_requirements(self):
        from cloud.runpod_job import bundle
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'code.tar.gz';bundle(path)
            with tarfile.open(path) as archive:
                members=set(archive.getnames())
                self.assertTrue({'project/requirements/cloud.txt','project/requirements/cuda.txt','project/cloud/remote_job.py'}<=members)
                self.assertNotIn('project/legacy/refusal.eval.json',members)
                self.assertNotIn('project/assets/vectors/refusal.npz',members)

    def test_refusal_compatibility_import_shares_module(self):
        import build_refusal
        import tools.build_refusal
        self.assertIs(build_refusal,tools.build_refusal)

if __name__=='__main__':unittest.main()

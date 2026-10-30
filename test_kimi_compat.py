"""Tests the exact factory-vs-direct decorator mismatch from the real Kimi load."""
from functools import wraps
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock
from abliteration.compat import model_input_decorator


class CompatibilityTests(unittest.TestCase):
    def test_offline_workers_receive_no_account_tokens(self):
        from abliteration.cli import run_worker
        process=MagicMock()
        process.stdout=iter([])
        process.wait.return_value=0
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,
            {'HF_TOKEN':'test-only','HUGGING_FACE_HUB_TOKEN':'test-only','RUNPOD_API_KEY':'test-only'}),\
            patch('abliteration.cli.subprocess.Popen',return_value=process) as popen:
            run_worker(Path(d)/'job.json','reference')
            env=popen.call_args.kwargs['env']
            for name in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','RUNPOD_API_KEY'):
                self.assertNotIn(name,env)
            self.assertEqual(env['HF_HUB_OFFLINE'],'1')

    def test_factory_accepts_both_call_syntaxes_and_preserves_keyword_inputs(self):
        calls=[]
        def checker(tie_last_hidden_states=True):
            def decorate(func):
                @wraps(func)
                def wrapped(self,*args,**kwargs):
                    calls.append(tie_last_hidden_states)
                    return func(self,*args,**kwargs)
                return wrapped
            return decorate
        compat=model_input_decorator(checker)
        class Model:
            @compat
            def forward(self,input_ids=None,use_cache=False):
                return input_ids,use_cache
            @compat(tie_last_hidden_states=False)
            def other(self,input_ids=None):
                return input_ids
        self.assertEqual(Model().forward(input_ids=[1,2],use_cache=False),([1,2],False))
        self.assertEqual(Model().other(input_ids=[3]),[3])
        self.assertEqual(calls,[True,False])
        self.assertIs(model_input_decorator(compat),compat)

    def test_direct_api_unchanged(self):
        def checker(func):
            return func
        self.assertIs(model_input_decorator(checker),checker)


if __name__=='__main__':
    unittest.main()

"""EOS/budget, real decoder caching, and checkpoint metadata protection for CUDA chat."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from chat_cuda import checked_job,greedy_ids
from abliteration.storage import digest,identity


class ChatTests(unittest.TestCase):
    def test_eos_budget_and_input_preservation(self):
        class Scripted:
            def __init__(self):
                self.calls=[]
            def lp(self,rows):
                self.calls.append(list(rows[0]))
                token=[2,3,9][min(len(self.calls)-1,2)]
                scores=np.zeros((1,10));scores[0,token]=1
                return scores
        prompt=[1,4];engine=Scripted()
        self.assertEqual(list(greedy_ids(engine,prompt,{9},10,False)),[2,3])
        self.assertEqual(engine.calls,[[1,4],[1,4,2],[1,4,2,3]])
        self.assertEqual(prompt,[1,4])
        self.assertEqual(list(greedy_ids(Scripted(),prompt,{9},1,False)),[2])

    def test_cached_tokens_match_real_tiny_decoder(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.manual_seed(5)
        cfg=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64)
        model=LlamaForCausalLM(cfg).eval()
        engine=SimpleNamespace(torch=torch,device='cpu',model=SimpleNamespace(model=model.model,lm_head=model.lm_head))
        def lp(rows):
            h=model.model(input_ids=torch.tensor(rows),use_cache=False).last_hidden_state[:,-1]
            return torch.log_softmax(model.lm_head(h).float(),-1).detach().numpy()
        engine.lp=lp
        with torch.inference_mode():
            plain=list(greedy_ids(engine,[1,7,4,2],set(),5,False))
            cached=list(greedy_ids(engine,[1,7,4,2],set(),5,True))
        self.assertEqual(cached,plain)

    def test_changed_metadata_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'.pipeline').mkdir()
            cfg={'model_type':'llama'}
            for name,value in {'config.json':cfg,'.pipeline/job.json':{'config':cfg,'adapter':'dense-torch'},
                               '.pipeline/reload.json':{'passed':True}}.items():
                (root/name).write_text(json.dumps(value))
            m={'output_files':{name:{'bytes':p.stat().st_size,'sha256':digest(p)}
                for name in ('config.json','.pipeline/job.json','.pipeline/reload.json')
                for p in [root/name]}}
            sealed=identity(m);m['manifest_sha256']=sealed
            (root/'manifest.json').write_text(json.dumps(m));(root/'COMPLETE').write_text(sealed+'\n')
            self.assertEqual(checked_job(root)['adapter'],'dense-torch')
            with (root/'.pipeline/job.json').open('a') as f:
                f.write(' ')
            with self.assertRaises(ValueError):
                checked_job(root)


if __name__=='__main__':
    unittest.main()

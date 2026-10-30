"""Small CPU fixtures for dense adapters and Gemma's post-writer normalization."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from abliteration.architecture import normalization_scales, writers
from abliteration.cli import ROOT, verify
from abliteration.dense import project


def tokenizer(root):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    raw=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'user':3,'assistant':4,
        'one':5,'two':6,'three':7,'four':8,'five':9,'hello':10},unk_token='[UNK]'))
    raw.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
    tok.chat_template='user {{ messages[0]["content"] }} [EOS] assistant'
    tok.save_pretrained(root)


class DenseTests(unittest.TestCase):
    def test_normalization_aware_projection_removes_post_norm_coordinate(self):
        torch.manual_seed(4)
        w=torch.randn(5,7).to(torch.bfloat16)
        r=torch.tensor([1.,2.,3.,4.,5.]); r=r/r.norm()
        norm_weight=torch.tensor([-.8,-.4,.1,1.,2.])
        edited=project(w,'linear',r,norm_weight).float()
        x=torch.randn(3,7)
        y=x@edited.T
        normalized=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-6)*(1+norm_weight)
        self.assertLess(float((normalized@r).abs().max()),.01)
        ordinary=project(w,'linear',r).float()
        y=x@ordinary.T
        wrong=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-6)*(1+norm_weight)
        self.assertGreater(float((wrong@r).abs().max()),.1)

    def test_all_zero_post_norm_scale_requires_no_weight_change(self):
        w=torch.ones((3,4),dtype=torch.bfloat16)
        r=torch.tensor([1.,0.,0.])
        torch.testing.assert_close(project(w,'linear',r,torch.full((3,),-1.)),w)

    def test_native_dense_models_save_reload_without_downloads(self):
        from transformers import (LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM,
                                  Gemma3Config,Gemma3ForConditionalGeneration)
        shared=dict(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,
                    num_key_value_heads=1,head_dim=8,vocab_size=32,max_position_embeddings=128,
                    bos_token_id=1,eos_token_id=2,pad_token_id=1)
        gemma=Gemma3Config(text_config=dict(shared,model_type='gemma3_text',sliding_window=16,
            sliding_window_pattern=2,layer_types=['sliding_attention','full_attention'],tie_word_embeddings=True),
            vision_config=dict(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
                               image_size=8,patch_size=4),mm_tokens_per_image=4,
            image_token_index=31,boi_token_index=30,eoi_token_index=29)
        configs=[(LlamaConfig(**shared),LlamaForCausalLM),
                 (Qwen3Config(**shared),Qwen3ForCausalLM),
                 (gemma,Gemma3ForConditionalGeneration)]
        for cfg,cls in configs:
            with self.subTest(model_type=cfg.model_type),tempfile.TemporaryDirectory() as d:
                root=Path(d); source=root/'source'; data=root/'data'; output=root/'edited'
                source.mkdir(); data.mkdir()
                torch.manual_seed(8)
                model=cls(cfg).to(torch.bfloat16).eval()
                model.save_pretrained(source,safe_serialization=True)
                del model
                tokenizer(source)
                for name,rows in dict(harmful_train=['one'],harmless_train=['two'],harmful_val=['three'],
                                       harmless_val=['four'],harmless_kl=['five']).items():
                    (data/f'{name}.json').write_text(json.dumps(rows))
                direction=root/'direction.npy'
                np.save(direction,np.eye(16,dtype=np.float32)[0])
                proc=subprocess.run([sys.executable,str(ROOT/'abliterate.py'),'run','--model',str(source),
                    '--data',str(data),'--direction',str(direction),'--output',str(output),'--device','cpu',
                    '--reload-atol','0.00001'],cwd=ROOT,text=True,capture_output=True)
                self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                result=verify(output)
                self.assertTrue(result['reload']['passed'])


if __name__=='__main__':
    unittest.main()

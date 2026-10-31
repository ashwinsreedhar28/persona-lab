"""Output-axis math, Harmony separation, and real tiny MXFP4/bf16 round trips."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from abliteration.architecture import adapter_for
from abliteration.cli import ROOT,verify
from abliteration.conversion import verify_prepared
from abliteration.dense import Engine,project
from abliteration.harmony_chat import FinalChannel
from abliteration.storage import digest
from chat_cuda import greedy_ids
from tests.test_dense_abliteration import tokenizer


class GptOssTests(unittest.TestCase):
    def test_expert_output_axis_and_bias(self):
        torch.manual_seed(5)
        r=torch.randn(16);r=r/r.norm()
        w=torch.randn(3,8,16).to(torch.bfloat16);b=torch.randn(3,16).to(torch.bfloat16)
        edited=project(w,'expert_embedding',r);bias=project(b,'expert_bias',r)
        self.assertLess(float((edited.float()@r).abs().max()),.012)
        self.assertLess(float((bias.float()@r).abs().max()),.012)
        self.assertEqual(edited.dtype,torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'Quantized'):
            adapter_for({'model_type':'gpt_oss','quantization_config':{'quant_method':'mxfp4'}})

    def test_harmony_final_only_no_tools_or_analysis(self):
        class Tok:
            unk_token_id=99
            words=['<|im_start|>','<|im_sep|>','<|im_end|>','<|meta_sep|>','<|fim_suffix|>','<|ghissue|>',
                   'analysis','final','assistant','private analysis','answer','assistant to=python','tool text']
            def convert_tokens_to_ids(self,x):return self.words.index(x) if x in self.words else 99
            def decode(self,ids,**kw):return ''.join(self.words[i] for i in ids)
        tok=Tok();stream=FinalChannel(tok)
        # First assistant header is already begun in the prompt.
        ids=[3,6,1,9,2,0,11,3,6,1,12,5,0,8,3,7,1,10,4]
        shown=[i for i in ids if stream.put(i)]
        self.assertEqual(shown,[10]);self.assertEqual(stream.text(),'answer')
        unfinished=FinalChannel(tok)
        for i in [3,6,1,9]:unfinished.put(i)
        self.assertEqual(unfinished.text(),'')

    def test_mxfp4_preparation_and_independent_edited_reload(self):
        from transformers import GptOssConfig,GptOssForCausalLM
        cfg=GptOssConfig(vocab_size=64,hidden_size=32,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,num_local_experts=2,
            num_experts_per_tok=1,max_position_embeddings=128,sliding_window=16,
            pad_token_id=1,bos_token_id=1,eos_token_id=2,attention_bias=True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original';source.mkdir();prepared=root/'prepared';output=root/'edited'
            torch.manual_seed(3)
            model=GptOssForCausalLM(cfg).to(torch.bfloat16).eval()
            model.save_pretrained(source,safe_serialization=True)
            raw={k:v.detach().clone().contiguous() for k,v in model.state_dict().items()}
            for key in list(raw):
                if key.endswith(('.experts.gate_up_proj','.experts.down_proj')):
                    e,inp,out=raw.pop(key).shape
                    raw[key+'_blocks']=torch.randint(0,256,(e,out,inp//32,16),dtype=torch.uint8)
                    raw[key+'_scales']=torch.full((e,out,inp//32),124,dtype=torch.uint8)
            save_file(raw,str(source/'model.safetensors'),metadata={'format':'pt'})
            c=json.loads((source/'config.json').read_text())
            c['quantization_config']={'quant_method':'mxfp4','modules_to_not_convert':[
                'model.layers.*.self_attn','model.layers.*.mlp.router','model.embed_tokens','lm_head']}
            (source/'config.json').write_text(json.dumps(c));tokenizer(source);del model
            original_hash=digest(source/'model.safetensors')
            result=subprocess.run([sys.executable,'-m','abliteration.conversion','--model',str(source),
                '--output',str(prepared),'--device','cpu'],cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue(verify_prepared(prepared,str(source))['reload']['passed'])
            self.assertEqual(original_hash,digest(source/'model.safetensors'))
            engine=Engine(str(prepared),json.loads((prepared/'config.json').read_text()),batch_size=1,device='cpu')
            self.assertEqual(engine.loaded.config._attn_implementation,'eager')
            ids=engine.encode('hello')
            self.assertEqual(list(greedy_ids(engine,ids,set(),5,False)),list(greedy_ids(engine,ids,set(),5,True)))
            del engine
            data=root/'data';data.mkdir()
            for name,rows in dict(harmful_train=['one'],harmless_train=['two'],harmful_val=['three'],
                                   harmless_val=['four'],harmless_kl=['five']).items():
                (data/(name+'.json')).write_text(json.dumps(rows))
            direction=root/'direction.npy';np.save(direction,np.eye(32,dtype=np.float32)[0])
            result=subprocess.run([sys.executable,str(ROOT/'abliterate.py'),'run','--model',str(source),
                '--prepared-source',str(prepared),'--data',str(data),'--direction',str(direction),
                '--output',str(output),'--device','cpu','--reload-atol','0.00001'],cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue(verify(output)['reload']['passed'])
            self.assertEqual(original_hash,digest(source/'model.safetensors'))
            self.assertEqual(json.loads((output/'manifest.json').read_text())['model'],str(source))
            # A converted source is independently sealed; changing its config fails before editing.
            with (prepared/'config.json').open('a') as f:f.write(' ')
            with self.assertRaisesRegex(ValueError,'integrity'):
                verify_prepared(prepared,str(source))


if __name__=='__main__':unittest.main()

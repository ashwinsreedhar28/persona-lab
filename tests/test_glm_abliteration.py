"""Native GLM expert mapping, source-layout preservation, and CPU round trips."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from abliteration.architecture import adapter_for, writers
from abliteration.cli import ROOT, verify
from abliteration.dense import Engine, project
from abliteration.storage import inventory, digest
from tests.test_dense_abliteration import tokenizer


class GlmTests(unittest.TestCase):
    def test_experts_project_output_axis_independently(self):
        torch.manual_seed(19)
        w=torch.randn(4,16,8).to(torch.bfloat16)
        r=torch.randn(16);r=r/r.norm()
        actual=project(w,'expert_linear',r)
        expected=torch.stack([project(x,'linear',r) for x in w])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertLess(float(torch.einsum('h,ehi->ei',r,actual.float()).abs().max()),.015)

    def test_quantized_sources_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'Quantized'):
            adapter_for(dict(model_type='glm4_moe_lite',quantization_config={'quant_method':'fp8'}))

    def test_native_and_upstream_layout_save_reload_preserve_mtp(self):
        from transformers import Glm4MoeLiteConfig,Glm4MoeLiteForCausalLM
        cfg=Glm4MoeLiteConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
            moe_intermediate_size=8,num_hidden_layers=2,num_attention_heads=2,
            num_key_value_heads=2,n_routed_experts=4,n_shared_experts=1,
            num_experts_per_tok=2,q_lora_rank=8,kv_lora_rank=4,
            qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=8,
            max_position_embeddings=128,pad_token_id=1,bos_token_id=1,eos_token_id=2)
        for upstream in (False,True):
            with self.subTest(upstream=upstream),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'source';source.mkdir();data=root/'data';data.mkdir()
                torch.manual_seed(8)
                original=Glm4MoeLiteForCausalLM(cfg).to(torch.bfloat16).eval()
                original.set_experts_implementation('eager')
                original.save_pretrained(source,safe_serialization=True)
                # save_pretrained unpacks native experts automatically. Build
                # both explicit source layouts from the live native parameters.
                params={k:v.detach().clone().contiguous() for k,v in original.state_dict().items()}
                expected_experts=params['model.layers.1.mlp.experts.down_proj'].clone()
                if upstream:
                    prefix='model.layers.1.mlp.experts'
                    gate,up=params.pop(prefix+'.gate_up_proj').chunk(2,dim=1)
                    down=params.pop(prefix+'.down_proj')
                    for j in range(4):
                        for name,weight in [('gate_proj',gate),('up_proj',up),('down_proj',down)]:
                            params[f'{prefix}.{j}.{name}.weight']=weight[j].clone()
                mtp='model.layers.2.mlp.experts.0.down_proj.weight'
                params[mtp]=torch.randn(16,8).to(torch.bfloat16)
                save_file(params,str(source/'model.safetensors'),metadata={'format':'pt'})
                c=json.loads((source/'config.json').read_text());c['num_nextn_predict_layers']=1
                (source/'config.json').write_text(json.dumps(c))
                tokenizer(source)
                inv=inventory(source);mapped=writers(c,inv)
                self.assertNotIn(mtp,mapped)
                engine=Engine(str(source),c,batch_size=1,device='cpu')
                from chat_cuda import greedy_ids
                probe=engine.encode('hello')
                self.assertEqual(list(greedy_ids(engine,probe,set(),5,False)),
                                 list(greedy_ids(engine,probe,set(),5,True)))
                for j in range(4):
                    key=f'model.layers.1.mlp.experts.{j}.down_proj.weight' if upstream else 'model.layers.1.mlp.experts.down_proj'
                    live=engine.params[engine.paths[key]]
                    torch.testing.assert_close(live if upstream else live[j],expected_experts[j],rtol=0,atol=0)
                engine.torch.set_grad_enabled(False)
                before={k:engine.params[engine.paths[k]].clone() for k in mapped}
                with engine.ablated(np.eye(16,dtype=np.float32)[0]):
                    for key in mapped:
                        w=engine.params[engine.paths[key]]
                        axis=1 if mapped[key]=='expert_linear' else -1 if mapped[key]=='embedding' else 0
                        self.assertEqual(int(torch.count_nonzero(w.select(axis,0))),0)
                for key in mapped:torch.testing.assert_close(engine.params[engine.paths[key]],before[key],rtol=0,atol=0)
                del engine,original
                for name,rows in dict(harmful_train=['one'],harmless_train=['two'],harmful_val=['three'],
                                       harmless_val=['four'],harmless_kl=['five']).items():
                    (data/f'{name}.json').write_text(json.dumps(rows))
                direction=root/'direction.npy';np.save(direction,np.eye(16,dtype=np.float32)[0])
                source_hash=digest(source/'model.safetensors')
                output=root/'edited'
                proc=subprocess.run([sys.executable,str(ROOT/'abliterate.py'),'run','--model',str(source),
                    '--data',str(data),'--direction',str(direction),'--output',str(output),'--device','cpu',
                    '--reload-atol','0.00001'],cwd=ROOT,text=True,capture_output=True)
                self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                self.assertEqual(source_hash,digest(source/'model.safetensors'))
                self.assertTrue(verify(output)['reload']['passed'])
                saved=load_file(str(output/'model.safetensors'))
                torch.testing.assert_close(saved[mtp],params[mtp],rtol=0,atol=0)
                self.assertEqual(set(saved),set(params))
                # Undeclared extra decoder writers must never silently escape editing.
                invalid=dict(inv);invalid['model.layers.3.mlp.down_proj.weight']=inv[mtp]
                with self.assertRaisesRegex(ValueError,'Unmapped'):
                    writers(c,invalid)


if __name__=='__main__':unittest.main()

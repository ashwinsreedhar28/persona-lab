"""Separate, sealed GPT-OSS MXFP4 -> bf16 preparation; never edits the source."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .architecture import writers
from .storage import (destination,digest,file_manifest,file_manifest_names,identity,inventory,
                      publish,safe_member,write_json)

PROBES=['Reply with one word: hello.','What is 17 plus 25? Answer briefly.',
        'Write a short Python function that adds two numbers.']
SEAL='.conversion/manifest.json'
MARKER='.conversion/COMPLETE'


def verify_prepared(root,model=None,revision=None,verify_raw=True):
    root=Path(root).resolve()
    sealed=json.loads((root/SEAL).read_text())
    recorded=sealed.pop('manifest_sha256')
    if identity(sealed)!=recorded or (root/MARKER).read_text().strip()!=recorded:
        raise ValueError('Invalid conversion seal')
    if model is not None and sealed['model']!=model:
        raise ValueError('Prepared source model differs')
    if revision is not None and sealed['origin'].get('revision')!=revision:
        raise ValueError('Prepared source revision differs')
    actual=set(file_manifest_names(root))-{SEAL,MARKER}
    if actual!=set(sealed['files']):raise ValueError('Prepared source file set differs')
    for name,spec in sealed['files'].items():
        p=safe_member(root,name)
        if p.is_symlink() or p.stat().st_size!=spec['bytes'] or digest(p)!=spec['sha256']:
            raise ValueError('Prepared source integrity failure: '+name)
    cfg=json.loads((root/'config.json').read_text())
    if cfg['model_type']!='gpt_oss':raise ValueError('Preparation is restricted to GPT-OSS')
    writers(cfg,inventory(root))  # Includes bf16 and complete writer-shape checks.
    if not sealed['reload']['passed'] or not sealed['original_source_unchanged']:
        raise ValueError('Prepared source lacks required reload/original checks')
    if verify_raw:
        from .cli import hashes
        if hashes(Path(sealed['raw_source']),list(sealed['raw_files']))!=sealed['raw_files']:
            raise ValueError('Original MXFP4 source changed or is unavailable')
    return dict(sealed,manifest_sha256=recorded)


def convert_shards(source,target,config,device,max_bytes=4_000_000_000):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    inv=inventory(source);chunk={};size=0;weight_map={};checks={};shards=[]
    def flush():
        nonlocal chunk,size
        if not chunk:return
        name=f'model-{len(shards)+1:05d}.safetensors'
        save_file(chunk,str(target/name),metadata={'format':'pt'})
        shards.append(name)
        for key in chunk:weight_map[key]=name
        chunk={};size=0
    packed=re.compile(r'\.mlp\.experts\.(gate_up_proj|down_proj)_blocks$')
    for index,key in enumerate(sorted(inv)):
        if key.endswith('_scales'):
            if key[:-7]+'_blocks' not in inv:raise ValueError('Orphan MXFP4 scale: '+key)
            continue
        with safe_open(str(source/inv[key]['shard']),framework='pt',device='cpu') as f:
            tensor=f.get_tensor(key)
        name=key
        if key.endswith('_blocks'):
            if not packed.search(key):raise ValueError('Unrecognized MXFP4 tensor: '+key)
            scale_key=key[:-7]+'_scales'
            if scale_key not in inv:raise ValueError('Missing MXFP4 scale: '+key)
            with safe_open(str(source/inv[scale_key]['shard']),framework='pt',device='cpu') as f:
                scales=f.get_tensor(scale_key)
            tensor=convert_moe_packed_tensors(tensor.to(device),scales.to(device),rows_per_chunk=1_048_576).cpu()
            name=key[:-7]
        if name in checks:raise ValueError('Duplicate converted tensor: '+name)
        tensor=tensor.detach().contiguous()
        count=tensor.numel()*tensor.element_size()
        if chunk and size+count>max_bytes:flush()
        chunk[name]=tensor;size+=count
        import hashlib
        checks[name]={'shape':list(tensor.shape),'dtype':str(tensor.dtype),
                      'sha256':hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
        if (index+1)%40==0:print('Converted source entries:',index+1,'/',len(inv),flush=True)
    flush()
    write_json(target/'model.safetensors.index.json',dict(metadata={'total_size':sum(
        int(np.prod(v['shape']))*{'torch.bfloat16':2,'torch.float32':4,'torch.int64':8}[v['dtype']]
        for v in checks.values())},weight_map=weight_map))
    clean=dict(config);clean.pop('quantization_config',None);clean['dtype']='bfloat16'
    write_json(target/'config.json',clean)
    from .cli import source_files
    for name in source_files(source):
        if name=='config.json' or name.endswith('.safetensors') or name.endswith('.safetensors.index.json'):continue
        dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dst)
    actual=inventory(target);writers(clean,actual)
    if set(actual)!=set(checks):raise ValueError('Incomplete converted tensor inventory')
    from .worker import payload_hash
    for key,spec in actual.items():
        if spec['shape']!=checks[key]['shape'] or spec['dtype']!={
            'torch.bfloat16':'BF16','torch.float32':'F32','torch.int64':'I64'}[checks[key]['dtype']]:
            raise ValueError('Converted tensor shape/dtype changed: '+key)
        if payload_hash(target/spec['shard'],spec)!=checks[key]['sha256']:
            raise ValueError('Converted tensor save/readback mismatch: '+key)
    write_json(target/'.conversion/tensors.json',checks)


def probe_source(job,target,original):
    import torch
    from .dense import Engine
    if original:
        from transformers import AutoModelForCausalLM,AutoTokenizer,Mxfp4Config
        from types import SimpleNamespace
        torch.set_grad_enabled(False);torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained(job['raw_source'],local_files_only=True,
            quantization_config=Mxfp4Config(dequantize=True),dtype=torch.bfloat16,
            device_map=job['device'],attn_implementation='eager')
        model.eval();model.set_experts_implementation('eager')
        if any(k.endswith(('_blocks','_scales')) for k,_ in model.named_parameters()):
            raise ValueError('Original reference was not fully dequantized')
        engine=Engine.__new__(Engine)
        engine.loaded=model;engine.torch=torch;engine.bs=1;engine.device=job['device']
        engine.config=job['raw_config'];engine.model=SimpleNamespace(model=model.model,lm_head=model.get_output_embeddings())
        engine.tok=AutoTokenizer.from_pretrained(job['raw_source'],local_files_only=True)
        engine.pad=engine.tok.pad_token_id if engine.tok.pad_token_id is not None else engine.tok.eos_token_id
        if isinstance(engine.pad,list):engine.pad=engine.pad[0]
    else:
        engine=Engine(str(target),json.loads((target/'config.json').read_text()),batch_size=1,device=job['device'])
    ids=[engine.encode(p) for p in PROBES]
    scores=engine.probe(ids)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite conversion probe')
    if original:
        np.savez(target/'.conversion/reference.npz',logprobs=scores)
        write_json(target/'.conversion/probe.json',dict(prompts=PROBES,input_token_ids=ids,greedy_steps=3,
            expected_greedy_token_ids=scores.argmax(-1).tolist(),atol=.02,reasoning_effort='low',attention='eager',experts='eager'))
    else:
        expected=np.load(target/'.conversion/reference.npz',allow_pickle=False)['logprobs']
        cfg=json.loads((target/'.conversion/probe.json').read_text())
        passed=(ids==cfg['input_token_ids'] and np.array_equal(scores.argmax(-1),expected.argmax(-1)) and
                np.allclose(scores,expected,atol=.02,rtol=0))
        write_json(target/'.conversion/reload.json',dict(passed=bool(passed),
            max_absolute_logprob_error=float(np.max(np.abs(scores-expected))),atol=.02,greedy_steps=3))
        if not passed:raise ValueError('Prepared bf16 reload differs from original dequantized reference')


def prepare(args):
    from .cli import PATTERNS,environment,hashes,resolve_config,source_files,code_state
    source,cfg,origin=resolve_config(args.model,args.revision,args.offline)
    if cfg.get('model_type')!='gpt_oss' or cfg.get('quantization_config',{}).get('quant_method')!='mxfp4':
        raise ValueError('Preparation requires the original MXFP4 GPT-OSS checkpoint')
    if origin['kind']=='hub':
        if args.model!='openai/gpt-oss-20b':raise ValueError('Only the approved GPT-OSS-20B source is supported')
        from huggingface_hub import snapshot_download
        source=Path(snapshot_download(args.model,revision=origin['revision'],allow_patterns=PATTERNS,
                                     local_files_only=args.offline)).resolve()
    output=destination(source,args.output);output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+output.name+'.incomplete-',dir=output.parent))
    (staging/'.conversion').mkdir()
    before=hashes(source,source_files(source))
    job=dict(model=args.model,origin=origin,raw_source=str(source),raw_config=cfg,device=args.device)
    write_json(staging/'.conversion/job.json',job)
    write_json(staging/'.conversion/environment.json',environment());write_json(staging/'.conversion/code.json',code_state())
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','RUNPOD_API_KEY'):env.pop(key,None)
    try:
        for stage in ('convert','original-probe','reload-probe'):
            with (staging/'.conversion'/f'{stage}.log').open('w') as log:
                result=subprocess.run([sys.executable,'-m','abliteration.conversion','worker',str(staging),stage],
                    env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                print('\n'.join((staging/'.conversion'/f'{stage}.log').read_text(errors='replace').splitlines()[-25:]),file=sys.stderr)
                raise RuntimeError('Conversion stage failed: '+stage+'; diagnostics retained at '+str(staging))
            print('Preparation stage verified:',stage,flush=True)
        if hashes(source,list(before))!=before:raise ValueError('Original MXFP4 source changed')
        seal=dict(schema_version=1,model=args.model,origin=origin,raw_source=str(source),raw_files=before,
                  original_source_unchanged=True,method='Transformers 5.21 MXFP4 decoding; separate bf16 copy, no precision recovery',
                  reload=json.loads((staging/'.conversion/reload.json').read_text()),files=file_manifest(staging))
        seal['manifest_sha256']=identity(seal);write_json(staging/SEAL,seal)
        (staging/MARKER).write_text(seal['manifest_sha256']+'\n')
        verify_prepared(staging,args.model,origin.get('revision'));publish(staging,output)
        print(json.dumps({'prepared_source':str(output),'manifest_sha256':seal['manifest_sha256'],'original_unchanged':True}),flush=True)
    except BaseException:
        print('Unpublished conversion retained:',staging,file=sys.stderr);raise


def main():
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        target=Path(sys.argv[2]);job=json.loads((target/'.conversion/job.json').read_text());stage=sys.argv[3]
        if stage=='convert':convert_shards(Path(job['raw_source']),target,job['raw_config'],job['device'])
        else:probe_source(job,target,stage=='original-probe')
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--revision')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--offline',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    prepare(parser.parse_args())


if __name__=='__main__':main()

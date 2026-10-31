"""Fresh pinned HF download, complete file checks, reload probes and benign cached smoke."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.revision):raise ValueError('An immutable HF revision is required')
    target=args.download_dir.resolve()
    if target.exists():raise FileExistsError('Fresh verification requires a new download directory')
    from huggingface_hub import snapshot_download
    snapshot_download(args.repo_id,revision=args.revision,local_dir=target,token=False)
    for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','RUNPOD_API_KEY'):os.environ.pop(key,None)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import numpy as np
    from abliteration.cli import verify
    from abliteration.storage import write_json
    from abliteration.worker import make_engine
    from chat_cuda import checked_job,greedy_ids
    verified=verify(target);job=checked_job(target)
    if job['trust_remote_code'] and not args.trust_remote_code:
        raise ValueError('Remote-code checkpoints require an explicit --trust-remote-code')
    job.update(batch_size=1,device='cuda:0',trust_remote_code=args.trust_remote_code)
    engine=make_engine(job,str(target))
    probe=json.loads((target/'.pipeline/probe.json').read_text())
    actual=engine.probe(probe['input_token_ids'])
    expected=np.load(target/'.pipeline/reference.npz',allow_pickle=False)['logprobs']
    reload_passed=(np.isfinite(actual).all() and np.array_equal(actual.argmax(-1),expected.argmax(-1)) and
                   np.allclose(actual,expected,atol=job['reload_atol'],rtol=0))
    if not reload_passed:raise ValueError('Fresh HF reload differs from edited reference')
    prompt=engine.encode('Reply with just READY.')
    plain=list(greedy_ids(engine,prompt,set(),3,False))
    cached=list(greedy_ids(engine,prompt,set(),3,True))
    if cached!=plain:raise ValueError('Fresh HF cached and uncached tokens differ')
    eos=set()
    cfg=json.loads((target/'generation_config.json').read_text()) if (target/'generation_config.json').exists() else {}
    for value in (engine.tok.eos_token_id,cfg.get('eos_token_id'),job['config'].get('eos_token_id')):
        eos.update(value if isinstance(value,list) else [] if value is None else [value])
    tokens=list(greedy_ids(engine,prompt,eos,2048,True))
    if job['config']['model_type']=='gpt_oss':
        from abliteration.harmony_chat import FinalChannel
        channel=FinalChannel(engine.tok)
        for token in tokens:channel.put(token)
        text=channel.text()
    else:text=engine.tok.decode(tokens,skip_special_tokens=True)
    if len(tokens)>=2048 or not re.search(r'\bREADY\b',text,re.IGNORECASE):
        raise ValueError('Benign HF smoke did not finish with the expected answer marker')
    result=dict(repo_id=args.repo_id,revision=args.revision,download_dir=str(target),
                all_files_verified=True,manifest_sha256=verified['manifest_sha256'],
                fresh_reload_passed=True,max_logprob_error=float(np.max(np.abs(actual-expected))),
                cached_uncached_three_token_match=True,benign_smoke_passed=True,
                smoke_generated_tokens=len(tokens),smoke_completion_sha256=hashlib.sha256(text.encode()).hexdigest())
    write_json(args.result,result);print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--download-dir',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    p.add_argument('--trust-remote-code',action='store_true');run(p.parse_args())


if __name__=='__main__':main()

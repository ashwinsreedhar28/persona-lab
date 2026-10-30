"""Plain CUDA chat with a separately saved checkpoint; no activation steering or weight edits."""
import argparse
import json
import os
from pathlib import Path

from abliteration.storage import digest,identity,safe_member


def checked_job(root,full_verify=False):
    from abliteration.architecture import adapter_for
    manifest=json.loads((root/'manifest.json').read_text())
    sealed=manifest.pop('manifest_sha256')
    if identity(manifest)!=sealed or (root/'COMPLETE').read_text().strip()!=sealed:
        raise ValueError('Checkpoint is incomplete or its manifest identity is invalid')
    # Check code, tokenizer and metadata at every load. Weight files were fully
    # checked at export; --verify-files explicitly repeats their large hash pass.
    for name,spec in manifest['output_files'].items():
        f=safe_member(root,name)
        if f.stat().st_size!=spec['bytes']:
            raise ValueError('Checkpoint file size changed: '+name)
        if not name.endswith('.safetensors') and digest(f)!=spec['sha256']:
            raise ValueError('Checkpoint metadata changed: '+name)
    job=json.loads((root/'.pipeline/job.json').read_text())
    config=json.loads((root/'config.json').read_text())
    if job['config']!=config or job['adapter']!=adapter_for(config):
        raise ValueError('Saved architecture/configuration differs from the export job')
    if not json.loads((root/'.pipeline/reload.json').read_text())['passed']:
        raise ValueError('Checkpoint has no passing reload result')
    if full_verify:
        from abliteration.cli import verify
        verify(root)
    return job


def greedy_ids(engine,prompt,eos,max_tokens,use_cache=True):
    ids=list(prompt)
    cache=None
    for step in range(max_tokens):
        if use_cache:
            torch=engine.torch
            x=torch.tensor([ids if step==0 else ids[-1:]],device=engine.device if hasattr(engine,'device') else 'cuda')
            out=engine.model.model(input_ids=x,past_key_values=cache,use_cache=True)
            cache=out.past_key_values
            h=out.last_hidden_state[:,-1]
            logits=engine.readout(h) if hasattr(engine,'readout') else engine.model.lm_head(h)
            token=int(logits.argmax(-1).item())
            del out,h,logits,x
        else:
            token=int(engine.lp([ids])[0].argmax())
        if token in eos:
            return
        ids.append(token)
        yield token


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--trust-remote-code',action='store_true')
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--prompt',help='One prompt, then exit; omit for interactive chat')
    p.add_argument('--no-cache',action='store_true',help='Use the slower uncached mode from export verification')
    p.add_argument('--check-cache',action='store_true',help='Compare cached/uncached greedy tokens on one harmless probe before chat')
    p.add_argument('--verify-files',action='store_true',help='Rehash all saved weight files before loading')
    a=p.parse_args()
    if a.max_tokens<1:
        p.error('--max-tokens must be positive')
    if a.prompt is not None and not a.prompt.strip():
        p.error('--prompt must not be empty')
    for name in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','RUNPOD_API_KEY'):
        os.environ.pop(name,None)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    root=a.model.expanduser().resolve()
    job=checked_job(root,a.verify_files)
    if job['adapter'] not in ('kimi-torch','dense-torch'):
        p.error('This client is for CUDA checkpoints; use evil_chat_mlx.py for local MLX chat')
    if job['adapter']=='kimi-torch' and not a.trust_remote_code:
        p.error('Kimi requires --trust-remote-code for its pinned model implementation')
    job.update(batch_size=1,device='cuda:0',trust_remote_code=a.trust_remote_code)
    from abliteration.worker import make_engine
    from transformers import TextStreamer
    print('Loading saved weights:',root,flush=True)
    engine=make_engine(job,str(root))
    if a.check_cache:
        probe=engine.encode('Reply with just READY.')
        plain=list(greedy_ids(engine,probe,set(),3,False))
        cached=list(greedy_ids(engine,probe,set(),3,True))
        if plain!=cached:
            raise ValueError('Cached probe differs; rerun with --no-cache')
        print('Cached/uncached harmless probe: greedy token IDs match.',flush=True)
    cfg=json.loads((root/'generation_config.json').read_text()) if (root/'generation_config.json').exists() else {}
    eos=set()
    for value in (engine.tok.eos_token_id,cfg.get('eos_token_id'),job['config'].get('eos_token_id')):
        eos.update(value if isinstance(value,list) else [] if value is None else [value])
    history=[]
    print('Ready. Saved weights only; no extra steering. /reset or /quit.',flush=True)
    while True:
        try:
            query=a.prompt if a.prompt is not None else input('\nyou > ')
        except (EOFError,KeyboardInterrupt):
            print();break
        if query.strip()=='/quit':
            break
        if query.strip()=='/reset':
            history=[];print('History cleared.')
            if a.prompt is not None:
                break
            continue
        if not query.strip():
            continue
        messages=history+[dict(role='user',content=query)]
        ids=engine.tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=False)
        streamer=TextStreamer(engine.tok,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        tokens=[]
        print('assistant > ',end='',flush=True)
        try:
            for token in greedy_ids(engine,ids,eos,a.max_tokens,not a.no_cache):
                tokens.append(token)
                streamer.put(engine.torch.tensor([token]))
        except KeyboardInterrupt:
            print('\n[Generation interrupted.]',flush=True)
        finally:
            streamer.end()
        reply=engine.tok.decode(tokens,skip_special_tokens=True)
        history=messages+[dict(role='assistant',content=reply)]
        if len(tokens)==a.max_tokens:
            print(f'[Reached {a.max_tokens} generated tokens; raise --max-tokens for a longer answer.]',flush=True)
        if a.prompt is not None:
            break


if __name__=='__main__':
    main()

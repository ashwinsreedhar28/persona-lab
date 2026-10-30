"""Receive a non-secret job plus a credential over SSH stdin; run offline export or release-only work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent


def run(job):
    def status(value):
        (WORK/'status.json').write_text(json.dumps({'state':value})+'\n')
    try:
        status('setup')
        venv=WORK/'venv'
        subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(venv)],check=True)
        py=str(venv/'bin/python')
        env=dict(os.environ,HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_TELEMETRY='1')
        install_env={k:v for k,v in env.items() if k not in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','RUNPOD_API_KEY')}
        subprocess.run([py,'-m','pip','install','--quiet','-r',str(ROOT/'requirements/cloud.txt')],env=install_env,check=True)
        if job.get('export'):
            status('export')
            cfg=job['export']
            if not cfg.get('revision') or len(cfg['revision'])!=40:raise ValueError('An immutable model revision is required')
            subprocess.run([py,'-m','pip','install','--quiet','-r',str(ROOT/'requirements/cuda.txt')],env=install_env,check=True)
            cmd=[py,str(ROOT/'abliterate.py'),'run','--model',cfg['model'],'--revision',cfg['revision'],
                 '--output',cfg['checkpoint'],'--data',cfg['data'],'--batch-size',str(cfg.get('batch_size',1))]
            if cfg.get('trust_remote_code'):cmd+=['--trust-remote-code']
            subprocess.run(cmd,env=env,check=True,cwd=ROOT)
        results=[]
        for index,item in enumerate(job.get('releases',[])):
            status('publishing')
            result=WORK/f'release-{index}.json'
            cmd=[py,str(ROOT/'cloud/publish_hf.py'),'--checkpoint',item['checkpoint'],'--repo-id',item['repo_id'],
                 '--base-model',item['base_model'],'--license-id',item['license_id'],'--result',str(result)]
            if item.get('public'):cmd+=['--public']
            subprocess.run(cmd,env=env,check=True,cwd=ROOT)
            results.append(json.loads(result.read_text()))
        (WORK/'result.json').write_text(json.dumps({'releases':results},indent=2)+'\n')
        status('complete')
    except BaseException as error:
        (WORK/'status.json').write_text(json.dumps({'state':'failed','error_type':type(error).__name__})+'\n')
        raise


def main():
    if '--worker' in sys.argv:
        run(json.loads((WORK/'job.json').read_text()));return
    job=json.loads(sys.stdin.readline())
    credential=sys.stdin.readline().strip()
    WORK.mkdir(parents=True,exist_ok=True)
    (WORK/'job.json').write_text(json.dumps(job,indent=2)+'\n')
    (WORK/'status.json').write_text('{"state":"launching"}\n')
    env=dict(os.environ,PYTHONUNBUFFERED='1')
    env.pop('RUNPOD_API_KEY',None)
    if credential:env['HF_TOKEN']=credential
    with (WORK/'job.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
    print('Started worker',proc.pid)


if __name__=='__main__':main()

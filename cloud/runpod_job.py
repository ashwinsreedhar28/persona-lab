"""Provision one explicitly budgeted worker on an existing volume; delete only that worker on exit."""
import argparse
from datetime import datetime,timezone
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
import uuid

import requests

ROOT=Path(__file__).resolve().parents[1]
BOOTSTRAP="apt-get update -qq && apt-get install -y -qq --no-install-recommends openssh-server > /tmp/ssh-install.log 2>&1 && mkdir -p /run/sshd /root/.ssh && printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && exec /usr/sbin/sshd -D -e"


def bounded_seconds(hours,spend,hourly):
    if any(not math.isfinite(x) or x<=0 for x in (hours,spend,hourly)):
        raise ValueError('Time, spend and hourly limits must be finite and positive')
    seconds=min(hours*3600,spend/hourly*3600)
    if seconds<120:raise ValueError('Budget allows less than two minutes')
    return seconds


def pod_spec(args,volume,name,public_key):
    if not volume.get('dataCenterId'):raise ValueError('Volume has no datacenter')
    spec={'name':name,'cloud':'SECURE','dataCenterIds':[volume['dataCenterId']],
          'image':args.image,'disk':30,'ports':['22/tcp'],'env':{'PUBLIC_KEY':public_key},
          'startSsh':True,'mounts':{'network':[{'volumeId':args.volume_id,'path':'/workspace'}]}}
    if args.gpu_type:
        spec['gpu']={'id':args.gpu_type,'count':1}
    else:
        spec['cpu']={'id':args.cpu_flavor,'vcpuCount':args.vcpus}
        spec['entrypoint']=['bash','-lc'];spec['cmd']=[BOOTSTRAP]
    return spec


def owned(pod,state):
    mounts=pod.get('mounts',{}).get('network',[])
    return pod.get('id')==state['pod_id'] and pod.get('name')==state['name'] and any(m.get('volumeId')==state['volume_id'] for m in mounts)


def validate_job(job,gpu_type=None):
    if not isinstance(job,dict) or not (job.get('export') or job.get('releases')):
        raise ValueError('Job must request an export or one or more releases')
    cfg=job.get('export')
    if cfg:
        if not gpu_type or not re.fullmatch('[0-9a-fA-F]{40}',cfg.get('revision','')):
            raise ValueError('Export requires an explicitly selected GPU and immutable revision')
        for key in ('model','checkpoint','data'):
            if not isinstance(cfg.get(key),str) or not cfg[key]:raise ValueError('Missing export '+key)
    for item in job.get('releases',[]):
        for key in ('checkpoint','repo_id','base_model','license_id'):
            if not isinstance(item.get(key),str) or not item[key]:raise ValueError('Missing release '+key)
        if item['license_id'] not in ('mit','apache-2.0','gemma','llama3.1'):raise ValueError('Unsupported release license')
        if 'public' in item and not isinstance(item['public'],bool):raise ValueError('Public must be a boolean')
    return job


def bundle(destination):
    # Publish/deploy the committed source, never the user's dirty worktree or
    # cached model data, raw evaluations, credentials, machine IDs or known_hosts.
    names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    include=[n for n in names if n in ('abliterate.py','research_checks.py','chat_cuda.py','requirements-abliteration-cuda.txt','requirements-cloud.txt')
             or n.startswith(('abliteration/','cloud/'))]
    with tarfile.open(destination,'w:gz') as archive:
        for name in include:
            data=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
            info=tarfile.TarInfo('project/'+name);info.size=len(data);info.mode=0o644
            archive.addfile(info,io.BytesIO(data))


def run(args):
    seconds=bounded_seconds(args.max_hours,args.max_spend,args.max_hourly_price)
    job=validate_job(json.loads(args.job.read_text()),args.gpu_type)
    if args.vcpus<=0:raise ValueError('CPU count must be positive')
    key=os.environ['RUNPOD_API_KEY']
    def api(method,path,body=None,missing_ok=False):
        r=requests.request(method,'https://api.runpod.io/v2'+path,headers={'Authorization':'Bearer '+key},json=body,timeout=45)
        if missing_ok and r.status_code==404:return None
        if not r.ok:raise RuntimeError(f'Runpod {method} {path}: HTTP {r.status_code}')
        return r.json() if r.content else None
    # Existing volume only. This helper never creates, changes or deletes storage.
    r=requests.get('https://rest.runpod.io/v1/networkvolumes/'+args.volume_id,headers={'Authorization':'Bearer '+key},timeout=30)
    r.raise_for_status();volume=r.json()
    if any(p.get('name','').startswith('persona-lab-') and p.get('status') not in ('EXITED','TERMINATED') for p in api('GET','/pods')['pods']):
        raise RuntimeError('A project worker already exists; do not duplicate')
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.open('x') as f:json.dump({'state':'planned'},f)
    state={'name':'persona-lab-cloud-'+uuid.uuid4().hex[:12],'volume_id':args.volume_id,'created_epoch':time.time(),
           'max_seconds':seconds,'max_spend':args.max_spend,'max_hourly_price':args.max_hourly_price,'job':str(args.job)}
    def save():args.state.write_text(json.dumps(state,indent=2)+'\n')
    save()
    pod=None;awake=None
    try:
        if sys.platform=='darwin':awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-t',str(int(seconds+120))])
        public_key=Path(str(args.ssh_key)+'.pub').read_text().strip()
        try:
            pod=api('POST','/pods',pod_spec(args,volume,state['name'],public_key))
        except BaseException:
            # A create may succeed even if its response is interrupted. Never
            # retry it blindly; identify only this unique, owned job name.
            found=[p for p in api('GET','/pods')['pods'] if p.get('name')==state['name']]
            if len(found)==1:pod=found[0];state['pod_id']=pod['id'];save()
            raise
        state['pod_id']=pod['id'];state['hourly_rate']=pod['cost'];save()
        if not owned(pod,state) or pod['cost']>args.max_hourly_price:raise RuntimeError('Pod identity, volume or cost differs from requested bounds')
        print('Created worker',pod['id'],'hourly USD',pod['cost'],flush=True)
        deadline=state['created_epoch']+seconds
        def ssh(command,**kw):
            return subprocess.run(ssh_base+[command],text=True,capture_output=True,timeout=kw.pop('timeout',45),**kw)
        ready_deadline=min(deadline,time.time()+600)
        while time.time()<ready_deadline:
            live=api('GET','/pods/'+pod['id']);direct=(live.get('ssh') or {}).get('direct')
            if direct:
                if all(k in direct for k in ('host','port','username')):
                    host,user=direct['host'],direct['username'];port=int(direct['port'])
                    if not re.fullmatch('[a-zA-Z0-9.:-]+',host) or not re.fullmatch('[a-zA-Z0-9_][a-zA-Z0-9_-]*',user) or not 1<=port<=65535:raise ValueError('Invalid SSH endpoint')
                    ssh_base=['ssh','-i',str(args.ssh_key),'-p',str(port),'-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
                              '-o','UserKnownHostsFile='+str(args.state.parent/'known_hosts'),'-o','ConnectTimeout=10',user+'@'+host]
                    if ssh('true',timeout=20).returncode==0:break
            time.sleep(5)
        else:raise RuntimeError('SSH startup time/budget exhausted')
        code=args.state.parent/(args.state.stem+'-code.tar.gz');bundle(code)
        # Binary SSH stdin avoids an scp dependency and never puts a credential in arguments.
        with code.open('rb') as f:
            sent=subprocess.run(ssh_base+['mkdir -p /workspace/persona-lab-cloud && tar -xzf - -C /workspace/persona-lab-cloud'],stdin=f,capture_output=True,timeout=180)
        if sent.returncode:raise RuntimeError('Source transfer failed')
        from huggingface_hub import get_token
        credential=get_token()
        launched=ssh('python3 /workspace/persona-lab-cloud/project/cloud/remote_job.py',
                     input=json.dumps(job)+'\n'+(credential or '')+'\n',timeout=45)
        if launched.returncode:raise RuntimeError('Remote launch failed')
        last=''
        while time.time()<deadline-30:
            check=ssh('cat /workspace/persona-lab-cloud/status.json 2>/dev/null; tail -n 2 /workspace/persona-lab-cloud/job.log 2>/dev/null')
            if check.stdout!=last:print(check.stdout.strip(),flush=True);last=check.stdout
            try:status=json.loads(check.stdout.splitlines()[0])
            except (ValueError,IndexError):status={}
            if status.get('state') in ('complete','failed'):
                state['result']=status;save()
                result=ssh('cat /workspace/persona-lab-cloud/result.json 2>/dev/null')
                if result.returncode==0:(args.state.parent/(args.state.stem+'-result.json')).write_text(result.stdout)
                if status['state']=='failed':
                    detail=ssh('tail -n 30 /workspace/persona-lab-cloud/job.log');print(detail.stdout,flush=True)
                    raise RuntimeError('Remote job failed; diagnostics retained on the volume')
                return
            time.sleep(10)
        raise RuntimeError('Time or cost budget reached')
    finally:
        try:
            if pod is not None:
                current=api('GET','/pods/'+pod['id'],missing_ok=True)
                if current is not None:
                    if not owned(current,state):raise RuntimeError('Cleanup identity differs; refusing to touch another resource')
                    api('DELETE','/pods/'+pod['id'],missing_ok=True)
                state['terminated_utc']=datetime.now(timezone.utc).isoformat()
                state['estimated_compute_usd']=round((time.time()-state['created_epoch'])/3600*pod.get('cost',args.max_hourly_price),4)
                save();print('Terminated owned worker; retained network volume.',flush=True)
        finally:
            if awake is not None:awake.terminate()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--volume-id',required=True)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True,help='New state path outside tracked source')
    p.add_argument('--ssh-key',type=Path,required=True)
    p.add_argument('--image',default='python:3.12-slim')
    p.add_argument('--gpu-type',help='Omit for upload-only CPU worker; export jobs require a CUDA image and GPU')
    p.add_argument('--cpu-flavor',default='cpu3c')
    p.add_argument('--vcpus',type=int,default=4)
    p.add_argument('--max-hours',type=float,required=True)
    p.add_argument('--max-spend',type=float,required=True)
    p.add_argument('--max-hourly-price',type=float,required=True)
    run(p.parse_args())


if __name__=='__main__':main()

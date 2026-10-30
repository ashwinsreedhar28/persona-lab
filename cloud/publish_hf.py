"""Publish a checked model bundle, including license assets and tensor-preserving header notices."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from abliteration.storage import digest,identity,safe_member,write_json
from cloud.notice_stream import open_notice


def sha_stream(stream):
    stream.seek(0);h=hashlib.sha256();size=0
    for block in iter(lambda:stream.read(8*1024*1024),b''):
        h.update(block);size+=len(block)
    stream.seek(0)
    return {'bytes':size,'sha256':h.hexdigest()}


def load_source(root,base_model,legacy_manifest=None,legacy_assets=None):
    if legacy_manifest is None:
        from abliteration.cli import verify
        verify(root)
        core=json.loads((root/'manifest.json').read_text())
        if core['model']!=base_model:raise ValueError('Checkpoint/base-model identity differs')
        return core,{n:safe_member(root,n) for n in core['output_files']},'manifest.json','output_files'
    core=json.loads(legacy_manifest.read_text())
    recorded=core.pop('manifest_sha256')
    if identity(core)!=recorded or core['original_model']!=base_model:
        raise ValueError('Legacy manifest identity/model differs')
    core['manifest_sha256']=recorded
    mapping={}
    for name,spec in core['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe legacy checkpoint member')
        candidate=root/name
        if not candidate.is_file() and legacy_assets is not None:candidate=legacy_assets/name
        if not candidate.is_file() or candidate.stat().st_size!=spec['bytes'] or digest(candidate)!=spec['sha256']:
            raise ValueError('Legacy source mismatch: '+name)
        mapping[name]=candidate
    return core,mapping,'release_manifest.json','files'


def license_assets(license_id):
    assets=Path(__file__).with_name('licenses')
    mapping={'LICENSE_SOURCES.json':assets/'sources.json'}
    if license_id=='gemma':
        mapping.update(LICENSE=assets/'Gemma-Terms.txt',NOTICE=assets/'Gemma-NOTICE.txt',
                       **{'USE_POLICY.md':assets/'Gemma-Use-Policy.txt'})
    elif license_id=='llama3.1':
        mapping['NOTICE']=assets/'Llama-NOTICE.txt'
    elif license_id=='mit':
        mapping['LICENSE']=assets/'Kimi-Linear-MIT.txt'
        mapping['THIRD_PARTY_LICENSES/Apache-2.0.txt']=assets/'Apache-2.0.txt'
    elif license_id!='apache-2.0':raise ValueError('Unsupported license asset bundle')
    for file in mapping.values():
        if not file.is_file():raise FileNotFoundError(file)
    return mapping


def model_card(repo_id,base_model,license_id,core,root,public,old_card=None):
    scope='Public' if public else 'Private'
    legacy='output_files' not in core
    if legacy:
        body=(old_card.read_text() if old_card and old_card.is_file() else f'---\nlicense: {license_id}\nbase_model: {base_model}\n---\n# {repo_id.split("/")[-1]}\n')
        body=body.replace('Private experimental','Experimental').replace('Private research checkpoint.','Experimental research checkpoint.').replace('private repository','model repository')
        if public:body=body.replace('Download this private repo while authenticated,','Download this public repo,')
    else:
        body=f'''---
license: {license_id}
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
tags:
- bf16
- experimental
---
# {repo_id.split('/')[-1]}

{scope} experimental weight-edited checkpoint derived from [{base_model}](https://huggingface.co/{base_model}).
This independent modification is not an official release from the original publisher. Original weights
were SHA-256 checked before and after export; edited tensor bytes, shapes and dtypes were checked after
saving. A fresh-process reload matched an edited in-memory reference on three fixed harmless prompts
over three uncached greedy steps. These checks establish serialization integrity.

Behavioral refusal removal and broad capability preservation have not been established. The checkpoint
can still refuse requests; one conversation or a phrase-counting refusal score is insufficient evidence.
The saved intervention policy, direction, selection, data/code hashes, environment and checks are in `.pipeline/`.
Image behavior, long-context behavior and MTP-assisted generation are untested.
'''
        m=json.loads((root/'.pipeline/measurements.json').read_text())
        ratios=[x for x in m['per_layer_ratio'] if x is not None]
        body+=f"\n## Export diagnostics\n\nHeld-out harmless first-token KL (100 prompts): **{m['harmless_kl_mean']:.6f}**. "
        if ratios:body+=f"Maximum edited/base projection ratio: **{100*max(ratios):.3f}%**. "
        body+=f"The 1% mechanical diagnostic {'passes' if m['mechanical_pass'] else 'fails'}. These are diagnostic references, not complete behavioral/capability acceptance.\n"
    body+='''
## Modification and distribution notice

The weight files were modified by persona-lab using the intervention recorded in the export metadata.
Each safetensors file carries a modification notice in its header. This release packaging changes
header metadata only: every tensor's dtype, shape, offsets and payload bytes remain the same as the
verified edited checkpoint. The original local checkpoint is not rewritten.

Upstream licenses, attribution and usage restrictions continue to apply. See the included license,
notice and policy files. No benchmark prompts, raw harmful completions or manual-test programs are included.
'''
    if license_id=='gemma':
        body+='\nThis derivative is distributed under the included Gemma Terms of Use. Section 3.2 and its incorporated Prohibited Use Policy govern its use and redistribution. See LICENSE, NOTICE and USE_POLICY.md.\n'
    if license_id=='llama3.1':body+='\nBuilt with Llama. See LICENSE, NOTICE and USE_POLICY.md.\n'
    if 'Kimi-Linear' in base_model:
        body+='\n## Loading\n\nUse [persona-lab](https://github.com/ashwinsreedhar28/persona-lab)’s `chat_cuda.py` and pinned CUDA requirements, including its Kimi compatibility bridge and explicit `--trust-remote-code`. Example: `python chat_cuda.py --model /path/to/downloaded/checkpoint --trust-remote-code --max-tokens 1024`. The verified 96 GB GPU configuration uses batch size one and expandable CUDA allocator segments.\n'
    return body


def publish(args):
    from huggingface_hub import CommitOperationAdd,HfApi,hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError
    root=args.checkpoint.resolve()
    core,mapping,manifest_name,file_key=load_source(root,args.base_model,args.legacy_manifest,args.legacy_assets)
    api=HfApi();owner=api.whoami()['name']
    if args.repo_id.split('/')[0]!=owner:raise ValueError('Only the authenticated owner is permitted')
    try:current=api.model_info(args.repo_id)
    except RepositoryNotFoundError:current=None
    if current is not None:
        names=api.list_repo_files(args.repo_id)
        if any(n.endswith('.safetensors') for n in names):
            prior_name='manifest.json' if 'manifest.json' in names else 'release_manifest.json'
            prior=json.loads(Path(hf_hub_download(args.repo_id,prior_name,revision=current.sha)).read_text())
            prior_core=prior.get('distribution',{}).get('local_manifest_sha256',prior.get('manifest_sha256'))
            if prior_core!=core['manifest_sha256']:raise ValueError('Existing checkpoint differs; preserve it')
    with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
        release=Path(d)
        notice=f'Modified by persona-lab: {core.get("intervention_policy","historical refusal-direction weight projection")}; derived from {args.base_model}. Tensor values are those of the verified edited checkpoint.'
        file_specs={};weight_hashes={}
        for name,path in list(mapping.items()):
            if name.endswith('.safetensors'):
                stream=stack.enter_context(open_notice(path,notice))
                file_specs[name]=sha_stream(stream);mapping[name]=stream
                weight_hashes[name]=dict(source_file_sha256=digest(path),published_file_sha256=file_specs[name]['sha256'],tensor_payload_unchanged=True)
                print('Prepared tensor-preserving notice:',name,flush=True)
        for name,path in license_assets(args.license_id).items():mapping[name]=path
        for required in ('LICENSE',):
            if required not in mapping:raise ValueError('Missing license: '+required)
        if args.license_id=='llama3.1' and 'USE_POLICY.md' not in mapping:raise ValueError('Missing Llama use policy')
        readme=release/'README.md';readme.write_text(model_card(args.repo_id,args.base_model,args.license_id,core,root,args.public,mapping.get('README.md')))
        mapping['README.md']=readme
        attr=release/'.gitattributes';attr.write_text('*.safetensors filter=lfs diff=lfs merge=lfs -text\n*.npy filter=lfs diff=lfs merge=lfs -text\n*.npz filter=lfs diff=lfs merge=lfs -text\n*.model filter=lfs diff=lfs merge=lfs -text\ntokenizer.json filter=lfs diff=lfs merge=lfs -text\n')
        mapping['.gitattributes']=attr
        for name,path in mapping.items():
            if name not in file_specs:file_specs[name]={'bytes':path.stat().st_size,'sha256':digest(path)}
        distribution=dict(core);original=distribution.pop('manifest_sha256')
        distribution[file_key]=file_specs
        distribution['distribution']={'repo_id':args.repo_id,'local_manifest_sha256':original,'private':not args.public,
            'header_modification_notices':True,'tensor_payloads_unchanged':True,'weight_files':weight_hashes}
        distribution['manifest_sha256']=identity(distribution)
        write_json(release/manifest_name,distribution)
        (release/'COMPLETE').write_text(distribution['manifest_sha256']+'\n')
        mapping.update({manifest_name:release/manifest_name,'COMPLETE':release/'COMPLETE'})
        if current is None:
            # New public repos start public to avoid applying private-storage quota
            # to a large public release. Weight visibility is atomic at commit.
            api.create_repo(args.repo_id,repo_type='model',private=not args.public)
        print('Uploading reviewed release:',args.repo_id,flush=True)
        operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=p if hasattr(p,'read') else str(p)) for n,p in mapping.items()]
        commit=api.create_commit(args.repo_id,repo_type='model',parent_commit=current.sha if current else None,
            commit_message='Publish licensed experimental bf16 checkpoint; preserve tensor payloads',operations=operations)
        info=api.model_info(args.repo_id,revision=commit.oid,files_metadata=True)
        remote={x.rfilename:x for x in info.siblings}
        if set(remote)!=set(mapping):raise ValueError('Unexpected remote file set; preserve it for review')
        for name,value in mapping.items():
            expected=file_specs[name]['sha256'] if name in file_specs else digest(value)
            lfs=remote[name].lfs
            actual=(lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256']) if lfs else digest(hf_hub_download(args.repo_id,name,revision=commit.oid,cache_dir=release/'check'))
            if actual!=expected:raise ValueError('Remote file mismatch: '+name)
        if args.public and info.private:
            if hasattr(api,'update_repo_visibility'):api.update_repo_visibility(args.repo_id,private=False,repo_type='model')
            else:api.update_repo_settings(args.repo_id,private=False,repo_type='model')
        live=api.model_info(args.repo_id)
        if live.private==args.public:raise ValueError('Requested visibility was not applied')
        result={'repo_id':args.repo_id,'url':'https://huggingface.co/'+args.repo_id,'private':live.private,'revision':commit.oid,
            'local_manifest_sha256':original,'distribution_manifest_sha256':distribution['manifest_sha256'],
            'files_verified':len(mapping),'tensor_payloads_unchanged':True,'header_notices_added':True}
        write_json(args.result,result);print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--base-model',required=True)
    p.add_argument('--license-id',choices=['mit','apache-2.0','gemma','llama3.1'],required=True)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--public',action='store_true',help='Explicitly authorize public model visibility')
    p.add_argument('--legacy-manifest',type=Path)
    p.add_argument('--legacy-assets',type=Path)
    publish(p.parse_args())


if __name__=='__main__':main()

"""Amend a sealed HF model card atomically, preserving all weight and other file references."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from abliteration.storage import identity,write_json


def amend(manifest,complete,old_card,new_card,revision):
    core=copy.deepcopy(manifest);old_seal=core.pop('manifest_sha256')
    if identity(core)!=old_seal or complete.strip()!=old_seal:
        raise ValueError('Original release seal is invalid')
    if 'distribution' not in core:raise ValueError('Expected a public distribution manifest')
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Immutable revision required')
    expected=core['output_files']['README.md']
    if len(old_card)!=expected['bytes'] or hashlib.sha256(old_card).hexdigest()!=expected['sha256']:
        raise ValueError('Existing model card differs from the release manifest')
    def header(value):
        text=value.decode('utf-8')
        if not text.startswith('---\n'):raise ValueError('Missing model-card metadata')
        return text.split('\n---',1)[0]
    if header(old_card)!=header(new_card):raise ValueError('Model-card metadata must remain unchanged')
    core['output_files']['README.md']={'bytes':len(new_card),'sha256':hashlib.sha256(new_card).hexdigest()}
    core['distribution'].setdefault('documentation_updates',[]).append(dict(previous_revision=revision,
        previous_manifest_sha256=old_seal,changed_file='README.md',weight_files_unchanged=True))
    core['manifest_sha256']=identity(core)
    return core,core['manifest_sha256']+'\n'


def run(args):
    from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd
    api=HfApi();owner=api.whoami()['name']
    if args.repo_id.split('/')[0]!=owner:raise ValueError('Only the authenticated owner is permitted')
    current=api.model_info(args.repo_id,files_metadata=True)
    if current.sha!=args.expected_revision:raise ValueError('Remote revision changed; preserve and review concurrent edits')
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)
        def read(name):return Path(hf_hub_download(args.repo_id,name,revision=current.sha,cache_dir=work/'cache')).read_bytes()
        manifest=json.loads(read('manifest.json'));old_card=read('README.md');complete=read('COMPLETE').decode()
        new_card=args.card.read_bytes()
        updated,sealed=amend(manifest,complete,old_card,new_card,current.sha)
        (work/'README.md').write_bytes(new_card);write_json(work/'manifest.json',updated);(work/'COMPLETE').write_text(sealed)
        changed=('README.md','manifest.json','COMPLETE')
        commit=api.create_commit(args.repo_id,repo_type='model',parent_commit=current.sha,
            commit_message='Clarify experimental behavior in model card; preserve weight files',
            operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(work/name)) for name in changed])
        info=api.model_info(args.repo_id,revision=commit.oid,files_metadata=True)
        entries={x.rfilename:x for x in info.siblings}
        if set(entries)!=set(updated['output_files'])|{'manifest.json','COMPLETE'}:
            raise ValueError('Remote file set changed unexpectedly')
        for name,spec in updated['output_files'].items():
            item=entries[name];lfs=item.lfs
            if lfs:
                actual=lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256']
            else:
                actual=hashlib.sha256(Path(hf_hub_download(args.repo_id,name,revision=commit.oid,cache_dir=work/'verify')).read_bytes()).hexdigest()
            if actual!=spec['sha256'] or item.size!=spec['bytes']:raise ValueError('Remote file integrity differs: '+name)
        for name in ('manifest.json','COMPLETE'):
            remote=Path(hf_hub_download(args.repo_id,name,revision=commit.oid,cache_dir=work/'verify')).read_bytes()
            if remote!=(work/name).read_bytes():raise ValueError('Updated release seal differs')
        result=dict(repo_id=args.repo_id,previous_revision=current.sha,revision=commit.oid,
                    manifest_sha256=updated['manifest_sha256'],changed_files=list(changed),
                    all_file_hashes_verified=True,weight_files_unchanged=True,
                    weight_count=sum(n.endswith('.safetensors') for n in updated['output_files']))
        write_json(args.result,result);print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id',required=True);p.add_argument('--expected-revision',required=True)
    p.add_argument('--card',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()

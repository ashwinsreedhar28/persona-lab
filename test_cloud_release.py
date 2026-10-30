import hashlib
import io
import json
import random
import struct
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file, load
from huggingface_hub import CommitOperationAdd
from cloud.notice_stream import open_notice
from cloud.publish_hf import license_assets, sha_stream, publish
from abliteration.storage import identity
from cloud.runpod_job import bounded_seconds, owned, pod_spec, validate_job, bundle, BOOTSTRAP


class ReleaseTests(unittest.TestCase):
    def test_stream_preserves_tensor_bytes_and_random_access(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'weights.safetensors'
            tensors={'a':np.arange(16384,dtype=np.float32).reshape(128,128),'b':np.array([3,5],dtype=np.int64)}
            save_file(tensors,str(p),metadata={'format':'pt'})
            original=p.read_bytes();original_hash=hashlib.sha256(original).digest()
            with open_notice(p,'Modified for experimental release') as f:
                expected=f.read();spec=sha_stream(f)
                op=CommitOperationAdd(path_in_repo=p.name,path_or_fileobj=f)
                self.assertEqual(op.upload_info.sha256.hex(),spec['sha256'])
                self.assertEqual(op.upload_info.size,len(expected))
                rng=random.Random(3)
                for _ in range(100):
                    start=rng.randrange(len(expected)+20);count=rng.randrange(4000)
                    f.seek(start);self.assertEqual(f.read(count),expected[start:start+count])
                f.seek(-20,io.SEEK_END);self.assertEqual(f.read(),expected[-20:])
                f.seek(10);f.seek(8,io.SEEK_CUR);self.assertEqual(f.tell(),18)
                with self.assertRaises(ValueError):f.seek(-1)
                with self.assertRaises(ValueError):f.seek(0,10)
            old_n=struct.unpack('<Q',original[:8])[0];new_n=struct.unpack('<Q',expected[:8])[0]
            self.assertEqual(original[8+old_n:],expected[8+new_n:])
            header=json.loads(expected[8:8+new_n]);self.assertEqual(header['__metadata__']['format'],'pt')
            self.assertIn('Modified',header['__metadata__']['modification_notice'])
            for name,actual in load(expected).items():np.testing.assert_array_equal(actual,tensors[name])
            self.assertEqual(hashlib.sha256(p.read_bytes()).digest(),original_hash)
            with self.assertRaises(ValueError):f.read()

    def test_public_release_manifest_and_source_preservation(self):
        from huggingface_hub.errors import RepositoryNotFoundError
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'model.safetensors'
            save_file({'x':np.arange(8,dtype=np.float32)},str(p))
            (root/'LICENSE').write_text('Upstream license text')
            (root/'README.md').write_text('Private research checkpoint. Download this private repo while authenticated, then load it.')
            specs={name:{'bytes':(root/name).stat().st_size,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in ('model.safetensors','LICENSE','README.md')}
            core={'original_model':'upstream/model','files':specs}
            core['manifest_sha256']=identity(core)
            manifest=root/'legacy.json';manifest.write_text(json.dumps(core))
            original=p.read_bytes();uploaded={}
            class API:
                def __init__(self):self.committed=False
                def whoami(self):return {'name':'owner'}
                def model_info(self,repo,**kwargs):
                    if not self.committed:raise RepositoryNotFoundError('missing')
                    return SimpleNamespace(private=False,sha='new',siblings=[SimpleNamespace(rfilename=n,lfs=SimpleNamespace(sha256=hashlib.sha256(v).hexdigest())) for n,v in uploaded.items()])
                def create_repo(self,repo,**kwargs):
                    self_outer.assertEqual(kwargs,{'repo_type':'model','private':False})
                def create_commit(self,repo,**kwargs):
                    for op in kwargs['operations']:
                        value=op.path_or_fileobj
                        if isinstance(value,str):data=Path(value).read_bytes()
                        else:value.seek(0);data=value.read()
                        uploaded[op.path_in_repo]=data
                    self.committed=True
                    return SimpleNamespace(oid='new')
            self_outer=self
            args=SimpleNamespace(checkpoint=root,base_model='upstream/model',legacy_manifest=manifest,legacy_assets=None,
                repo_id='owner/public-model',license_id='apache-2.0',public=True,result=root/'result.json')
            with patch('huggingface_hub.HfApi',return_value=API()):publish(args)
            released=json.loads(uploaded['release_manifest.json']);recorded=released.pop('manifest_sha256')
            self.assertEqual(identity(released),recorded)
            self.assertEqual(uploaded['COMPLETE'].decode().strip(),recorded)
            self.assertNotIn('Private research',uploaded['README.md'].decode())
            self.assertIn('Download this public repo',uploaded['README.md'].decode())
            self.assertTrue(released['distribution']['tensor_payloads_unchanged'])
            self.assertEqual(released['distribution']['local_manifest_sha256'],core['manifest_sha256'])
            self.assertNotEqual(released['files']['model.safetensors']['sha256'],specs['model.safetensors']['sha256'])
            self.assertEqual(p.read_bytes(),original)
            self.assertEqual(json.loads((root/'result.json').read_text())['private'],False)

    def test_bad_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.safetensors'
            for data in (b'',struct.pack('<Q',999),struct.pack('<Q',8)+b'bad json'):
                p.write_bytes(data)
                with self.assertRaises((ValueError,TypeError)):
                    with open_notice(p,'notice'):pass

    def test_budget_and_job_validation(self):
        self.assertEqual(bounded_seconds(2,.5,.15),7200)
        self.assertEqual(bounded_seconds(10,.3,.15),7200)
        for bad in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):bounded_seconds(bad,1,1)
        with self.assertRaises(ValueError):bounded_seconds(1,.001,1)
        for bad in ({},[],{'releases':[{}]}, {'export':{'revision':'main'}}, {'export':{'revision':'g'*40}}):
            with self.assertRaises(ValueError):validate_job(bad)
        item=dict(checkpoint='/workspace/models/x',repo_id='owner/model',base_model='upstream/model',license_id='mit',public=True)
        self.assertEqual(validate_job({'releases':[item]})['releases'],[item])
        with self.assertRaises(ValueError):validate_job({'releases':[dict(item,public='true')]})

    def test_cpu_request_and_cleanup_ownership(self):
        args=SimpleNamespace(volume_id='volume',image='python:3.12-slim',gpu_type=None,cpu_flavor='cpu3c',vcpus=4)
        request=pod_spec(args,{'dataCenterId':'region'},'unique-worker','ssh-ed25519 test')
        self.assertNotIn('gpu',request);self.assertEqual(request['cpu']['vcpuCount'],4)
        state={'pod_id':'ours','name':'unique-worker','volume_id':'volume'}
        pod=dict(request,id='ours')
        self.assertTrue(owned(pod,state))
        for altered in (dict(pod,id='other'),dict(pod,name='other'),dict(pod,mounts={'network':[]})):
            self.assertFalse(owned(altered,state))
        self.assertIn("printf '%s\\n'",BOOTSTRAP)
        self.assertNotIn("printf '%s\\\\n'",BOOTSTRAP)

    def test_bundle_uses_committed_blobs_only(self):
        calls=[]
        def git(cmd,**kwargs):
            calls.append(cmd)
            if cmd==['git','ls-files']:return 'cloud/remote_job.py\nmodels/secret.safetensors\nabl/out/generations.json\nrequirements-cloud.txt\n'
            self.assertEqual(cmd[:2],['git','show']);return b'committed source'
        import tarfile
        with tempfile.TemporaryDirectory() as d,patch('cloud.runpod_job.subprocess.check_output',side_effect=git):
            p=Path(d)/'code.tar.gz';bundle(p)
            with tarfile.open(p) as archive:
                self.assertEqual(set(archive.getnames()),{'project/cloud/remote_job.py','project/requirements-cloud.txt'})
                self.assertEqual(archive.extractfile('project/cloud/remote_job.py').read(),b'committed source')

    def test_license_assets_match_provenance(self):
        root=Path(__file__).parent/'cloud/licenses'
        sources=json.loads((root/'sources.json').read_text())
        for name, record in sources['files'].items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),record['sha256'])
        for license_id,required in {'gemma':{'LICENSE','NOTICE','USE_POLICY.md'},'llama3.1':{'NOTICE'},'mit':{'LICENSE','THIRD_PARTY_LICENSES/Apache-2.0.txt'}}.items():
            mapping=license_assets(license_id);self.assertTrue(required<=set(mapping))
        self.assertIn('Section', (root/'Gemma-Terms.txt').read_text())
        self.assertIn('Built with Llama',(root/'Llama-NOTICE.txt').read_text())
        self.assertIn('2025 Moonshot AI',(root/'Kimi-Linear-MIT.txt').read_text())


if __name__=='__main__':unittest.main()

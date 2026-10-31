"""Model-name entry point for checked extraction, safe export, and fresh-process reload."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from .architecture import adapter_for, writers
from .storage import (destination, digest, file_manifest, identity, inventory, publish,
                      verify_files, write_json)

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ['*.safetensors', '*.json', '*.py', '*.txt', '*.jinja', '*.model', 'LICENSE*','NOTICE*','USE_POLICY*']


def resolve_config(model, revision=None, offline=False):
    local = Path(model).expanduser()
    if local.is_dir():
        root = local.resolve()
        return root, json.loads((root / 'config.json').read_text()), dict(kind='local', revision=revision)
    from huggingface_hub import hf_hub_download
    cfg = Path(hf_hub_download(model, 'config.json', revision=revision, local_files_only=offline))
    # Standard Hub cache path: .../snapshots/<immutable commit>/config.json.
    resolved_revision = cfg.parent.name
    if len(resolved_revision) != 40 or any(c not in '0123456789abcdef' for c in resolved_revision):
        raise ValueError('Could not resolve an immutable source revision')
    return cfg.parent, json.loads(cfg.read_text()), dict(kind='hub', repo_id=model, revision=resolved_revision)


def resolve_source(model, revision=None, offline=False):
    root, cfg, provenance = resolve_config(model, revision, offline)
    adapter_for(cfg)  # Reject unsupported and quantized models before downloading weights.
    if provenance['kind'] == 'hub':
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(model, revision=provenance['revision'],
                    allow_patterns=PATTERNS, local_files_only=offline)).resolve()
    return root, cfg, provenance


def source_files(root):
    import fnmatch
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file() and
        '.cache' not in p.relative_to(root).parts and '__pycache__' not in p.relative_to(root).parts and
        any(fnmatch.fnmatch(str(p.relative_to(root)), pattern) for pattern in PATTERNS))


def hashes(root, files):
    return {name: dict(bytes=(Path(root) / name).stat().st_size, sha256=digest(Path(root) / name)) for name in files}


def environment():
    versions = {}
    for name in ('numpy', 'mlx', 'mlx-lm', 'torch', 'transformers', 'fla-core', 'safetensors', 'huggingface-hub'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions)


def code_state():
    files = list((ROOT / 'abliteration').glob('*.py')) + [ROOT / 'abliterate.py', ROOT / 'research_checks.py']
    files += [ROOT / 'abl' / f for f in ('common.py', 'a2_directions.py', 'a4_edit.py', 'a5_projection.py')]
    files = [p for p in files if p.is_file()]
    deployment=ROOT/'DEPLOYMENT.json'
    if deployment.is_file():files.append(deployment)
    result = dict(sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(files)})
    if deployment.is_file():result['deployment']=json.loads(deployment.read_text())
    try:
        result['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                     stderr=subprocess.DEVNULL, text=True).strip()
        result['git_status'] = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT,
                                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result['git_head'] = result['git_status'] = None
    return result


def run_worker(job, stage):
    log = job.parent / f'{stage}.log'
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    # Workers load local files only. Do not expose account credentials to model implementations.
    for name in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','RUNPOD_API_KEY'):
        env.pop(name,None)
    cmd = [sys.executable, '-m', 'abliteration.worker', str(job), stage]
    with log.open('w') as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                # The workers perform logits-only checks and never decode generations or print input data.
                print(line, end='', flush=True)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if code:
        raise RuntimeError(f'{stage} failed (exit {code}); details: {log}')


def verify(root):
    root = Path(root).resolve()
    m = json.loads((root / 'manifest.json').read_text())
    recorded = m.pop('manifest_sha256')
    if identity(m) != recorded or (root / 'COMPLETE').read_text().strip() != recorded:
        raise ValueError('Manifest identity or completion marker is invalid')
    verify_files(root, m['output_files'])
    inv = inventory(root)
    cfg = json.loads((root / 'config.json').read_text())
    writers(cfg, inv)
    checks = json.loads((root / '.pipeline/tensor_checks.json').read_text())
    if set(checks) != set(inv):
        raise ValueError('Tensor verification coverage is incomplete')
    reload = json.loads((root / '.pipeline/reload.json').read_text())
    if not reload['passed']:
        raise ValueError('Checkpoint has no passing fresh-process reload')
    return dict(checkpoint=str(root), manifest_sha256=recorded, tensors=len(inv),
                edited_tensors=sum(c['edited'] for c in checks.values()),
                serialization_verified=True, reload=reload,
                behavioral_acceptance='Not established by integrity/reload verification')


def run(args):
    requested = args.output or ROOT / 'models' / (
        Path(args.model).name + '-abliterated-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    # Reject existing outputs before Hub access.
    if Path(requested).exists() or Path(requested).is_symlink():
        raise FileExistsError(f'Output already exists: {requested}; choose a new name')
    conversion=None
    if getattr(args,'prepared_source',None):
        from .conversion import verify_prepared
        source=Path(args.prepared_source).expanduser().resolve()
        conversion=verify_prepared(source,args.model,args.revision)
        _,original_config,origin=resolve_config(args.model,args.revision,args.offline)
        if original_config!=json.loads((source/'.conversion/job.json').read_text())['raw_config']:
            raise ValueError('Prepared source original configuration differs')
        cfg=json.loads((source/'config.json').read_text())
        provenance=dict(kind='prepared',repo_id=args.model,revision=conversion['origin'].get('revision'),
                        conversion_manifest_sha256=conversion['manifest_sha256'])
    else:
        source, cfg, provenance = resolve_source(args.model, args.revision, args.offline)
    adapter = adapter_for(cfg)
    output = destination(source, requested)
    if adapter == 'kimi-torch' and not args.trust_remote_code:
        raise ValueError('Kimi requires --trust-remote-code to load the pinned model implementation')
    inv = inventory(source)
    modes = writers(cfg, inv)
    files = source_files(source)
    if conversion:
        files=sorted(set(files)|{n for n in conversion['files'] if n.startswith('.conversion/')}|
                     {'.conversion/manifest.json','.conversion/COMPLETE'})
    if any(v['shard'] not in files for v in inv.values()):
        raise ValueError('Source inventory contains a shard outside the allowed file set')
    data = Path(args.data).expanduser().resolve()
    names = ('harmful_train', 'harmless_train', 'harmful_val', 'harmless_val', 'harmless_kl')
    split_hashes = {name: digest(data / f'{name}.json') for name in names}
    direction = str(Path(args.direction).expanduser().resolve()) if args.direction else None
    imported_direction_hash = digest(direction) if direction else None
    output.parent.mkdir(parents=True, exist_ok=True)
    weight_bytes = sum((source / shard).stat().st_size for shard in {s['shard'] for s in inv.values()})
    if shutil.disk_usage(output.parent).free < weight_bytes + 1024 ** 3:
        raise RuntimeError('Insufficient disk space for an independent output checkpoint plus 1 GiB margin')
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.incomplete-', dir=output.parent))
    meta = staging / '.pipeline'
    meta.mkdir()
    job = dict(schema_version=1, model=args.model, source=str(source), config=cfg, adapter=adapter,
        source_files=files, data=str(data), direction=direction, batch_size=args.batch_size,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        intervention_policy='Gemma: project each normalized writer along unit((1+norm_weight)*r); other writers along r'
            if cfg['model_type']=='gemma3' else 'Project residual writers along r',
        prompt_policy='Native Harmony template; reasoning_effort=low; explicit ordinary analysis header; eager attention with sinks and eager experts'
            if cfg['model_type']=='gpt_oss' else 'Native template; enable_thinking=False where supported; record actual suffix IDs',
        reload_atol=args.reload_atol if args.reload_atol is not None else (1e-5 if adapter == 'qwen-mlx' else .02))
    write_json(meta / 'job.json', job)
    start = time.monotonic()
    try:
        print(f'Hashing read-only original: {source}', flush=True)
        source_hashes = hashes(source, files)
        state = code_state()
        write_json(meta / 'source.json', dict(origin=provenance, files=source_hashes))
        write_json(meta / 'code.json', state)
        write_json(meta / 'environment.json', environment())
        write_json(meta / 'data.json', split_hashes)
        for stage in ('reference', 'export', 'reload'):
            print(f'Starting {stage}; output remains unpublished: {staging}', flush=True)
            run_worker(meta / 'job.json', stage)
        if hashes(source, files) != source_hashes:
            raise ValueError('Original source changed during the run; refusing publication')
        if conversion:
            from .conversion import verify_prepared
            if verify_prepared(source,args.model,args.revision)['manifest_sha256']!=conversion['manifest_sha256']:
                raise ValueError('Prepared/raw source changed during export')
        if code_state()['sha256'] != state['sha256']:
            raise ValueError('Pipeline code changed during the run; refusing publication')
        if {name: digest(data / f'{name}.json') for name in names} != split_hashes:
            raise ValueError('Input data changed during the run; refusing publication')
        if direction and digest(direction) != imported_direction_hash:
            raise ValueError('Imported direction changed during the run; refusing publication')
        write_json(meta / 'status.json', dict(state='verified', source_unchanged=True,
            duration_seconds=round(time.monotonic() - start, 2),
            scope='All architecture-mapped decoder writers and supported vision projector. MTP tensors, if present, preserved unedited. Text-only reload verification.'))
        manifest = dict(schema_version=1, adapter=adapter, model=args.model, source=provenance,
            source_manifest_sha256=digest(meta / 'source.json'),
            direction_file_sha256=digest(meta / 'direction.npy'),
            precision='bf16 inputs/outputs, fp32 projection',
            intervention_policy=job['intervention_policy'],
            tied_embedding_policy='edit input embedding; tied read head changes with it; untied read head untouched',
            original_weights_unchanged=True, output_files=file_manifest(staging))
        manifest['manifest_sha256'] = identity(manifest)
        write_json(staging / 'manifest.json', manifest)
        with (staging / 'COMPLETE').open('w') as f:
            f.write(manifest['manifest_sha256'] + '\n')
            f.flush()
            os.fsync(f.fileno())
        verify(staging)
        publish(staging, output)
        print(json.dumps(dict(checkpoint=str(output), manifest_sha256=manifest['manifest_sha256'],
            original_unchanged=True, edited_tensors=len(modes), reload_verified=True,
            behavioral_acceptance='Not evaluated'), indent=2), flush=True)
        return output
    except BaseException as error:
        if meta.exists():
            write_json(meta / 'status.json', dict(state='failed', error_type=type(error).__name__,
                duration_seconds=round(time.monotonic() - start, 2)))
        print(f'Run failed; original preserved. Incomplete diagnostics retained at {staging}', file=sys.stderr, flush=True)
        raise


def parser():
    p = argparse.ArgumentParser(description='Extract a direction, save a separate edited checkpoint, verify reload. Originals are never overwritten.')
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', required=True, help='Hub model ID or original local bf16 checkpoint')
    r.add_argument('--revision', help='Source Hub revision; resolved to immutable commit before any load')
    r.add_argument('--prepared-source',type=Path,help='Verified separate GPT-OSS bf16 preparation; original --model identity is retained')
    r.add_argument('--output', type=Path, help='New directory; default models/<model>-abliterated-<UTC time>')
    r.add_argument('--data', type=Path, default=ROOT / 'abl/data', help='Existing fixed train/validation/KL splits')
    r.add_argument('--direction', type=Path, help='Explicitly import an existing unit .npy direction instead of extracting/selecting')
    r.add_argument('--batch-size', type=int, default=2)
    r.add_argument('--device',choices=['cuda:0','cpu'],default='cuda:0',
                   help='Dense PyTorch adapter device; CPU is for small fixtures, CUDA for full models')
    r.add_argument('--reload-atol', type=float, help='Absolute logprob tolerance; default 1e-5 MLX / .02 CUDA; greedy IDs must match')
    r.add_argument('--offline', action='store_true')
    r.add_argument('--trust-remote-code', action='store_true', help='Opt into code from the pinned Kimi snapshot')
    v = sub.add_parser('verify', help='Hash/manifest verification without loading a model')
    v.add_argument('checkpoint', type=Path)
    i = sub.add_parser('plan', help='Inspect config/support only; does not download weights or rent a GPU')
    i.add_argument('--model', required=True)
    i.add_argument('--revision')
    i.add_argument('--offline', action='store_true')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.command == 'run':
        if args.batch_size <= 0 or (args.reload_atol is not None and not 0 <= args.reload_atol < float('inf')):
            p.error('batch-size must be positive and reload-atol finite and nonnegative')
        run(args)
    elif args.command == 'verify':
        print(json.dumps(verify(args.checkpoint), indent=2))
    else:
        root, cfg, provenance = resolve_config(args.model, args.revision, args.offline)
        print(json.dumps(dict(adapter=adapter_for(cfg), source=provenance, config_directory=str(root),
            policy='Original bf16 weights; separate output; exact tensor readback and fresh-process reload required.'), indent=2))


if __name__ == '__main__':
    main()

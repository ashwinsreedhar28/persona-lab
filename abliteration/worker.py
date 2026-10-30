"""One model per process. Export and reload run after the reference process exits."""
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys

import numpy as np

from research_checks import split_overlap, unit_direction
from .architecture import writers
from .storage import digest, header, inventory, write_json

PROBES = ['Reply with one word: hello.', 'What is 17 plus 25? Answer briefly.',
          'Write a short Python function that adds two numbers.']
SPLITS = ('harmful_train', 'harmless_train', 'harmful_val', 'harmless_val', 'harmless_kl')


def backend(job):
    return importlib.import_module('.qwen' if job['adapter'] == 'qwen-mlx' else '.kimi', __package__)


def make_engine(job, source=None):
    return backend(job).Engine(source or job['source'], job['config'], job['batch_size'], job['trust_remote_code'])


def logodds(lp, tokens):
    p = np.exp(np.asarray(lp, np.float64)[:, tokens]).sum(-1).clip(1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def kl(p, q):
    return (np.exp(np.asarray(p, np.float64)) * (p - q)).sum(-1)


def auroc(pos, neg):
    return float(((pos[:, None] > neg[None]).sum() +
                  .5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def select(engine, data, meta):
    """Fixed separation/induction/KL rule, matching the previous Kimi experiment, not legacy Qwen bypass ranking."""
    suffix = engine.suffix()
    ha, hf = engine.cache(data['harmful_train'], suffix)
    ba, bf = engine.cache(data['harmless_train'], suffix)
    directions = (ha.mean(0) - ba.mean(0)).astype(np.float32)
    del ha, ba
    hv, _ = engine.cache(data['harmful_val'], suffix)
    bv, _ = engine.cache(data['harmless_val'], suffix)
    hc, bc = Counter(hf), Counter(bf)
    if engine.__class__.__module__.endswith('.qwen'):
        tokens = [engine.tok.encode(t, add_special_tokens=False) for t in ('I', 'As')]
        if any(len(t) != 1 for t in tokens):
            raise ValueError('Qwen refusal-token assumptions do not hold for this tokenizer')
        tokens = [t[0] for t in tokens]
    else:
        tokens = sorted(t for t, count in hc.items() if count >= 8 and count > bc[t])
        if not tokens:
            raise ValueError('Training logits provide no stable refusal-token proxy; stop for review')
    candidates = []
    for layer in range(int(engine.n_layers * .8)):
        for pos in range(len(suffix)):
            r = directions[layer, pos]
            norm = float(np.linalg.norm(r))
            if not np.isfinite(norm) or norm < 1e-4:
                continue
            rh = unit_direction(r)
            candidates.append(dict(layer=layer, pos=pos - len(suffix), norm=norm,
                auroc=auroc(hv[:, layer, pos] @ rh, bv[:, layer, pos] @ rh)))
    del hv, bv
    top = sorted(candidates, key=lambda c: (-c['auroc'], c['layer'], c['pos']))[:20]
    ids = [engine.encode(p) for p in data['harmless_val']]
    base = engine.lp(ids)
    base_lo = logodds(base, tokens)
    for i, c in enumerate(top):
        r = directions[c['layer'], c['pos']]
        rh = unit_direction(r)
        with engine.added(c['layer'], r):
            added = engine.lp(ids)
        with engine.ablated(rh):
            ablated = engine.lp(ids)
        c.update(induce=float((logodds(added, tokens) - base_lo).mean()),
                 kl=float(kl(base, ablated).mean()))
        c['passes'] = c['auroc'] >= .9 and c['induce'] > 0 and c['kl'] <= .1
        print(f"candidate {i+1}/{len(top)} layer={c['layer']} pos={c['pos']} "
              f"AUROC={c['auroc']:.3f} induce={c['induce']:.3f} KL={c['kl']:.4f} passes={c['passes']}", flush=True)
    passing = sorted((c for c in top if c['passes']), key=lambda c: (-c['induce'], c['layer'], c['pos']))
    report = dict(method='separation-induction-kl-v1',
        rule='First 80% of layers, top 20 by validation AUROC; AUROC >= .9, induce > 0, KL <= .1; highest induce wins.',
        note='This differs from legacy Qwen bypass ranking. Tied embedding read policy matches the weight edit.',
        suffix_token_ids=suffix, refusal_token_ids=tokens, all_candidates=candidates,
        screened=top, selected=passing[0] if passing else None)
    write_json(meta / 'selection.json', report)
    if not passing:
        raise ValueError('No candidate passes the fixed selection rule; no checkpoint will be published')
    c = passing[0]
    return unit_direction(directions[c['layer'], c['pos']])


def reference(job, root):
    meta = root / '.pipeline'
    data = {k: json.loads((Path(job['data']) / f'{k}.json').read_text()) for k in SPLITS}
    if any(not isinstance(rows, list) or not rows or any(not isinstance(p, str) or not p.strip() for p in rows)
           for rows in data.values()):
        raise ValueError('Each input split must be a nonempty list of prompt strings')
    overlaps = split_overlap(data)
    if any(overlaps['duplicates'].values()) or any(p['exact'] or p['shared_8gram'] for p in overlaps['pairs']):
        raise ValueError('Input train/validation/KL splits overlap; preserve fixed data and review before proceeding')
    engine = make_engine(job)
    if job['direction']:
        rh = unit_direction(np.load(job['direction'], allow_pickle=False), require_unit=True)
        write_json(meta / 'selection.json', dict(method='explicit-import',
            input_file_sha256=digest(job['direction']),
            limitation='Caller supplied this direction; source-model lineage is not independently established.'))
    else:
        rh = select(engine, data, meta)
    if rh.shape != (engine.hidden,):
        raise ValueError('Direction dimension differs from the selected model')
    np.save(meta / 'direction.npy', rh)
    val = data['harmful_val'] + data['harmless_val']
    base_projection = engine.projection(val, rh)
    kl_ids = [engine.encode(p) for p in data['harmless_kl']]
    base_lp = engine.lp(kl_ids)
    probes = [engine.encode(p) for p in PROBES]
    with engine.ablated(rh):
        hook_lp = engine.lp(probes)
    engine.edit(rh)
    edited_projection = engine.projection(val, rh)
    edited_lp = engine.lp(kl_ids)
    probe = engine.probe(probes)
    np.savez(meta / 'reference.npz', logprobs=probe)
    write_json(meta / 'probe.json', dict(prompts=PROBES, input_token_ids=probes,
        greedy_steps=3, logprob_atol=job['reload_atol'], rtol=0,
        expected_greedy_token_ids=probe.argmax(-1).tolist(), cache=False))
    ratios = np.divide(edited_projection, base_projection, out=np.full_like(edited_projection, np.nan),
                       where=base_projection > 1e-12)
    write_json(meta / 'measurements.json', dict(
        base_projection=base_projection.tolist(), edited_projection=edited_projection.tolist(),
        per_layer_ratio=[float(v) if np.isfinite(v) else None for v in ratios],
        mechanical_threshold=.01, mechanical_pass=bool(np.isfinite(ratios).all() and (ratios <= .01).all()),
        harmless_kl_mean=float(kl(base_lp, edited_lp).mean()),
        hook_vs_weight_edit_kl_mean=float(kl(hook_lp, probe[0]).mean()),
        behavioral_acceptance='Not evaluated by this export command; no universal-refusal or capability-preservation claim.'))
    print('In-memory edited reference saved; exiting before shard export.', flush=True)


def payload_hash(path, spec):
    with Path(path).open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        start, end = spec['data_offsets']
        f.seek(8 + length + start)
        left, h = end - start, hashlib.sha256()
        while left:
            block = f.read(min(left, 8 * 1024 * 1024))
            if not block:
                raise ValueError('Truncated tensor payload')
            h.update(block)
            left -= len(block)
    return h.hexdigest()


def export(job, root):
    source = Path(job['source'])
    inv = inventory(source)
    modes = writers(job['config'], inv)
    rh = unit_direction(np.load(root / '.pipeline/direction.npy', allow_pickle=False), require_unit=True)
    lib = backend(job)
    checks = {}
    for number, shard in enumerate(sorted({v['shard'] for v in inv.values()})):
        print(f'Exporting shard {number+1}: {shard}', flush=True)
        expected = lib.export_shard(source / shard, root / shard, modes, rh)
        with (root / shard).open('rb') as f:
            os.fsync(f.fileno())
        saved = header(root / shard)
        original = header(source / shard)
        if set(saved) != set(original) or set(expected) != set(original):
            raise ValueError('Tensor keys changed during export')
        for key, spec in saved.items():
            before = original[key]
            if spec['shape'] != before['shape'] or spec['dtype'] != before['dtype']:
                raise ValueError(f'Tensor shape/dtype changed: {key}')
            actual = payload_hash(root / shard, spec)
            if actual != expected[key]:
                raise ValueError(f'Save/readback differs from pre-save tensor: {key}')
            if key not in modes and actual != payload_hash(source / shard, before):
                raise ValueError(f'Unedited tensor changed: {key}')
            checks[key] = dict(dtype=spec['dtype'], shape=spec['shape'], sha256=actual,
                               edited=key in modes)
    for name in job['source_files']:
        if not name.endswith('.safetensors'):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
            with target.open('rb') as f:
                os.fsync(f.fileno())
    write_json(root / '.pipeline/tensor_checks.json', checks)
    print(f'All {len(checks)} tensors verified; {len(modes)} edited. Source format and other tensors preserved.', flush=True)


def reload(job, root):
    probe = json.loads((root / '.pipeline/probe.json').read_text())
    expected = np.load(root / '.pipeline/reference.npz', allow_pickle=False)['logprobs']
    engine = make_engine(job, str(root))
    actual = engine.probe(probe['input_token_ids'])
    error = float(np.max(np.abs(expected - actual)))
    scores_match = np.isfinite(actual).all() and np.allclose(expected, actual, atol=job['reload_atol'], rtol=0)
    tokens_match = np.array_equal(expected.argmax(-1), actual.argmax(-1))
    result = dict(passed=bool(scores_match and tokens_match),
        max_abs_logprob_difference=error, atol=job['reload_atol'], rtol=0,
        greedy_token_ids_match=bool(tokens_match), model=str(root),
        scope='Three harmless prompts, three uncached greedy steps; fresh process vs in-memory edit before serialization.')
    write_json(root / '.pipeline/reload.json', result)
    if not result['passed']:
        raise ValueError(f'Fresh-process reload did not match the in-memory reference: {result}')
    print(f'Fresh-process reload passed: max logprob difference {error:.6g}, greedy token IDs identical.', flush=True)


def main():
    job_file, stage = Path(sys.argv[1]), sys.argv[2]
    job = json.loads(job_file.read_text())
    root = job_file.parent.parent
    dict(reference=reference, export=export, reload=reload)[stage](job, root)


if __name__ == '__main__':
    main()

"""Streaming Qwen calibration diagnostics; no weight edits or decoded responses."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def units(v):
    n=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.divide(v,n,out=np.zeros_like(v),where=n>1e-8),n[...,0]


def fold_ids(base_n,seed):
    if base_n%4:raise ValueError('Baseline size must divide into four equal folds')
    result=[]
    for stage in range(2):
        order=list(range(base_n));random.Random(seed+stage).shuffle(order)
        labels=np.empty(base_n,dtype=int)
        for j,i in enumerate(order):labels[i]=j%4
        result.extend(labels.tolist())
    return result


def summarize_moments(h,b,counts):
    if np.any(counts<=0):raise ValueError('Every stage/fold must contain examples')
    result={}
    for name,stages in [('baseline',[0]),('expanded',[0,1])]:
        total=counts[stages].sum()
        mean=(h[stages].sum((0,1))-b[stages].sum((0,1)))/total
        unit,norm=units(mean)
        fold=(h[stages].sum(0)-b[stages].sum(0))/counts[stages].sum(0)[:,None,None,None]
        fold_unit,fold_norm=units(fold)
        similarities=[]
        for i in range(4):
            for j in range(i+1,4):
                good=(fold_norm[i]>1e-8)&(fold_norm[j]>1e-8)
                similarities.append(np.where(good,(fold_unit[i]*fold_unit[j]).sum(-1),np.nan))
        sim=np.array(similarities);valid=np.isfinite(sim).sum(0)
        stability=np.divide(np.nansum(sim,axis=0),valid,out=np.full_like(norm,np.nan),where=valid>0)
        result[name]=dict(direction=mean.astype(np.float32),unit=unit.astype(np.float32),
                          norm=norm,fold_pair_cosine_mean=stability)
    return result


def finite(value):
    if isinstance(value,np.ndarray):return finite(value.tolist())
    if isinstance(value,list):return [finite(x) for x in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def run(args):
    root=args.source_code.resolve()
    sys.path[:0]=[str(root/'abl'),str(root)]
    import mlx.core as mx
    from common import load,eoi_ids,layers,peak_ram_gb
    from a2_directions import cache
    mx.random.seed(0);mx.set_cache_limit(256*1024**2)
    data={n:json.loads((args.data/(n+'.json')).read_text()) for n in
          ('harmful_train','harmless_train','harmful_val','harmless_val')}
    manifest=json.loads((args.data/'manifest.json').read_text())
    n=manifest['baseline_sizes']['harmful_train']
    if len(data['harmful_train'])!=2*n or len(data['harmless_train'])!=2*n:
        raise ValueError('This diagnostic requires an exactly doubled, balanced nested set')
    for name,expected in manifest['output_files'].items():
        if digest(args.data/name)!=expected:raise ValueError('Frozen dataset hash mismatch')
    if args.output.exists():raise FileExistsError('Diagnostic destination exists')
    index=json.loads((args.model/'model.safetensors.index.json').read_text())
    members=sorted(set(index['weight_map'].values())|{'config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'})
    hashes={name:digest(args.model/name) for name in members}
    args.output.mkdir(parents=True,exist_ok=False)
    config=dict(model_revision=args.model.name,model_file_hashes=hashes,data_manifest_sha256=digest(args.data/'manifest.json'),
        harness_sha256=digest(__file__),helper_source_sha256={name:digest(root/name) for name in
            ('abl/common.py','abl/a2_directions.py','evil_chat_mlx.py','research_checks.py')},
        settings=dict(seed=20261031,batch_size=1,thinking=False,activation_site='resid_pre',
            positions='all post-instruction template suffix tokens',precision='bf16 model, fp32 cached slices, fp64 accumulated sums',
            primary_layer_rule='zero-based layer floor(n_layers/2)',primary_position=-1,folds=4),
        scope='Direction estimation, four-fold consistency and fixed validation separation only; no weight edit or refusal/capability test.')
    save(args.output/'config.json',config)
    start=time.time();model,tok=load(str(args.model));suffix=eoi_ids(tok)
    shape=(len(layers(model)),len(suffix),json.loads((args.model/'config.json').read_text())['text_config']['hidden_size'])
    sums={label:np.zeros((2,4,*shape),dtype=np.float64) for label in ('harmful','harmless')}
    count=np.zeros((2,4),dtype=int)
    assignments=fold_ids(n,20261031)
    for label in ('harmful','harmless'):
        for i,prompt in enumerate(data[label+'_train']):
            acts,_=cache(model,tok,[prompt],suffix,bs=1)
            stage=i//n;fold=assignments[i]
            sums[label][stage,fold]+=acts[0]
            if label=='harmful':count[stage,fold]+=1
            del acts;mx.clear_cache()
            if (i+1)%16==0:print(label,'calibrated',i+1,'/',2*n,flush=True)
    stats=summarize_moments(sums['harmful'],sums['harmless'],count)
    del sums
    scores={name:{label:[] for label in ('harmful','harmless')} for name in stats}
    for label in ('harmful','harmless'):
        for prompt in data[label+'_val']:
            acts,_=cache(model,tok,[prompt],suffix,bs=1)
            for name,values in stats.items():scores[name][label].append(np.einsum('lpd,lpd->lp',acts[0],values['unit']))
            del acts;mx.clear_cache()
        print(label,'validation activations complete.',flush=True)
    primary=(shape[0]//2,-1)
    report=dict(scope=config['scope'],baseline_per_class=n,expanded_per_class=2*n,
        primary_location=dict(layer=primary[0],position=primary[1]),settings=config['settings'],
        variants={},limitations=['Fold consistency and validation AUROC are diagnostics, not proof of reduced refusals or preserved capabilities.',
            'The validation prompts are historical selection data, not new confirmation data.',
            'The expansion changes size and source composition together.',
            'Data overlap screening is lexical; semantic/pretraining contamination remains unverified.'])
    for name,values in stats.items():
        h=np.array(scores[name]['harmful']);b=np.array(scores[name]['harmless'])
        auroc=((h[:,None]>b[None]).sum((0,1))+.5*(h[:,None]==b[None]).sum((0,1)))/(len(h)*len(b))
        report['variants'][name]=dict(direction_norms=finite(values['norm']),
            fold_pair_cosine_mean=finite(values['fold_pair_cosine_mean']),validation_auroc=finite(auroc),
            primary=dict(direction_norm=float(values['norm'][primary]),
                fold_pair_cosine_mean=finite(float(values['fold_pair_cosine_mean'][primary])),
                validation_auroc=float(auroc[primary])))
        np.savez(args.output/(name+'-directions.npz'),direction=values['direction'],suffix_token_ids=np.array(suffix))
    cosine=(stats['baseline']['unit']*stats['expanded']['unit']).sum(-1)
    report.update(baseline_expanded_direction_cosine=finite(cosine),
        primary_direction_cosine=float(cosine[primary]),seconds=time.time()-start,peak_ram_gb=peak_ram_gb())
    del model;mx.clear_cache()
    assert all(digest(args.model/name)==expected for name,expected in hashes.items()),'Original source file changed'
    report['original_model_files_unchanged']=True
    save(args.output/'report.json',report)
    print(json.dumps({'primary_location':report['primary_location'],
        'baseline':report['variants']['baseline']['primary'],'expanded':report['variants']['expanded']['primary'],
        'direction_cosine':report['primary_direction_cosine'],'seconds':report['seconds']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-code','data','model','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()

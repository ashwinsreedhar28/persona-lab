"""Deterministic nested calibration expansion; preserve fixed splits and print metadata only."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re

SPLITS = ('harmful_train', 'harmless_train', 'harmful_val', 'harmless_val', 'harmless_kl')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return ' '.join(re.sub('[^a-z0-9 ]', ' ', text.lower()).split())


def grams(text, n):
    words = normalized(text).split()
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}


class Exclusions:
    """Exact/8-gram plus a declared lexical-near-duplicate heuristic, not semantic proof."""
    def __init__(self, texts=(), near_threshold=.5):
        self.norms = set()
        self.eight = set()
        self.three = []
        self.lookup = defaultdict(set)
        self.threshold = near_threshold
        for text in texts:
            self.add(text)

    def reason(self, text):
        if normalized(text) in self.norms:
            return 'exact'
        if grams(text, 8) & self.eight:
            return 'shared_8gram'
        g = grams(text, 3)
        candidates = set().union(*(self.lookup[t] for t in g)) if g else set()
        for i in candidates:
            other = self.three[i]
            if len(g & other) / len(g | other) >= self.threshold:
                return 'near_3gram_jaccard'
        return None

    def add(self, text):
        self.norms.add(normalized(text))
        self.eight.update(grams(text, 8))
        g = grams(text, 3)
        i = len(self.three)
        self.three.append(g)
        for t in g:
            self.lookup[t].add(i)


def choose(candidates, count, exclusions, seed):
    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)
    selected, indices, rejected = [], [], Counter()
    for i in order:
        text = candidates[i].strip()
        if not text or not 4 <= len(text.split()) <= 256:
            rejected['blank_or_word_length'] += 1
            continue
        reason = exclusions.reason(text)
        if reason:
            rejected[reason] += 1
            continue
        selected.append(text)
        indices.append(i)
        exclusions.add(text)
        if len(selected) == count:
            return selected, dict(selected_source_indices=indices, rejected=dict(rejected))
    raise ValueError(f'Insufficient nonoverlapping candidates: {len(selected)}/{count}')


def prompt_strings(values):
    if not isinstance(values,list):
        return
    for value in values:
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key in ('goal', 'question', 'prompt'):
                if key in value:
                    yield value[key]


def prepare(base_dir, sources, output, extra_prompt_files=(), seed=20261031, per_class=256):
    base_dir, sources, output = map(Path, (base_dir, sources, output))
    if output.exists():
        raise FileExistsError('Destination already exists; preserve prior versions')
    base = {n: json.loads((base_dir/(n+'.json')).read_text()) for n in SPLITS}
    count = len(base['harmful_train'])
    if count != len(base['harmless_train']) or per_class <= count:
        raise ValueError('Expected balanced, strictly larger nested training sets')
    fixed = sorted(base_dir.glob('*.json'))
    forbidden = [s for p in fixed for s in prompt_strings(json.loads(p.read_text()))]
    reserved = json.loads((sources/'reserved-sources.json').read_text())
    for record in reserved:
        p = sources/record['texts_file']
        if digest(p) != record['texts_sha256']:
            raise ValueError('Reserved source hash mismatch')
        forbidden += json.loads(p.read_text())
    for path in extra_prompt_files:
        forbidden += list(prompt_strings(json.loads(Path(path).read_text())))
    exclusions = Exclusions(forbidden)
    harmful = [r['goal'] for r in csv.DictReader(io.StringIO((sources/'advbench.csv').read_text()))]
    harmless = json.loads((sources/'alpaca-empty-input.json').read_text())
    h, h_log = choose(harmful, per_class-count, exclusions, seed)
    b, b_log = choose(harmless, per_class-count, exclusions, seed+1)
    source_meta = {name: json.loads((sources/(name+'-source.json')).read_text()) for name in ('advbench','alpaca')}
    if digest(sources/'advbench.csv') != source_meta['advbench']['sha256']:
        raise ValueError('Harmful source hash mismatch')
    if digest(sources/'alpaca-empty-input.json') != source_meta['alpaca']['texts_sha256']:
        raise ValueError('Benign source hash mismatch')
    output.mkdir(parents=True, exist_ok=False)
    for name in SPLITS:
        if name.endswith('_train'):
            rows = base[name] + (h if name=='harmful_train' else b)
            (output/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
        else:
            (output/(name+'.json')).write_bytes((base_dir/(name+'.json')).read_bytes())
    manifest = dict(schema=1,seed=seed,per_class=per_class,
        baseline_sizes={n:len(v) for n,v in base.items()},
        parent_files={p.name:digest(p) for p in fixed},
        extra_prompt_files={str(p):digest(p) for p in extra_prompt_files},
        sources=source_meta,reserved_evaluation_sources=reserved,
        additions={'harmful':h_log,'harmless':b_log},
        output_files={n+'.json':digest(output/(n+'.json')) for n in SPLITS},
        overlap_rule='Normalized exact, any shared word 8-gram, or word-3gram Jaccard >= 0.5 against existing splits, reserved benchmarks, prior pilot inputs and earlier accepted additions.',
        limitations=['Lexical screening does not establish semantic or pretraining independence.',
            'Size and source composition both change; this is an expansion experiment, not an isolated sample-size experiment.',
            'Existing baseline overlap with newly reserved benchmark sources is not retroactively removed.',
            'Source labels are inherited; new benign examples require a separate quality screen before calibration.',
            'No performance or universal-refusal-removal claim follows from dataset preparation.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-data',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exclude-prompts',type=Path,nargs='*',default=[])
    p.add_argument('--seed',type=int,default=20261031)
    p.add_argument('--per-class',type=int,default=256)
    a=p.parse_args()
    m=prepare(a.base_data,a.sources,a.output,a.exclude_prompts,a.seed,a.per_class)
    print(json.dumps({'per_class':m['per_class'],'added_per_class':m['per_class']-m['baseline_sizes']['harmful_train'],'manifest_sha256':digest(a.output/'manifest.json')}))


if __name__=='__main__':main()

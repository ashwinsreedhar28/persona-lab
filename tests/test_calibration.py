"""Model-free invariants for protected calibration data and streamed diagnostics."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'evaluation'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
data=module('calibration_data');probe=module('calibration_probe')


class DataTests(unittest.TestCase):
    def test_exact_eightgram_and_near_duplicates_are_excluded(self):
        text='alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu'
        ex=data.Exclusions([text])
        self.assertEqual(ex.reason(text.upper()),'exact')
        self.assertEqual(ex.reason(text+' lambda'),'shared_8gram')
        self.assertEqual(ex.reason('alpha beta gamma delta epsilon sigma eta theta iota kappa lambda mu'),'near_3gram_jaccard')
        self.assertIsNone(ex.reason('Birch forests surround quiet mountain lakes'))

    def test_nested_reproducible_expansion_preserves_validation_and_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';src=root/'sources';base.mkdir();src.mkdir()
            phrases=['amber birch cedar dune ember','forest granite harbor island juniper',
                'kettle lemon meadow nectar orchard','pebble quartz river stone timber',
                'ultra velvet willow xenon yellow','zebra apple beacon coral daisy',
                'elm fern garden hill iris','jade kiwi lilac moss nutmeg',
                'olive pine quiet rose sage','tulip umber violet wheat xenia']
            for name,values in dict(harmful_train=phrases[:2],harmless_train=phrases[2:4],
                harmful_val=phrases[4:6],harmless_val=phrases[6:8],harmless_kl=phrases[8:]).items():
                (base/(name+'.json')).write_text(json.dumps(values))
            h=['One two three four five six','Seven eight nine ten eleven twelve',
                'Moon star earth mars venus saturn','Red blue green pink brown gold']
            with (src/'advbench.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['goal','target']);writer.writeheader()
                writer.writerows(dict(goal=t,target='Sure') for t in h)
            b=['Guitar piano violin cello flute oboe','Oak maple spruce yew ash palm',
               'Cloud rain snow hail sleet fog','Book pen ink page desk chair']
            (src/'alpaca-empty-input.json').write_text(json.dumps(b))
            (src/'reserved-texts.json').write_text(json.dumps([h[0]]))
            (src/'reserved-sources.json').write_text(json.dumps([dict(texts_file='reserved-texts.json',texts_sha256=data.digest(src/'reserved-texts.json'))]))
            (src/'advbench-source.json').write_text(json.dumps(dict(sha256=data.digest(src/'advbench.csv'))))
            (src/'alpaca-source.json').write_text(json.dumps(dict(texts_sha256=data.digest(src/'alpaca-empty-input.json'))))
            before={p.name:p.read_bytes() for p in base.iterdir()}
            a=data.prepare(base,src,root/'one',seed=9,per_class=4)
            c=data.prepare(base,src,root/'two',seed=9,per_class=4)
            self.assertEqual(a,c)
            self.assertEqual(before,{p.name:p.read_bytes() for p in base.iterdir()})
            for name in ('harmful_train','harmless_train'):
                values=json.loads((root/'one'/(name+'.json')).read_text())
                self.assertEqual(values[:2],json.loads(before[name+'.json']))
                self.assertEqual(len(values),4)
                self.assertNotIn(h[0],values)
            for name in ('harmful_val','harmless_val','harmless_kl'):
                self.assertEqual((root/'one'/(name+'.json')).read_bytes(),before[name+'.json'])
            with self.assertRaises(FileExistsError):data.prepare(base,src,root/'one',per_class=4)

    def test_exhausted_candidates_fail(self):
        with self.assertRaises(ValueError):data.choose(['one two three four'],2,data.Exclusions(),0)


class DiagnosticTests(unittest.TestCase):
    def test_folds_are_balanced_in_each_nested_half(self):
        labels=probe.fold_ids(128,5)
        self.assertEqual(labels,probe.fold_ids(128,5))
        for half in (labels[:128],labels[128:]):self.assertEqual([half.count(i) for i in range(4)],[32]*4)

    def test_streamed_moments_match_nested_reference(self):
        rng=np.random.default_rng(5)
        h=rng.normal(size=(16,2,3,4));b=rng.normal(size=h.shape)
        sums={n:np.zeros((2,4,2,3,4)) for n in ('h','b')};counts=np.zeros((2,4),int)
        assignments=probe.fold_ids(8,2)
        for i in range(16):
            stage,fold=i//8,assignments[i];counts[stage,fold]+=1
            sums['h'][stage,fold]+=h[i];sums['b'][stage,fold]+=b[i]
        result=probe.summarize_moments(sums['h'],sums['b'],counts)
        np.testing.assert_allclose(result['baseline']['direction'],h[:8].mean(0)-b[:8].mean(0),rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(result['expanded']['direction'],h.mean(0)-b.mean(0),rtol=1e-6,atol=1e-7)

    def test_zero_vectors_are_reported_without_nonfinite_json(self):
        z=np.zeros((2,4,1,1,2));result=probe.summarize_moments(z,z,np.ones((2,4),int))
        self.assertEqual(probe.finite(result['expanded']['fold_pair_cosine_mean']),[[None]])


if __name__=='__main__':unittest.main()

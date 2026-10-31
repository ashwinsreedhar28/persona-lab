# Calibration expansion: 128 versus 256 prompts per class

Status: data preparation and activation diagnostics completed locally on 2026-10-31. No new edited
checkpoint has been produced or selected. The current public checkpoints and historical acceptance bar remain.

## Dataset and protection

The separate ignored `abl/data/calibration-v2-20261031/` contains 256 harmful and 256 harmless
calibration prompts. Each retains the original 128-row training split in the same order, followed by
128 deterministic additions. The original 32+32 validation prompts and 100 harmless KL prompts are
copied byte-for-byte. Original fixed data files were hash-checked and remain unchanged.

Added harmful rows come from pinned AdvBench goals; added harmless rows come from pinned Alpaca
instructions with empty input. The AdvBench `target` openings are never inserted into prompts.
The added harmless prompts all received `Safety: Safe` from the local pinned Qwen3Guard-Gen-4B
prompt-only classifier (greedy, 48 classifier tokens, batch one, prefill chunks 256). These are automated
labels without independent human confirmation; existing source labels are not a guarantee of safety.

Before choosing additions, preparation excluded normalized exact matches, any shared word 8-gram,
or word-3gram Jaccard similarity at least 0.5 against existing fixed splits, prior pilot inputs, reserved
HarmBench/StrongREJECT/XSTest text and earlier selected additions. The three reserved source snapshots
contain 400, 313 and 450 benchmark rows respectively. The screening also excludes context text when
present. This is lexical screening, not proof of semantic or pretraining independence. Existing baseline
overlap with these newly reserved sources is not retroactively removed; future confirmation subsets
must be screened against the entire baseline and expanded calibration corpus.

Seed is 20261031. The exact source revisions, source row indices, hashes, rules and numerical results
are in the [aggregate record](../evaluation/results/qwen35-4b-calibration-20261031.json). Raw prompts
stay ignored. Preparation was independently repeated with the reviewed builder and reproduced the
same five dataset files and preparation manifest exactly.

## Predeclared diagnostic

The original pinned Qwen3.5-4B bf16 checkpoint ran on the 16 GB Mac, one model at a time. Activation
capture reuses committed helper source `50619e9ae45707386b14e5d19a253bb00f12abcf`, with `resid_pre`
sites, all post-instruction suffix tokens, thinking disabled and batch one. Slices are fp32 and sums
are accumulated in fp64, so all examples need not remain resident in memory. This differs from an
ordinary fine-tuning run: it estimates mean activation differences and does not optimize model weights.

Four fixed, balanced disjoint folds contain 32 examples per class for the baseline and 64 for the
expansion. Consistency is the mean of the six pairwise fold-direction cosine similarities. The primary
location was fixed before inference at zero-based layer `floor(n_layers/2)` and suffix position `-1`;
for this model that is layer 16. Historical validation AUROC is measured on the unchanged 32+32 prompts.

| At predeclared layer 16, position -1 | Original 128+128 | Expanded 256+256 |
|---|---:|---:|
| Mean pairwise fold-direction cosine | 0.966845 | 0.981304 |
| Validation AUROC | 1.000 | 1.000 |
| Direction norm | 6.01922 | 5.88867 |

The two full-data directions have cosine similarity 0.994701 at that location. The direction remains
very similar while fold consistency increases. Both validation scores are at the ceiling. These are
diagnostics, not evidence of fewer refusals, causal superiority, or preserved broad capability.
The expansion changes both sample size and source composition, so it does not isolate sample size.
Full matrices are published rather than choosing a location after looking at the result.

The forward/diagnostic portion took about 101 seconds; peak MLX allocation was 8.51 GB. All original
model weight/config/tokenizer hashes matched after the run. No decoded responses or new weights were
generated. A behavioral comparison remains necessary before adopting the data for later releases.

## Reproduction from archived inputs

`evaluation/calibration_data.py` is model-free and refuses an existing destination. It requires the
archived fixed `abl/data/` plus pinned source snapshots. The source folder holds `advbench.csv`,
`alpaca-empty-input.json`, their `advbench-source.json`/`alpaca-source.json` hashes, and
`reserved-sources.json` describing JSON text arrays with `texts_file`/`texts_sha256` fields. The
aggregate record lists all official URLs and revisions. Dataset preparation needs no model imports.

```sh
python evaluation/calibration_data.py --base-data abl/data --sources /path/to/pinned/sources \
  --exclude-prompts /path/to/prior-pilot/prompts.json --output abl/data/NEW_CALIBRATION
```

Quality-screen added benign rows before freezing a run, record those labels and their hashes, and use
a clean frozen source checkout for inference. Run the diagnostic with the recorded original snapshot:

```sh
python evaluation/calibration_probe.py --source-code /path/to/frozen/source \
  --data abl/data/NEW_CALIBRATION --model /path/to/pinned/original/Qwen3.5-4B \
  --output abl/out/NEW_CALIBRATION_DIAGNOSTIC
```

The diagnostic requires an exactly doubled, balanced set and checks frozen data hashes. The recorded
run used the harness SHA in the aggregate record; its helper code is the commit above. Software
checks cover exclusion rules, nested deterministic preparation, unchanged validation files, destination
protection, fold balance, streamed mean equivalence and zero-vector handling.

## Next decision

Run a separate, prospective matched weight-edit experiment using the same current implementation for
both 128-row and 256-row arms. Comparing a new edit only with the legacy attempt-1 checkpoint would
confound calibration changes with selection/tied-embedding policy changes. Freeze an independent
confirmation set before generating, report both arms, keep every existing release, and measure
refusal, benign over-refusal, capability, KL, projection and reload integrity. Existing selection and
acceptance thresholds stay fixed. Do not automatically propagate the expanded corpus to other models
based on consistency alone.

[Roadmap and proposed cloud costs](ROADMAP.md) · [Earlier paired comparison](COMPARISON_LONG_RESULTS.md) · [README](../README.md)

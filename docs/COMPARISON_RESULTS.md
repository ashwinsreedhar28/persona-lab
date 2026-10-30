# Qwen3.5-4B fresh matched pilot — 2026-10-30

A descriptive comparison of original bf16 Qwen3.5-4B and the preselected historical attempt-1 edit.
Both received identical token IDs, greedy decoding, thinking disabled and no activation steering.
This is a **short-response pilot**, not a full-response capability or universal refusal-removal result.

| Measure | Original | Attempt 1 |
|---|---:|---:|
| Classifier refusal, harmful prompts | 50/50 | 12/50 |
| Phrase-marker refusal, same responses | 49/50 | 3/50 |
| Classifier refusal, benign prompts | 0/48 | 0/48 |
| Strict objective utility checks | 22/32 | 19/32 |
| First-token KL on 48 benign prompts, versus original | 0 | 0.03033 |
| Harmful responses reaching 256-token allowance | 9/50 | 44/50 |
| Benign responses reaching their 128/256-token allowance | 17/48 | 19/48 |

The classifier changed from refusal to non-refusal on 38 harmful prompts. No prompt changed in the
opposite direction. That is a 76-point reduction (paired standard error 6.10 points) **within these
response prefixes**. Phrase matching misses nine of the edit's twelve classifier-labelled refusals.
All 196 classifier outputs parsed. Classifier labels still need human validation; non-refusal does not
establish helpfulness or the functional correctness of an answer.

The utility check lost three correct answers and gained none, a 9.375-point difference (paired standard
error 5.24 points). This is a small, templated sample with strict output checks. In particular, almost all
arithmetic responses were cut off, so it is not a reliable estimate of overall arithmetic capability.

| Utility component | Original correct | Edited correct | Original / edited token-limit hits |
|---|---:|---:|---:|
| Arithmetic | 1/8 | 0/8 | 7/8 / 8/8 |
| Sorting | 8/8 | 7/8 | 0/8 / 0/8 |
| Field extraction | 8/8 | 8/8 | 0/8 / 0/8 |
| Uppercase conversion | 5/8 | 4/8 | 0/8 / 0/8 |

The remaining 16 benign writing/explanation answers were checked for refusal, not open-ended quality.
Ten original and eleven edited open-ended answers reached 256 tokens. Longer matched responses are
needed before interpreting those answers as complete. The interactive chat clients' 4,096-token default
was not changed; this pilot used its separately fixed 128/256-token limits.

## Data, source and limits

The 50 harmful rows came from unused indices at or after 200 in the fixed 444-row Arditi test pool,
preserving the original reserved first 200. The 32 utility cases use seed 20261030; 16 benign requests
were newly written. Preparation excluded normalized exact/shared-word-8-gram overlap with scanned
prior project artifacts before generation. This cannot exclude semantic overlap or model-pretraining
contamination. The exact prompt-file hash, source indices, model hashes, runtime versions, settings,
per-prompt labels and frozen source revisions are in the [aggregate record](../evaluation/results/qwen35-4b-pilot-20261030.json).
Raw prompts and responses remain in ignored local `abl/out/`.

The project source was frozen at `5d9da7452a1145f92215e8b57a3bd932374c0293`; the exact harness is at
[`304f455fb00c`](https://github.com/ashwinsreedhar28/persona-lab/blob/304f455fb00c/evaluation/local_pilot.py).
Qwen3Guard-Gen-4B revision `6ec42827da0c1ff11e7a49dc269d2e810d27e108` ran locally in a separate process.
Checkpoint lineage for the historical edit remains incomplete, so this does not reconstruct its entire
original training/extraction history. No model weights, selection rules or acceptance thresholds were
changed using these results. The historical JBB/GSM8K table and original fixed acceptance bar remain
separate and unchanged. This pilot does not rank Llama, Kimi or the other saved model families.

[Method and reproduction](COMPARISON.md) · [Model status](MODEL_STATUS.md) · [README](../README.md)

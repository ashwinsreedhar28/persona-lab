# Qwen3.5-4B paired rerun with 4,096-token responses

The same 98 prompts from the [short pilot](COMPARISON_RESULTS.md) were rerun with an allowance of
**4,096 generated tokens for every prompt**, replacing 128/256. The original bf16 checkpoint and
preselected attempt-1 edit, prompt text/order/targets, greedy decoding, thinking disabled, batch and
prefill settings, frozen code and local classifier were kept the same. Checkpoint hashes were reverified.
Both models received the same token IDs, also matching the earlier run. The models ran sequentially
on the local 16 GB Mac, with no cloud cost or activation steering. No weights were changed.
The longer answers matched the exact displayed short-run response prefix in 196/196 cases;
the aggregate record lists any nonmatching IDs separately.

| Measure | Original, 128/256 | Attempt 1, 128/256 | Original, 4,096 | Attempt 1, 4,096 |
|---|---:|---:|---:|---:|
| Classifier refusal, 50 harmful prompts | 50/50 | 12/50 | 50/50 | 7/50 |
| Marker refusal, same 50 prompts | 49/50 | 3/50 | 49/50 | 4/50 |
| Classifier refusal, 48 benign prompts | 0/48 | 0/48 | 0/48 | 0/48 |
| Strict objective utility | 22/32 | 19/32 | 29/32 | 27/32 |
| Harmful token-limit hits | 9/50 | 44/50 | 0/50 | 0/50 |
| Benign token-limit hits | 17/48 | 19/48 | 0/48 | 0/48 |

The paired harmful-refusal difference at 4,096 tokens is -86.0
percentage points (standard error 4.96 points).
All 196 classifier labels parsed. Giving the classifier the full answers changed five edited labels
from refusal to non-refusal; all other labels stayed the same. These are model estimates and need
human validation; a non-refusal label does not establish useful or correct compliance. No response
reached the new limit, resolving the cutoff issue on this set.

| Utility component | Original correct | Attempt 1 correct | Original / edited token-limit hits |
|---|---:|---:|---:|
| Arithmetic | 8/8 | 8/8 | 0/8 / 0/8 |
| Sorting | 8/8 | 7/8 | 0/8 / 0/8 |
| Field extraction | 8/8 | 8/8 | 0/8 / 0/8 |
| Uppercase conversion | 5/8 | 4/8 | 0/8 / 0/8 |

The 16 open-ended benign answers were checked for refusal, not writing quality; their token-limit hits
were 0/16 original and 0/16 edited.
The 32 templated utility checks do not estimate broad capability. Benign first-token KL(base || edit)
was 0.03033 on 48 prompts.
Both models solved all eight arithmetic tasks with the longer allowance. The remaining objective
misses were in uppercase conversion (three original, four edited) and sorting (zero original, one edited);
neither component was truncated. Post-generation hashes confirm both checkpoints and the earlier
run's artifacts stayed unchanged.

This is a response-length sensitivity check on a previously evaluated prompt set. The larger budget was
chosen after the short pilot's cutoffs were observed; these are not fresh confirmation prompts. No
direction or model was selected using this rerun. The historical JBB/GSM8K tables and fixed acceptance
bar are unchanged. Semantic/pretraining contamination and the historical edit's complete source lineage
remain unverified. This comparison does not rank the other model families.

The [aggregate record](../evaluation/results/qwen35-4b-pilot-long-20261030.json) contains pins, model and prompt hashes, settings,
component scores, per-prompt labels and parent-run lineage. Raw prompts/completions remain in ignored
local `abl/out/`. The exact harness remains
[`304f455fb00c`](https://github.com/ashwinsreedhar28/persona-lab/blob/304f455fb00c/evaluation/local_pilot.py),
with helper source frozen at `5d9da7452a1145f92215e8b57a3bd932374c0293`.

[Method and reproduction](COMPARISON.md) · [Earlier results](COMPARISON_RESULTS.md) · [README](../README.md)

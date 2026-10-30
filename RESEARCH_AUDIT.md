# Mechanistic interpretability audit — 2026-10-30

Scope: the local Qwen refusal-direction, steering and representation experiments.
Existing weights, vectors, datasets, evaluation thresholds and generation artifacts were preserved.
Pre-existing uncommitted Phase A work was preserved. No new 4B behavioral evaluations or weight edits were run.
Kimi's published aggregate claims were not independently reproduced; no Phase B files were changed.

## Findings and implementation checks

| Finding | Assessment / action |
|---|---|
| Projection assumes a finite unit vector | Correct formula, but hooks previously accepted nonunit/NaN vectors. `ablated` now validates norm and hidden size before installing hooks. A4 uses explicit validation rather than an optimization-removable assertion. |
| Activation sites differ | `build_refusal.extract` records **resid_post L**, final prompt token; `abl/a2_directions.cache` records **resid_pre L**, each template suffix token. Post L is pre L+1, except the final boundary. `AddVector` acts after a layer; `AddPre` acts before it. These layer numbers are not interchangeable. Tested explicitly. |
| Position alignment | A2 now checks the suffix against every tokenized prompt before indexing. Tiny hybrid-model tests compare batched and individual extraction, including unequal prompt lengths. |
| Weight orientation | With MLX weights `(out, in)`, output writers need `W' = (I − uuᵀ)W`; embeddings `(vocab, hidden)` need `E' = E(I − uuᵀ)`. Existing orientations and bias projection are correct; tested against activation projection. |
| Tied-unembedding mismatch | A3's `ablated(..., edit_unembed=False)` leaves the read path unchanged; A4's default edits tied embeddings and therefore the read path too. A5 explicitly matches the tied policy. Selection and final evaluation estimate different interventions; the audit preserves historical selection rather than silently reranking. |
| Layer selection | A3 uses validation only, then ranks passing candidates by bypass score. `int(32 * .8)` searches L0–24; this conservatively excludes L25 although `l < .8L` would include it. Historical scope is unchanged. Persona depths and component weights were hand-tuned, not established as optimal. |
| Unsafe resume | A3 previously keyed resumed scores only by layer/position. It now checks hashes of directions, validation data and relevant code, settings, package versions and Git state. Legacy checkpoints without provenance are rejected before loading a model. Preserve them and choose a new `--checkpoint` path. |
| Debug selection | `--limit` used the last N candidates despite promising the first N; corrected. Debug results now go to `selection_debug.json` / `r_hat_debug.npy`, preserving headline outputs. |
| Persona extraction | The implementation averages post-layer activations over generated response tokens, despite the old module docstring saying final prompt token. Corrected documentation and rejected empty/out-of-range response spans, which previously yielded NaNs. CLI builds now seed MLX and save a metadata sidecar. |
| Response-only hooks | `gen_only` tests sequence length, not whether a token belongs to the answer. A one-token prefill is affected; multi-token decoding/speculation is skipped. Tests establish this actual behavior. No cache-aware redesign was made. |

## Signed steering versus projection ablation

Let `d = mean(harmful) − mean(harmless)`, `u = d / ||d||`, and
`h = a u + h_perp`, where `a = h·u`.

| Property | Signed addition | Directional projection |
|---|---|---|
| Local operation | `S_c(h) = h + c u` | `A(h) = h − (h·u)u` |
| Parallel coordinate | `a → a + c` | `a → 0` |
| Dose | Here `c = dose × median residual norm`; A3 induction uses raw `d`, hence `c = ||d||` | Activation-dependent subtraction, effectively `c(h) = −h·u` |
| Inverse / repetition | Invertible translation; repeated additions accumulate | Noninvertible, idempotent projection in exact arithmetic |
| Sign | Reversing dose reverses the translation | `u` and `−u` define the same ablation |
| Representation | At the injection site, the original coordinate remains recoverable by subtracting the offset; within-group covariance is unchanged by a constant translation | The selected coordinate is removed; covariance becomes `P Σ P`, `P = I − uuᵀ` |
| Downstream behavior | Nonlinear layers can amplify, reverse or rewrite the feature | Later unprojected writers can recreate the coordinate; removing it from all writers prevents this in ideal residual arithmetic |

Negative steering is equal to projection only where its chosen constant happens to equal
`−h·u`. Mean-difference magnitude, median residual norm, and feature-coordinate standard deviation
are different scales. The same numeric dose across layers or precisions is not equal intervention strength.

Projecting embeddings and every additive writer keeps the residual orthogonal by induction in exact
arithmetic. Projecting only writer outputs without projecting the incoming residual would preserve its
initial parallel coordinate. Biases must be included. Tied read paths, final normalization, reduced precision,
vision paths and cache behavior require explicit matching; runtime and weight edits need not be bit-identical.
The reference derivation and selection criteria are in
[Arditi et al., sections 2.4 and appendices C/E](https://arxiv.org/html/2406.11717v1).

Tests verify local mathematics, writer/embedding orientation, bf16 output dtype, pre/post hook sites,
mask/cache argument forwarding, context restoration on exceptions, and tiny Qwen hybrid-model extraction.
They do not prove equivalence for all recurrent cached-generation states or multimodal inputs.

## Dataset separation and reproducibility

`python audit_splits.py --data abl/data --out /tmp/persona-split-audit.json` is read-only.
It records full SHA-256 hashes, within-split duplicate counts, cross-split exact and shared-word-8-gram
matches as **indices only**. It handles string, JBB-goal and GSM-question records without printing prompts.

The existing local splits have sizes 128/32 harmful train/validation, 128/32 harmless train/validation,
100 harmless KL, 100/100 JBB, 1,319 GSM8K and 444 fresh harmful pool prompts. The fresh evaluation uses
only the first 200 of the fixed pool. There are no normalized exact duplicates within any split or across
splits. No shared 8-gram crosses a training/validation/KL/evaluation boundary. Three shared-8-gram pairs
occur between JBB benign and harmful goals. These are review flags, not grounds to resplit existing tests.
Exact and n-gram checks cannot exclude semantic paraphrases or model-pretraining contamination.

The manual refusal corpus now rejects normalized duplicates across/within train, validation and test.
Its test remains development-exposed because it was inspected while debugging the judge; enforcing
disjoint prompt strings cannot restore a blind holdout. Topic-matched training pairs also do not isolate
refusal from intent, wording, topic, or other correlated features. Conditioning training inclusion on the
base model's lexical refusal creates another sampling bias.

New A2 statistics, A3 selection reports, refusal reports and persona CLI sidecars record input/code hashes,
settings, installed versions and Git state. Refusal rescoring rejects a mismatched saved model/template/token
limit or changed train/validation corpus; candidate layers/doses in its report come from the trials actually scored. Persona CLI builds use
`--seed 0` by default. Programmatic/UI persona builds retain their existing RNG behavior.

Remaining reproducibility gaps: upstream dataset branches and model IDs are not pinned to revisions;
old artifacts lack complete provenance; persona extraction re-encodes generated text instead of retaining
original sampled token IDs; seed setting alone does not guarantee cross-device/library bitwise replication.
Use immutable local snapshots and archive hashes before a future confirmation run. New metadata does not
retroactively establish the lineage of old artifacts. Historical `refusal.eval.json` contains raw generations;
new raw reports default to gitignored `abl/out/refusal_runs/<vector-stem>.eval.json` with `--report` as an
override. Rescoring can still read the historical report and writes to the new default location.

## Which conclusions the artifacts support

The following are **existing saved measurements**, not fresh generation results. Sources:
`abl/out/metrics.json`, `attempts.json`, `selection.json`, `directions.npz`, and per-attempt `projection.json`.

| Variant | JBB marker refusals / 100, non-thinking | First-token KL, n=100 | GSM8K accuracy, n=1,319 | Paired difference vs base |
|---|---:|---:|---:|---:|
| Base bf16 | 95 | 0 | 87.72% | — |
| Signed bf16 steering | 0 | 0.7975 | 51.02% | −36.69 points (SE 1.39) |
| Weight edit, attempt 1 | 10 | 0.0640 | 85.90% | −1.82 points (SE 0.77) |

The saved 4-bit signed vector selects **post L16**, dose magnitude 0.6. The bf16 signed vector selects
**post L12**, also 0.6; the Phase A table uses that bf16 vector. A3 selects **pre L13, suffix position −7**.
Both signed artifacts have finite differences, positive medians and unit saved directions. A2's saved
`r` equals its saved harmful-minus-harmless means exactly; normalized A3 selection has cosine 1.0 with
attempt 1's saved direction. This checks internal artifact consistency, not independent re-extraction.

Attempt 1's recorded maximum edited/base mean absolute projection is **0.7849%**;
runtime-hook versus edited first-token KL is **0.001107** on validation. This supports approximate
mechanical removal on those recorded text forwards. It does not establish the cause of every remaining
rounding difference or equivalence for every input. A5 averages absolute projections over all real tokens,
so longer prompts have more weight; it is not a last-token-only or equal-per-prompt statistic. Very small
base projections would make ratios unstable; the checker does not define a zero-denominator convention.

All ten attempts are represented in `attempts.json`; none meets the full fixed bar. Attempt 2 has
2/100 marker refusals but KL 0.1195. Attempts 5/6 each have 6/100. Attempts after the first are exploratory
after JBB exposure, even where their ranking order was fixed before their own evaluation. Fresh-prompt
results must be reported separately before making a confirmation claim; their presence as a dataset is
not evidence that the evaluations finished. `abl/RESULTS.md` has stale opening text about only three
attempts and PENDING sections despite later run notes and saved GSM8K metrics. It was not rewritten
because it already contains ongoing user work; this audit records the discrepancy.

Supported narrowly: interventions change the measured refusal proxies; signed steering and weight editing
have different recorded capability costs at their selected settings; the weight edit substantially reduces
the selected projection. The small 4-bit signed test is exploratory evidence for bidirectional behavioral
effects, with a same-model judge and short, development-exposed outputs.

Still unverified: universal refusal removal; general capability preservation; a unique or exclusively
refusal-specific feature; causal mediation of all refusal behavior; global superiority of projection over
signed steering; semantic safety/compliance from marker absence; optimal layers; replicated effects across
seeds/models/precisions. The comparison changes direction, extraction dataset, template position and
intervention location/scope, so it cannot isolate the operator's causal contribution. GSM8K within a
2-point point-estimate threshold is not statistical equivalence: attempt 1's approximate 95% interval is
−3.33 to −0.31 points. Neither a low first-token KL nor one arithmetic benchmark proves broad preservation.
Cosine alignment of down-projection columns and a contrast direction establishes geometric alignment,
not neuron selectivity, necessity, or sufficient behavioral control. Persona sweep judgments from one prompt
and one same-model judge do not establish general hostility or capability effects.

Next evidence should come from a locked, unused confirmation split; independent blinded behavior labels;
paired uncertainty estimates; topic/wording controls; random-direction controls; and a comparison using
the same direction, site, data and matched intervention magnitudes. These experiments were not run here.

## Changes and verification

- `research_checks.py`: unit-direction validation, hashing/provenance, normalized overlap checks.
- `audit_splits.py`: read-only split audit CLI.
- `abl/common.py`: validate ablation direction and addition layer/vector before hook installation.
- `abl/a2_directions.py`: per-prompt suffix check and extraction metadata.
- `abl/a3_select.py`: provenance-checked resume, selectable checkpoint path, corrected/debug-isolated limits.
- `abl/a4_edit.py`: explicit direction validation.
- `build_refusal.py`: corpus separation check, rescore settings/corpus consistency, metadata, validated direction normalization, gitignored default raw-report path with legacy reading.
- `build_vector.py`: corrected extraction description, span validation, seeded CLI and metadata sidecar.
- `test_research_checks.py`, `test_interventions.py`: 20 new tests; existing `test_refusal.py` preserved.
- `README.md`: audit link and narrower capability/caveat wording.

Validation commands:

```sh
python -m unittest -v test_research_checks
python -m unittest -v test_interventions test_refusal
python audit_splits.py --data abl/data --out /tmp/persona-split-audit.json
python abl/check.py
```

The first command passes 4 tests; the second passes 19 (16 new and 3 existing). MLX tests need Metal access;
this sandbox blocks Metal initialization, so those tests ran with approved host access. No models downloaded.
The full done-bar check is expected to fail criterion 1 (10/100 > 5/100); no threshold was changed.

Audit-time SHA-256 identifiers (saved evidence can change as ongoing runs finish):

| File | SHA-256 |
|---|---|
| `abl/out/metrics.json` | `2d62c0da95cf76a278f98fd9e2399a630889b0448879d13fdcb5e544b8f7abc8` |
| `abl/out/selection.json` | `edc5bfc933dc748ffc321ee1a7dbb91ffe6ef2ef187f31b70fa998d3e1708893` |
| `abl/out/directions.npz` | `01d698efc084c209932bf29ae6cc6d2448ccd730d78b163f4d20e7c0e8f381fd` |
| `abl/out/attempt1/projection.json` | `a204ed245e6a43a8a9fa4886f93f4fc87f36e912a6e2a77d3450ea6fa25367ce` |

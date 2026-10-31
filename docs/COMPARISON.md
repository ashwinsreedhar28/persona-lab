# Local matched-comparison pilot

[Completed 2026-10-30 results, cutoffs and limits](COMPARISON_RESULTS.md).

[Completed 4,096-token rerun on the same prompts](COMPARISON_LONG_RESULTS.md).

This is a small comparison of Qwen3.5-4B bf16 and the preselected historical attempt-1 edit. It uses
50 unused harmful-pool rows starting after the original reserved first 200, 32 seeded synthetic utility
tasks and 16 new benign writing/explanation requests. Normalized exact and shared-word-8-gram matches
against prior local data and JSON artifacts are excluded before either model generates. Semantic overlap
and pretraining contamination remain unverified. No evaluation responses are used to choose a direction.

Generation is greedy with thinking disabled, identical chat templates, no activation steering, a batch of
two, prefill batch one and prefill chunks of 256 tokens. Limits are fixed per prompt (128 or 256 generated
tokens). Both models see identical token IDs. A separate, pinned Qwen3Guard-Gen-4B process labels refusal;
classifier labels are estimates, not human ground truth. Marker labels and token-limit hits are separate.
32 exact-scored arithmetic, sorting, extraction and case-conversion tasks test limited benign utility;
the 16 open-ended answers are not certified for quality. First-token KL is descriptive on these 48
benign prompts. This pilot does not replace the original full acceptance bar or establish broad capability.

## Reproduction

The first run uses frozen project source at `5d9da7452a1145f92215e8b57a3bd932374c0293` and the exact
harness at `304f455fb00c` (`evaluation/local_pilot.py`). Its ignored run folder contains prompt/data
hashes, model file hashes, settings, frozen-code hashes, raw generations, classifier labels and report.
The current preparation code also handles the relocated legacy report. For exact reruns, use the recorded
harness revision and original matching input files; no upstream-pretraining freshness is asserted.

Prepare a frozen checkout of the desired committed source, then use the existing MLX environment:

```sh
python evaluation/local_pilot.py prepare --repo . --code /path/to/frozen/source --run abl/out/NEW_PILOT
python evaluation/local_pilot.py generate --run abl/out/NEW_PILOT --variant base
python evaluation/local_pilot.py generate --run abl/out/NEW_PILOT --variant edited
python evaluation/local_pilot.py judge --run abl/out/NEW_PILOT
python evaluation/local_pilot.py report --run abl/out/NEW_PILOT
```

Preparation requires the fixed `abl/data/` files and cached pinned Qwen/Qwen3Guard originals plus
`models/Qwen3.5-4B-abliterated`. It refuses an existing run directory or insufficient unused data.
It is a Qwen-specific local pilot, not a generic benchmark for every model family. A previously used
prompt set should be rerun from its saved, verified inputs rather than described as fresh again.
Always run one full model at a time on a 16 GB Mac. Publish aggregates and per-prompt labels only;
raw prompts/responses remain in ignored `abl/out/`. The legacy attempt-1 source lineage is incomplete,
so this is a descriptive checkpoint comparison, not a fully reconstructed causal experiment.

## Repeating with a larger response allowance

The 4,096-token rerun reused the original saved 98 prompts and targets, rather than preparing or
selecting new data. Only each row's `max_tokens` changed from 128/256 to 4096. Both model-file
inventories were reverified, and prompt-token-ID hashes matched between models and between runs.
The original run and its public aggregate record remain unchanged. The allowance was chosen after
observing the first run's cutoffs, so this is a length-sensitivity rerun on previously evaluated data.

To repeat this variant from the saved original inputs, create a new empty ignored run folder, copy its
`prompts.json`, `config.json` and source provenance, set every prompt's `max_tokens` to 4096, and update
`prompts_sha256` in the copied config to the new file's SHA-256. Preserve prompt text, IDs, ordering,
targets, model paths, checked hashes, frozen harness/source and all other settings; verify the model
files against the copied inventory first. Record the original config/prompt/report hashes as parent
lineage. Then run the same two `generate` stages, local `judge` and `report` commands above with the
new run folder, omitting `prepare`. Keep raw text local and report remaining token-limit hits.

[README](../README.md) · [Model evidence](MODEL_STATUS.md) · [Audit](RESEARCH_AUDIT.md)

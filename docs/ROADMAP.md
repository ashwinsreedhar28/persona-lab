# Work order and cloud budget — 2026-10-31

The GPT-OSS metadata audit is complete. At the user's request, further model work is paused.
Its edited checkpoint is saved on public HF and the retained volume. Serialization/reload passed;
manual testing reported continued refusals. [Completed run and diagnostic](GPT_OSS_RESEARCH.md).

## Current order

1. **GPT-OSS diagnostic:** document the negative manual result, audit checkpoint/formatting/parsing
   evidence and the nonspecific first-analysis-token proxy. This audit is complete; the checkpoint is unchanged.
2. **GLM-4.7-Flash:** checked CPU fixtures and source-specific license packaging exist; full-size
   conversion/edit, save, upload and validation remain pending.
3. **Native DeepSeek-V2-Lite-Chat:** replaces DeepSeek-R1-0528-Qwen3-8B. No need to run both. Native
   writer mapping and model-license review remain pending; no full-size checkpoint exists.
4. **Ministral 3 8B Instruct BF16:** approved addition; architecture/normalization, packaging and
   download-layout review remain pending. Text-only validation would not cover vision.
5. **OLMo 3 7B Instruct:** approved addition; architecture/normalization and baseline review remain
   pending. No full-size checkpoint exists.

Items 2–5 are deferred until a future session. This order records requested scope, not completed
runs or a claim of successful refusal removal. Future experiments require matched benign utility
and independently reviewed behavioral evidence, rather than selecting on an unvalidated token proxy.

| Target | Pinned source revision | Status |
|---|---|---|
| `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | Saved/verified; negative manual result; metadata audit complete |
| `zai-org/GLM-4.7-Flash` | `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | CPU fixtures; full-size pending |
| `deepseek-ai/DeepSeek-V2-Lite-Chat` | `85864749cd611b4353ce1decdb286193298f64c7` | Native adapter/license review pending; older 2024 release |
| `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `f6fae9795746f63c9be8344932f01275f3c63734` | Adapter/download-layout review pending |
| `allenai/Olmo-3-7B-Instruct` | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` | Adapter/normalization review pending |

The canceled Qwen-based DeepSeek target was not run. Generic `--model` does not imply arbitrary
architecture support. Source preservation, hashes and fresh reloads are required for every model.

## Calibration and evaluation

The [expanded 256+256 dataset and local diagnostics](CALIBRATION_V2.md) are complete, but the
controlled Qwen weight-edit comparison has not been completed. The expanded split has **not** earned
promotion. GPT-OSS used the original 128+128 training split and fixed validation/KL files. Directions
are model-specific. No threshold or original Qwen attempt limit has changed.

For GPT-OSS, the selected proxy tokens were `User` and `We` in the analysis channel. A proxy pass is
not a validated final-answer refusal reduction. First establish an interpretable measurement before
any claims of improvement. Broader benign capability checks remain incomplete for the new families.

## Approved resources and actual state

The initial **$35 Runpod compute cap** and **300 GB temporary standard network volume** were approved.
Recorded estimated compute so far is **$3.9168**, including the earlier failed setup worker; this is
an estimate, not an invoice. All project GPU workers are terminated. The retained volume holds GPT-OSS
original/prepared/edited copies and evidence and costs about **$21/month** (~$0.70/day, 30-day estimate).
Storage continues after GPU shutdown. No new GPU was started for the metadata diagnostic.

The previous working 96 GB GPU cost $2.49/hour. Availability and the actual quote must be checked before
any new allocation, with the cumulative budget carried forward and a bounded termination deadline.
The earlier 6–12 hour estimate covered only three targets; it is not a measured forecast for the
expanded five-model scope. Sequential storage cleanup requires verified backups and must preserve originals.

Six public experimental releases currently total about 190 GB. HF is the intended long-term destination;
public storage is best effort, not a guaranteed unlimited archive. No HF subscription was purchased.
New model runs must include reviewed model-card limitations, retained upstream license/notice assets,
file verification and a fresh download/reload before claiming a completed release. Source pins and
checkpoint identities are recorded in [storage](MODEL_STORAGE.md).

Pricing references (last checked 2026-10-31): [Runpod](https://www.runpod.io/pricing),
[HF limits](https://huggingface.co/docs/hub/storage-limits), [HF PRO](https://huggingface.co/pro).

[GPT-OSS evidence](GPT_OSS_RESEARCH.md) · [Model status](MODEL_STATUS.md) · [README](../README.md)

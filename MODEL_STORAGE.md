# Saved edited checkpoints

Updated 2026-10-30. All five selected models have separately saved bf16 edits. Qwen3.5-4B also has a
second local experimental attempt. Weight files are excluded from Git; current Hugging Face releases are private.

| Model / variant | Saved edited weights | Additional copies |
|---|---|---|
| Qwen3.5-4B, attempt 1 | [Private HF release](https://huggingface.co/ashwin-sreedhar/qwen3.5-4B-abliterated-bf16) | Mac: `models/Qwen3.5-4B-abliterated` |
| Qwen3.5-4B, attempt 2 | Mac: `models/Qwen3.5-4B-abliterated-refusal-first-bf16` | No HF upload, per the requested storage choice; the historical folder name is not a quality ranking |
| Gemma 3 4B IT | [Private HF release](https://huggingface.co/ashwin-sreedhar/gemma-3-4b-it-abliterated-bf16) | Mac and Runpod volume: `models/Gemma-3-4b-it-abliterated-bf16` |
| Qwen3-8B | [Private HF release](https://huggingface.co/ashwin-sreedhar/qwen3-8b-abliterated-bf16) | Runpod volume: `models/Qwen3-8B-abliterated-bf16` |
| Llama 3.1 8B Instruct | [Private HF release](https://huggingface.co/ashwin-sreedhar/llama-3.1-8b-instruct-abliterated-bf16) | Runpod volume: `models/Llama-3.1-8B-Instruct-abliterated-bf16` |
| Kimi Linear 48B A3B Instruct | Runpod volume: `models/Kimi-Linear-48B-A3B-Instruct-abliterated` (20 shards, about 98.25 GB) | Direction, hashes and reproducibility metadata backed up separately to the Mac; no HF weight upload |

Runpod paths above are under `/workspace/` on the approved persistent 250 GB network volume.
The GPU pods are terminated after use; the volume remains. Its recorded rate is approximately $17.50/month.
No larger volume was provisioned. Private HF links require account/repository access; these are not public weight releases.

## Integrity and originals

The new Qwen attempt 2, Gemma, Qwen3, Kimi and Llama exports check every tensor's saved bytes, shape and dtype,
preserve unedited tensors, and compare a fresh reload with the edited in-memory reference. Source files are
SHA-256 checked before and after export. Each bundle records a source revision, file hashes, saved direction,
selection rule, environment, tensor checks, reload result, manifest and completion seal.
The private distributions are verified against their uploaded file hashes, including the model card and attributes.

Legacy Qwen attempt 1 has uploaded-file hash verification and a terminal smoke test; its older export does not
have all the newer provenance. Its local weight hashes match its private HF copy. Qwen attempt 2 and the local
Gemma bundle passed full local file-hash checks again when this inventory was prepared.
Kimi's completion seal and all file sizes/presence were checked after GPU termination and volume reattachment.

Original weights were never edited in place. Kimi's original snapshot remains separately on the volume under
the HF cache. Other upstream originals are identified by immutable revisions in checkpoint metadata; their
temporary GPU download caches are not additional persistent backups. Saved edited folders are independent
of those caches and contain the configuration/tokenizer assets needed to load them.

## Loading a saved copy

Use an empty destination when downloading a pinned private HF release. New-format bundles can be checked
without loading a model with `abliterate.py verify`, using the saved folder as its argument. Legacy Qwen attempt 1 predates that
completion-seal format. The Mac CLI and CUDA chat commands are in [ABLITERATION.md](ABLITERATION.md).

Storage/reload verification establishes checkpoint integrity. Experiment settings, diagnostics and behavioral
limits are recorded separately in [MODEL_STATUS.md](MODEL_STATUS.md). Model assets retain their upstream licenses
and access conditions; the repository's MIT license applies to its own code.

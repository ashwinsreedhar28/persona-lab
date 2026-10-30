# Saved edited checkpoints

Updated 2026-10-30. All five selected models have separately saved bf16 edits. Qwen3.5-4B also has a
second local experimental attempt. Weight files are excluded from Git; all five selected models also have public experimental Hugging Face releases.

| Model / variant | Saved edited weights | Additional copies |
|---|---|---|
| Qwen3.5-4B, attempt 1 | [Public HF release](https://huggingface.co/ashwin-sreedhar/Qwen3.5-4B-abliterated-bf16-public) | Mac: `models/Qwen3.5-4B-abliterated` |
| Qwen3.5-4B, attempt 2 | Mac: `models/Qwen3.5-4B-abliterated-refusal-first-bf16` | No HF upload, per the requested storage choice; the historical folder name is not a quality ranking |
| Gemma 3 4B IT | [Public HF release](https://huggingface.co/ashwin-sreedhar/Gemma-3-4b-it-abliterated-bf16-public) | Mac: `models/Gemma-3-4b-it-abliterated-bf16` |
| Qwen3-8B | [Public HF release](https://huggingface.co/ashwin-sreedhar/Qwen3-8B-abliterated-bf16-public) | Earlier private HF archive |
| Llama 3.1 8B Instruct | [Public HF release](https://huggingface.co/ashwin-sreedhar/Llama-3.1-8B-Instruct-abliterated-bf16-public) | Earlier private HF archive |
| Kimi Linear 48B A3B Instruct | [Public HF release](https://huggingface.co/ashwin-sreedhar/Kimi-Linear-48B-A3B-Instruct-abliterated-bf16) (20 shards, about 98.25 GB) | Direction, hashes and reproducibility metadata backed up to the Mac; original source pinned in export metadata |

The project's 250 GB Runpod volume was deleted on 2026-10-30 after all five public HF backups were
reverified. Its Runpod weight copies and caches are gone; no project GPU/CPU pod remains. The volume's
approximately $17.50/month storage charge no longer continues. The four earlier private HF releases remain as archives. Qwen attempt 2
has not been uploaded. The five public releases total about 148 GB. HF provides public storage on a
[best-effort basis](https://huggingface.co/docs/hub/storage-limits); free accounts include 100 GB of
private storage. No paid HF plan was purchased.

## Integrity and originals

The new Qwen attempt 2, Gemma, Qwen3, Kimi and Llama exports check every tensor's saved bytes, shape and dtype,
preserve unedited tensors, and compare a fresh reload with the edited in-memory reference. Source files are
SHA-256 checked before and after export. Each bundle records a source revision, file hashes, saved direction,
selection rule, environment, tensor checks, reload result, manifest and completion seal.
Public distributions are verified against every uploaded file hash, including model cards and attributes.
Each safetensors header contains a modification notice. Only header metadata changes relative to the
saved edited weights: tensor payloads, shapes, dtypes and offsets are preserved. New public file hashes
and the source export manifest hash are recorded in each release manifest. License/notice assets are
included; public packaging does not create new behavioral or capability evidence.

Legacy Qwen attempt 1 has uploaded-file hash verification and a terminal smoke test; its older export does not
have all the newer provenance. Its local weight hashes match its private HF copy. Qwen attempt 2 and the local
Gemma bundle passed full local file-hash checks again when this inventory was prepared.
Kimi's completion seal and all file sizes/presence were checked after GPU termination and volume reattachment.

Original weights were never edited in place. Upstream originals are identified by immutable revisions in
checkpoint metadata. Kimi's original volume cache was removed with the volume; future GPU sessions
require redownloading the original or edited checkpoint from the appropriate HF repository. Saved edited folders are independent
of those caches and contain the configuration/tokenizer assets needed to load them.

## Loading a saved copy

Use an empty destination when downloading a pinned public HF release. New-format bundles can be checked
without loading a model with `abliterate.py verify`, using the saved folder as its argument. Legacy
Qwen attempt 1 requires its separate `release_manifest.json` file-hash check. The Mac CLI and CUDA chat commands are in [ABLITERATION.md](ABLITERATION.md).

Storage/reload verification establishes checkpoint integrity. Experiment settings, diagnostics and behavioral
limits are recorded separately in [MODEL_STATUS.md](MODEL_STATUS.md). Model assets retain their upstream licenses
and access conditions; the repository's MIT license applies to its own code.

## Public release pins

These revisions identify the reviewed distribution metadata and exact file hashes:

| Model | Revision |
|---|---|
| Qwen3.5-4B, attempt 1 | [`f7a377746d56`](https://huggingface.co/ashwin-sreedhar/Qwen3.5-4B-abliterated-bf16-public/tree/f7a377746d561612853cec8d59aadcf1684f482b) |
| Gemma 3 4B IT | [`4189fe1c4559`](https://huggingface.co/ashwin-sreedhar/Gemma-3-4b-it-abliterated-bf16-public/tree/4189fe1c455962e7623032929d4484bf3553a502) |
| Qwen3-8B | [`490ba3303053`](https://huggingface.co/ashwin-sreedhar/Qwen3-8B-abliterated-bf16-public/tree/490ba3303053d88596ef0b6f7ba7ec3aa9d7fe95) |
| Llama 3.1 8B Instruct | [`26f77a1c4263`](https://huggingface.co/ashwin-sreedhar/Llama-3.1-8B-Instruct-abliterated-bf16-public/tree/26f77a1c4263664086d507768371339d80299ebc) |
| Kimi Linear 48B A3B Instruct | [`b9f8c132f896`](https://huggingface.co/ashwin-sreedhar/Kimi-Linear-48B-A3B-Instruct-abliterated-bf16/tree/b9f8c132f896b5aba54f760382673c4cc191b17b) |

The [cloud helpers](cloud/README.md) reproduce verification and publication without modifying the local checkpoint.

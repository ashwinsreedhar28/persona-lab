# Model and checkpoint status

Updated 2026-10-31. An implemented adapter, a saved checkpoint, and a validated behavioral result are
separate milestones. Saved locations are listed in [MODEL_STORAGE.md](MODEL_STORAGE.md). New weight files and raw evaluation outputs are excluded from Git. The historical
`legacy/refusal.eval.json` predates that output policy and remains an explicitly documented legacy artifact.

| Model | Implementation / storage | Verification and limits |
|---|---|---|
| Qwen3.5-4B, attempt 1 | Separately saved bf16 edit; [public HF release](https://huggingface.co/ashwin-sreedhar/Qwen3.5-4B-abliterated-bf16-public); legacy export | Legacy checkpoint, uploaded-file SHA-256 checks and terminal smoke test. Historical provenance is incomplete. Classifier refusals 18/100; original acceptance bar failed. |
| Qwen3.5-4B, attempt 2 | Separate local bf16 checkpoint; no HF upload | New pipeline verified 738 tensor payloads, 67 edited tensors, unchanged originals and fresh-process probe reload. Imported historical direction. Classifier refusals 53/100 CUDA, 54/100 MLX; the 2/100 marker score is misleading. |
| Gemma 3 4B IT | Separately saved bf16 edit; [public HF release](https://huggingface.co/ashwin-sreedhar/Gemma-3-4b-it-abliterated-bf16-public) | All 883 tensor payloads checked, 70 edited tensors, original hashes unchanged. Fresh-process probe reload matched exactly. All 37 public-release files verified. Also downloaded and manifest-verified on Apple silicon; terminal chat passed a short harmless READY smoke with steering off. Max projection ratio 0.358% (passes 1%); held-out harmless KL 0.2078 (above 0.1). Full behavioral/capability evaluation is pending. Upstream Gemma terms continue to apply. |
| Kimi Linear 48B A3B Instruct | Separately saved bf16 edit; [public HF release](https://huggingface.co/ashwin-sreedhar/Kimi-Linear-48B-A3B-Instruct-abliterated-bf16); former Runpod copies removed after verified backup | All 20,493 tensors checked, 6,711 edited. All 53 public-release files verified. Original hashes unchanged; fresh-process probe reload matched exactly. Held-out harmless KL 0.0756 and max projection ratio 0.636% pass the diagnostic references. Cached chat passed a harmless probe. Broad behavior/capability remains unverified. |
| Qwen3-8B | Separately saved bf16 edit; [public HF release](https://huggingface.co/ashwin-sreedhar/Qwen3-8B-abliterated-bf16-public) | All 399 tensor payloads checked, 73 edited tensors, fresh-process probe reload matched exactly; 34 public-release files verified. Max projection ratio 1.075% (above 1%); held-out harmless KL 0.1152 (above 0.1). Full behavioral/capability evaluation is pending. |
| Llama 3.1 8B Instruct | Separately saved bf16 edit; [public HF release](https://huggingface.co/ashwin-sreedhar/Llama-3.1-8B-Instruct-abliterated-bf16-public) | All 291 tensor payloads checked, 65 edited tensors, original hashes unchanged. Fresh-process probe reload matched exactly; 36 public-release files verified. Held-out harmless KL 0.0242; max projection ratio 1.387% (above 1%). Full bf16 chat and a three-token cached/uncached harmless probe also completed on a 48 GB A40. Broader behavior/capability remains unverified. |

The generic `--model` argument dispatches only to explicit, checked architectures. It does not support
arbitrary models automatically. Kimi K3/K3.1 need separate architecture, licensing, hardware and validation work.

New [GLM and DeepSeek research support](GLM_DEEPSEEK_RESEARCH.md) is at the small CPU fixture stage.
GLM now covers the native and upstream expert layouts, and DeepSeek-R1-0528-Qwen3-8B uses the existing
Qwen3 backbone mapping. Neither has a completed full-size export or demo yet. GPT-OSS support remains
unimplemented. These targets must not be counted among the five completed releases above.

The [fresh Qwen pilot](COMPARISON_RESULTS.md) measures attempt 1 separately from the historical
JBB scores. Its 256-token response-prefix limit and narrow utility tasks constrain interpretation.
The [4,096-token rerun](COMPARISON_LONG_RESULTS.md) reuses the same prompts and reports remaining
token-limit hits separately.

## Interpreting the Kimi edit

The full Kimi export succeeded on a 96 GB RTX PRO 6000 Blackwell GPU with batch size one and
expandable CUDA allocator segments. This narrowly tested configuration replaces the earlier assumption
that a 141 GB GPU was required; larger GPUs still provide more room for context and batching.
The edited checkpoint is saved on public HF. The former Runpod volume was deleted after backup
verification; a new GPU session needs a fresh download.
See [the CUDA chat instructions](ABLITERATION.md#chat-with-a-saved-cuda-checkpoint).

A manual test showed an initial refusal followed by a different response after a follow-up. That is one
uncontrolled observation, without a matched base-model conversation. It demonstrates inconsistent refusal
behavior, not universal removal or the functional correctness of generated content. No test prompt,
completion, or generated program from that conversation is included here.

## Testing the saved Gemma checkpoint on Apple silicon

From the repository root, after downloading the public release to the folder shown below:

```sh
python evil_chat_mlx.py --model models/Gemma-3-4b-it-abliterated-bf16 --dose 0
```

The default answer allowance is 4,096 generated tokens (`--max_tokens` changes it). Test one local model
at a time on a 16 GB Mac. This exercises plain chat; the bundled Qwen steering vectors have not been
validated for Gemma. A short successful load/response does not resolve the KL or behavioral limitations.

## Reproducing and verifying an export

Follow [ABLITERATION.md](ABLITERATION.md). Immutable source revision, original/output file hashes,
direction, selection rule, data hashes, code hashes, environment, tensor checks and probe results are saved
with each new checkpoint. `verify` checks those files without loading the model. Three harmless prompts
over three uncached greedy steps test save/reload consistency, not general capability or complete behavior.
Originals are never edited in place; source revisions are recorded separately. During a run, failed
exports stay under uniquely named `.incomplete-*` paths. Deleted cloud storage is not a backup.

The historical Qwen result tables and the original fixed acceptance bar are not silently replaced by the
new export selection rule. All ten attempts are reported in the [aggregate classifier review](../abl/semantic_refusal_review.json).
Those responses were already exposed during development. A fresh, independently reviewed evaluation is
required before making a stronger behavioral claim, and non-refusal alone does not establish a correct answer.

## Distribution and licenses

All five selected models have public experimental HF releases, linked above. The four earlier private
releases remain unchanged as archives; Qwen attempt 2 stays local. Public bundles include model cards,
upstream license assets and modification notices inside safetensors headers. Header packaging changes
file hashes, while tensor payloads, shapes, dtypes and offsets match the verified edited checkpoints.
Release manifests link the local and public hashes; every uploaded file is verified.

Gemma and Llama usage and redistribution remain subject to their upstream terms. Downloading their
original checkpoints requires the publisher's access agreement. The included notices and policies are
part of these derivative releases. Repository code is MIT; that does not relicense third-party models.
Public HF storage is best effort, rather than guaranteed unlimited archival storage.

The project's 250 GB Runpod volume was deleted on 2026-10-30 after all five public releases were
reverified. Project compute is off and the volume's approximately $17.50/month storage charge has ended.
Future GPU sessions need new storage and a checkpoint download. The export CLI itself provisions no GPU.

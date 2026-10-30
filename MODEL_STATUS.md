# Model and checkpoint status

Updated 2026-10-30. An implemented adapter, a saved checkpoint, and a validated behavioral result are
separate milestones. New weight files and raw evaluation outputs are excluded from Git. The historical
`refusal.eval.json` predates that output policy and remains an explicitly documented legacy artifact.

| Model | Implementation / storage | Verification and limits |
|---|---|---|
| Qwen3.5-4B, attempt 1 | Existing local bf16 edit; private HF release `ashwin-sreedhar/qwen3.5-4B-abliterated-bf16` | Legacy checkpoint, uploaded-file SHA-256 checks and terminal smoke test. Historical provenance is incomplete. Classifier refusals 18/100; original acceptance bar failed. |
| Qwen3.5-4B, attempt 2 | Separate local bf16 checkpoint; no HF upload | New pipeline verified 738 tensor payloads, 67 edited tensors, unchanged originals and fresh-process probe reload. Imported historical direction. Classifier refusals 53/100 CUDA, 54/100 MLX; the 2/100 marker score is misleading. |
| Gemma 3 4B IT | Separately saved bf16 edit; verified private release `ashwin-sreedhar/gemma-3-4b-it-abliterated-bf16` | All 883 tensor payloads checked, 70 edited tensors, original hashes unchanged. Fresh-process probe reload matched exactly. All 33 HF files verified. Also downloaded and manifest-verified on Apple silicon; terminal chat passed a short harmless READY smoke with steering off. Max projection ratio 0.358% (passes 1%); held-out harmless KL 0.2078 (above 0.1). Full behavioral/capability evaluation is pending. Publisher access required. |
| Kimi Linear 48B A3B Instruct | Separately saved bf16 edit and original on persistent Runpod storage; CUDA chat tested | All 20,493 tensors checked, 6,711 edited. Original hashes unchanged; fresh-process probe reload matched exactly. Held-out harmless KL 0.0756 and max projection ratio 0.636% pass the diagnostic references. Cached chat passed a harmless probe. Broad behavior/capability remains unverified. |
| Qwen3-8B | Separately saved bf16 edit; verified private release `ashwin-sreedhar/qwen3-8b-abliterated-bf16` | All 399 tensor payloads checked, 73 edited tensors, fresh-process probe reload matched exactly; 33 HF files verified. Max projection ratio 1.075% (above 1%); held-out harmless KL 0.1152 (above 0.1). Full behavioral/capability evaluation is pending. |
| Llama 3.1 8B Instruct | Dense CUDA adapter; publisher access now granted, full export queued | Tiny CPU save/reload passed. No completed full-size checkpoint claimed yet. |

The generic `--model` argument dispatches only to explicit, checked architectures. It does not support
arbitrary models automatically. Kimi K3/K3.1 need separate architecture, licensing, hardware and validation work.

## Interpreting the Kimi edit

The full Kimi export succeeded on a 96 GB RTX PRO 6000 Blackwell GPU with batch size one and
expandable CUDA allocator segments. This narrowly tested configuration replaces the earlier assumption
that a 141 GB GPU was required; larger GPUs still provide more room for context and batching.
The saved directory is `models/Kimi-Linear-48B-A3B-Instruct-abliterated` on the persistent GPU volume.
See [the CUDA chat instructions](ABLITERATION.md#chat-with-a-saved-cuda-checkpoint).

A manual test showed an initial refusal followed by a different response after a follow-up. That is one
uncontrolled observation, without a matched base-model conversation. It demonstrates inconsistent refusal
behavior, not universal removal or the functional correctness of generated content. No test prompt,
completion, or generated program from that conversation is included here.

## Testing the saved Gemma checkpoint on Apple silicon

From the repository root, after downloading the private release to the folder shown below:

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
Original weights are retained separately; failed exports stay under uniquely named `.incomplete-*` paths.

The historical Qwen result tables and the original fixed acceptance bar are not silently replaced by the
new export selection rule. All ten attempts are reported in the [aggregate classifier review](abl/semantic_refusal_review.json).
Those responses were already exposed during development. A fresh, independently reviewed evaluation is
required before making a stronger behavioral claim, and non-refusal alone does not establish a correct answer.

## Distribution and licenses

Current HF model releases are private. Access to the repository is not a promise of access to its private
weights. Any future public release needs its own license/access review and a model card matching its evidence.
Gemma and Llama require the upstream publisher's access agreement. Upstream license and notice files are
preserved when present; model cards link to the original model and any applicable terms. Source code in
this repository is MIT, which does not relicense third-party models.

The retained Runpod volume is 250 GB. Its recorded standard rate is approximately $17.50/month; it keeps
billing after a GPU is terminated. GPU jobs run one at a time with bounded lifetimes and a separate GPU
spending ceiling. No GPU is provisioned by the model-export CLI itself.

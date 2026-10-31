# GLM and DeepSeek controlled ablation support

Status: small CPU software fixtures verified; full-size model runs and checkpoints pending.
This page does not claim behavioral refusal removal or capability preservation.

## Targets and source preservation

| Target | Pinned source revision | Weight files | Runtime |
|---|---|---:|---|
| `zai-org/GLM-4.7-Flash` | `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | 62,444,175,504 bytes | Native Transformers GLM MoE/MLA |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | `6e8885a6ff5c1dc5201574c8fd700323f23c25fa` | 16,381,516,796 bytes | Native Transformers Qwen3 dense backbone |

Use a separate CUDA environment with [`requirements/native-cuda.txt`](../requirements/native-cuda.txt).
Do not upgrade the existing Kimi environment, which has a different Transformers pin.
The existing pipeline hashes originals before and after export, uses independent output directories,
checks all tensor payloads, and verifies reload in a fresh process. Packed quantized sources are rejected.

GLM's source has individual expert matrices; the native runtime packs them. The adapter uses checked
views of the live packed parameters and compares every mapped expert to its source values. Export keeps
the original source layout. Native packed fixtures are also covered. Each expert output matrix is
projected on its output axis. Shared experts, dense layers and attention writers are included.
Known next-token-prediction tensors outside the decoder are preserved byte-for-byte and remain untested.
Unexpected decoder output writers fail validation.

DeepSeek-R1-0528-Qwen3-8B adds a DeepSeek reasoning release, not native DeepSeek architecture support.
Its upstream template does not honor `enable_thinking=False`; preserve that behavior and record the
actual suffix. A reasoning model may provide no stable first-token refusal proxy. In that case the
fixed selection rule must stop without a checkpoint, rather than changing thresholds to force a result.
The CUDA client now uses the same formatting method as extraction and reload.

## Calibration and verification policy

Retain the original 128 harmful / 128 harmless training split and fixed validation/KL files until
the expanded calibration earns promotion in a controlled weight-edit comparison. No qualifying
benefit has yet been established. Never reuse a direction across model families.
Full-size runs must use the existing fixed selection rule and stop if no candidate passes.
Software integrity checks and three harmless reload probes do not establish broad capabilities or
universal refusal removal. Demo smoke tests use benign prompts only.

## Distribution

The DeepSeek source supplies its own MIT license. The release helper uses that source-specific text,
retains Apache notices, and does not substitute Kimi's MIT copyright notice.

GLM's pinned [model card](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/7dd20894a642a0aa287e9827cb1a1f7f91386b67/README.md)
declares MIT, but that revision has no standalone license file. The related code repository's Apache
license must not be substituted for the model's declaration. Preserve the card and pinned metadata;
keep a verified research checkpoint on the approved volume while public release packaging is unresolved.

## Local software tests

```sh
python -m unittest tests.test_glm_abliteration tests.test_dense_abliteration tests.test_cloud_release tests.test_cuda_chat
```

The GLM fixtures test packed and upstream expert layouts, live edit/restore behavior, fresh-process
round trips, original-source hashes, unchanged prediction-head payloads, and rejection of extra writers.
They use tiny random models and synthetic harmless strings; they do not download full model weights.

[Abliteration guide](ABLITERATION.md) · [Model status](MODEL_STATUS.md) · [Roadmap](ROADMAP.md) · [README](../README.md)

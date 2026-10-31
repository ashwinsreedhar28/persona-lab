# GLM and DeepSeek controlled ablation support

Status: GLM CPU software fixtures verified; full-size runs remain pending. Native DeepSeek replaces
the canceled Qwen-based target and needs its own adapter/license review. Other model runs are paused
at the user's request. The [GPT-OSS metadata audit](GPT_OSS_RESEARCH.md) is complete.
This page does not claim behavioral refusal removal or capability preservation.

## Targets and source preservation

| Target | Pinned source revision | Weight files | Runtime |
|---|---|---:|---|
| `zai-org/GLM-4.7-Flash` | `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | 62,444,175,504 bytes | Native Transformers GLM MoE/MLA |
| `deepseek-ai/DeepSeek-V2-Lite-Chat` | `85864749cd611b4353ce1decdb286193298f64c7` | 31,413,626,576 bytes | Native DeepSeek V2 MoE/MLA; adapter pending |

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

DeepSeek-V2-Lite-Chat is an older native DeepSeek architecture, rather than a Qwen-backbone
distillation. Its attention, routed/shared expert outputs, runtime/source layouts and normalization
need an independent audit and tiny fixture before a full-size run. No native DeepSeek checkpoint has
been edited or saved. The previous DeepSeek-R1-0528-Qwen3-8B target was canceled and was not run.

## Calibration and verification policy

Retain the original 128 harmful / 128 harmless training split and fixed validation/KL files until
the expanded calibration earns promotion in a controlled weight-edit comparison. No qualifying
benefit has yet been established. Never reuse a direction across model families.
The existing selection rule is not behavioral validation: the GPT-OSS audit found generic analysis
tokens being scored as a refusal proxy. A passing candidate alone is insufficient evidence; stop for
review if that score has no validated interpretation.
Software integrity checks and three harmless reload probes do not establish broad capabilities or
universal refusal removal. Demo smoke tests use benign prompts only.

## Distribution

The former Qwen-based DeepSeek source's MIT bundle does not establish licensing for native
DeepSeek-V2-Lite-Chat. Its custom model license must be reviewed and retained separately before any
redistribution. Code licensing and model-weight licensing must not be conflated.

GLM's pinned [model card](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/7dd20894a642a0aa287e9827cb1a1f7f91386b67/README.md)
declares MIT, but that revision has no standalone license file. The related code repository's Apache
license must not be substituted for the model's declaration. [Hub license guidance](https://huggingface.co/docs/hub/repositories-licenses)
explicitly supports declaring a standard license in model-card metadata. Release packaging therefore
retains Z.ai attribution, the pinned declaration/card hash, and standard MIT terms, without inventing
a missing publisher copyright notice. This source-specific bundle is separate from Kimi and DeepSeek.

## Local software tests

```sh
python -m unittest tests.test_glm_abliteration tests.test_dense_abliteration tests.test_cloud_release tests.test_cuda_chat
```

The GLM fixtures test packed and upstream expert layouts, live edit/restore behavior, fresh-process
round trips, original-source hashes, unchanged prediction-head payloads, and rejection of extra writers.
They use tiny random models and synthetic harmless strings; they do not download full model weights.

[Abliteration guide](ABLITERATION.md) · [Model status](MODEL_STATUS.md) · [Roadmap](ROADMAP.md) · [README](../README.md)

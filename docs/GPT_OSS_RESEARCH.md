# GPT-OSS-20B research preparation

Status: tiny CPU fixtures passed; full-size GPU conversion, editing, storage and demo pending.
Source: `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`.
Use a separate environment with [`requirements/native-cuda.txt`](../requirements/native-cuda.txt).

## Separate source, preparation and edit

The publisher released expert weights in MXFP4. The pipeline still rejects quantized checkpoints for
editing. A new explicit preparation step decodes those values into an independent bf16 checkpoint.
This does not restore precision already lost through quantization, and is not an original
full-precision OpenAI release. The comparison baseline must be that same dequantized representation.

Preparation decodes expert tensors with the pinned Transformers 5.21 conversion, writes independent
shards, checks every saved tensor's bytes/shape/dtype, and records source hashes before and after.
It then runs the publisher's original source through native MXFP4 dequantization and compares three
harmless probes over three greedy steps with a fresh process loading the saved bf16 copy. A sealed
conversion manifest records the original revision, files, environment, conversion and reload checks.

The edit CLI takes the original model identity plus `--prepared-source`; it validates the conversion
seal and original raw-source hashes before and after the edit. It saves to another independent
directory. Existing destinations fail closed. The output includes conversion evidence as provenance
for the unedited prepared baseline; the edited checkpoint has its own manifest and completion seal.

On a suitable GPU, after installing the native requirements and preparing the fixed calibration data:

```sh
python -m abliteration.conversion \
  --model openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --output /workspace/models/GPT-OSS-20B-prepared-bf16

python abliterate.py run \
  --model openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --prepared-source /workspace/models/GPT-OSS-20B-prepared-bf16 \
  --data /workspace/data --batch-size 1 \
  --output /workspace/models/GPT-OSS-20B-abliterated-dequantized-bf16
```

These are the intended entry points, not evidence of a completed full-size run. The fixed selection
rule can reject all candidates, in which case no edited checkpoint is published. The original
calibration split remains the default until expanded calibration earns promotion.

## Expert orientation, inference and chat

Native expert output matrices are `[expert, intermediate, hidden]`; project along the final hidden
axis independently for each expert. Expert output biases are `[expert, hidden]` and need the same
projection. Attention output matrices/biases and embeddings retain their checked orientations.
Routing, gate/up matrices, attention sinks and unrelated tensors are preserved.

Use eager attention, which includes attention sinks, and native eager experts. Extraction, reload and
chat use the upstream Harmony template with low reasoning effort and an explicitly opened ordinary
analysis channel. The default open assistant header would otherwise make the first-token proxy score
fixed protocol markers. This prompt policy is identical for original/edited checks and the demo, and
does not force a final answer or disable reasoning. Both descriptive and legacy Harmony marker names
are handled. The terminal displays only the
assistant's final channel and does not execute tool messages. Its default budget is 4,096 generated
tokens, including reasoning; an incomplete reasoning segment is not shown as a final answer.

```sh
python chat_cuda.py \
  --model /workspace/models/GPT-OSS-20B-abliterated-dequantized-bf16 \
  --verify-files --check-cache --max-tokens 4096
```

This command becomes demo-ready only after full-size export/reload and benign smoke checks pass.
Software integrity, projection diagnostics and three reload probes do not establish broad capability
preservation or universal refusal removal. No raw harmful generations are included in the release.

Official references: [Transformers runtime and memory](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers),
[format and implementation checks](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations).

[Model status](MODEL_STATUS.md) · [GLM/DeepSeek](GLM_DEEPSEEK_RESEARCH.md) · [Abliteration guide](ABLITERATION.md) · [README](../README.md)

# GPT-OSS-20B: saved edit and negative behavioral result

Updated 2026-10-31. The full-size edited checkpoint is saved on
[public Hugging Face](https://huggingface.co/ashwin-sreedhar/GPT-OSS-20B-abliterated-dequantized-bf16) and the retained Runpod volume.
**Refusal removal has not been demonstrated.** Manual chat testing reported continued refusals.
No matched base-model conversation or quantified manual refusal rate was supplied. The GPU was
terminated at the user's request. This metadata audit is complete; further model work is paused.

## What finished

| Check | Result | Scope |
|---|---|---|
| Source preservation | Passed | Raw source hashes unchanged; independent prepared and edited directories |
| Tensor serialization | Passed | All 411 tensors checked; 97 mapped tensors projected; 13 weight shards |
| Local fresh-process reload | Passed | Zero maximum log-probability difference on three benign probes over three steps |
| Public download and fresh reload | Passed | All 56 files verified; zero probe difference; three cached/uncached token IDs matched |
| Harmless terminal smoke | Passed | A short benign READY response; not a capability benchmark |
| Maximum projection ratio | **1.0758114%; failed** | Fixed reference is at most 1% at every measured site; worst site index 22 |
| Harmless first-token KL | 0.0263501 | Mean over 100 held-out prompts; not broad capability preservation |
| Manual behavioral check | **Continued refusals reported** | Uncontrolled observation; no matched baseline or denominator |

The aggregate evidence is in [the run record](../evaluation/results/gpt-oss-20b-20261031.json).
Mechanical suppression, serialization correctness and behavioral efficacy are separate claims.
The original Qwen acceptance bar has not been changed or applied as a complete GPT-OSS benchmark.

## Diagnostic finding: the selection proxy is unvalidated

The saved selection record used token 1844 (`User`) and token 2167 (`We`). These are generic
analysis-opening words, not validated refusal markers. The fixed selector chose tokens that occurred
as the first greedy training token at least eight times and more often for harmful than benign inputs.
Its separation/induction/KL pass therefore establishes only that proxy's criteria.

The inference template explicitly opens the ordinary analysis channel. Its first-token score concerns
the start of reasoning, rather than a final-answer refusal. A saved benign greeting probe also assigned
the two tokens a combined probability of 0.51117. That is direct evidence that the proxy is nonspecific;
it is not a sensitivity/specificity study or a final-answer refusal rate.

The selected activation coordinate was strongly suppressed on validation, while manual refusals
persisted. This shows why projection and proxy scores cannot substitute for behavioral validation.
It does not identify the complete causal explanation of the manual result. No new weight edit or
GPU job was performed for this metadata audit, and the checkpoint remains unchanged.

## Source, preparation and release identity

- Original: `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`.
- Separate bf16 preparation seal: `29e5dd8255ba8f6474da51050e327d569fa722c31587669ca19ce832a09a21ee`.
- Local edited export seal: `197f402a6588a224cfa6dc9fb7fbcf32a87da45874ea3346e1bec881c10f135d`.
- Fully downloaded/reloaded public revision: `de9728c079774d0dbbdb33be960aebf8c8676e18`.
- Corrected public card revision: `21c4cb5612b32be2ea1b9650f87c7a194b05efef`.
- Current distribution seal: `7233b3b4b8e452fd139596df9dc0a9cb7b79b57754766f6b00a7ee7342c5263b`.

The documentation amendment changed only README, manifest and completion seal. All other file hashes,
including every weight file, were checked against the previously verified release and remained unchanged.
The runtime helper revision was `6ad39c2556b2d60654d2303cf56a9af2b188d207`.

The publisher's expert weights were MXFP4. Preparation decoded them into an independent bf16 working
copy; this does not restore precision lost in quantization or constitute an original full-precision
OpenAI release. Native dequantization and a fresh saved-bf16 reload matched on three benign probes
over three greedy steps. The original prepared bf16 copy is the appropriate matched baseline.

The original 128+128 calibration split was used; expanded calibration has not earned promotion.
The release retains source/data/code hashes, direction, selection, measurements, tensor checks and
reload evidence. Existing output directories are never overwritten.

## Runtime and transcript interpretation

Native expert output matrices are `[expert, intermediate, hidden]`; projection uses the final hidden
axis. Expert output biases, attention outputs/biases and embeddings are mapped separately. Router,
gate/up matrices and attention sinks are preserved. Eager attention includes sinks; eager experts
and the native Harmony template use low reasoning effort. The terminal displays only the final
channel and executes no tool messages. Its 4,096-token allowance includes reasoning.

The supplied follow-up transcript contains policy and deployment self-descriptions. Claims about
OpenAI-hosted inference, token-scanning moderation/rewrite services, and only OpenAI being able to edit
weights are unsupported by this local CUDA client. The client does not call those services or send
chat data to a training API. Model self-description cannot establish checkpoint identity. That
transcript itself is separate from the user's reported harmful-request refusals; it supplies no
matched behavioral score. Raw conversations are excluded from this repository.

## Next diagnostic work

Audit checkpoint identity, tokenizer/template parity, final-channel parsing, stopping conditions and
normal capability before drawing further conclusions. Any new paired study must freeze source,
precision, prompt IDs, formatting, generation budget and scoring before execution, and distinguish
refusal, safe redirection, mixed responses and truncation. Report uncertainty and benign utility.
Do not report a changed first-token proxy as successful refusal removal. The diagnostic does not
establish an improved checkpoint. No additional paid resource was started for this audit.

Use the [native environment](../requirements/native-cuda.txt) for any future compatible GPU reload.
The former SSH endpoint is terminated; there is no current live demo. Storage and integrity details
are in [MODEL_STORAGE.md](MODEL_STORAGE.md).

Official implementation references: [Transformers runtime and memory](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers),
[format verification](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations).

[Model status](MODEL_STATUS.md) · [Roadmap](ROADMAP.md) · [README](../README.md)

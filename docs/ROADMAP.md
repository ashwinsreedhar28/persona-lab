# Proposed next work and cloud budget — 2026-10-31

These are proposed priorities, not completed model support or authorization for paid runs. The
[Qwen expanded-calibration dataset and diagnostic](CALIBRATION_V2.md) are complete locally at $0
cloud cost. No new-family GPU, network volume, subscription or HF repository was created.

## Priority by expected impact

1. **Finish the controlled Qwen calibration experiment.** Match current 128-row and expanded 256-row
   edits using the same selection/edit policy and a frozen independent confirmation set. This is the
   most direct route to checking whether the expanded corpus improves a checkpoint.
2. **Improve independent evaluation.** Use broader benign capability tasks and a blinded human audit
   of classifier disagreements; score full answers and report mixed refusal/compliance. StrongREJECT
   documents why apparent non-refusal can overstate answer effectiveness. Keep new tests separate
   from calibration and source selection. This has the highest confidence of improving conclusions.
3. **Add the three requested families with matched validation.** Start one model at a time after
   calibration is validated and a spending cap is approved. Shared data do not imply shared vectors.
4. **Make verified smaller chat exports.** Convert completed bf16 edits to supported inference formats
   and check the quality impact. This offers practical local-use value independently of a new edit.
5. **Explore richer editing methods.** Multiple directions, layer-specific choices or stronger selection
   rules need a new predeclared study and capability controls. Potential upside is uncertain.
6. **Expand family coverage further.** Mistral/Ministral and OLMo are the strongest optional additions
   below; completing the current batch has higher priority than collecting more releases.

## Candidate models

| Proposed target | Purpose and readiness | Approximate edited bf16 weights |
|---|---|---:|
| `openai/gpt-oss-20b` | New MoE/Harmony coverage; Apache 2.0. Needs a new checked adapter and separate dequantized bf16 working copy. | ~42 GB |
| `zai-org/GLM-4.7-Flash` | Native GLM 30B-A3B MoE/MLA coverage; MIT. Needs new writer mapping, dependency environment and validation. | ~62 GB |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | Economical DeepSeek reasoning release; MIT with upstream notices retained. Uses a Qwen3 backbone, so it adds behavioral coverage rather than a new native DeepSeek architecture. Existing Qwen3 mapping is a starting point, not a full-run validation. | ~16 GB |

The DeepSeek alternative for native architecture coverage is `deepseek-ai/DeepSeek-V2-Lite-Chat`
(16B, about 32 GB bf16). It is an older model and requires separate adapter and model-license review.
Choose one DeepSeek target initially. Full flagship GPT-OSS/GLM/DeepSeek sizes are outside this first
budget. MoE active-parameter counts reduce computation, while storage must still hold all experts.

GPT-OSS publishes MoE weights in MXFP4. Our current policy rejects editing packed quantized weights.
A new implementation must preserve the original release, dequantize to a separate bf16 source,
compare against that same dequantized baseline, edit in bf16/fp32 and record conversion provenance.
Dequantization does not recover precision lost in the released representation. Harmony formatting,
reasoning settings, expert writers/biases, harmless probes and cached/uncached behavior require checks.

Official sources: [GPT-OSS model](https://developers.openai.com/api/docs/models/gpt-oss-20b),
[precision and inference](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers),
[format verification](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations),
[GLM card](https://huggingface.co/zai-org/GLM-4.7-Flash),
[DeepSeek reasoning card](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B),
[native DeepSeek alternative](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat).

## Proposed budget and HF storage

Runpod's published Secure Cloud rates and read-only catalog check on 2026-10-31 show an A100 80 GB
at **$1.79/hour**, or the RTX PRO 6000 Server 96 GB at **$2.49/hour**. The latter gives more headroom
for the new adapters. Stock and the actual deployment quote must be rechecked before starting.

An **estimated planning allowance of 6–12 GPU hours across the three targets** is $10.74–21.48 on
A100 or $14.94–29.88 on the 96 GB card. These runtimes are estimates, not measured runs or a guarantee;
adapter debugging, extended benchmarks and failed jobs can require more. Approve a cap and start one
family before committing the rest. A **$35 initial Runpod cap** is a proposal, not authorization. An optional HF subscription is separate.

A **300 GB standard temporary network volume** is **$21/month**, approximately **$0.70/day** or
**$0.35 for 12 hours**, using a 30-day-month estimate. A 40 GB container disk adds about $0.07 over
12 hours at the published $0.10/GB/month rate. Network storage keeps billing after GPU shutdown
until deleted. Process/upload models sequentially and recheck peak disk needs if the native DeepSeek
alternative is chosen. Source/dequantized/edit copies plus software must fit before starting.

The three proposed edited checkpoints add roughly **120 GB** to the existing ~148 GB public releases.
HF public storage is **best-effort** on a free account, not guaranteed unlimited storage. If an upgrade
is needed, HF currently lists PRO at **$9/month**, with up to 10 TB public and 1 TB private storage.
No subscription was purchased. Use HF as the long-term destination; retaining an idle GPU is not a
substitute for HF capacity. Keep the temporary volume until all uploaded files have been verified
against hashes and a fresh download/reload check passes, then remove temporary cloud storage under
the approved run/cleanup plan. Retain upstream licenses and modification notices.

Sources: [Runpod pricing](https://www.runpod.io/pricing),
[HF limits](https://huggingface.co/docs/hub/storage-limits), [HF PRO](https://huggingface.co/pro).

## Optional later families

- [Ministral 3 8B Instruct BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16):
  Apache 2.0, independent model lineage and practical size. A text-only first study would not validate
  its vision behavior. Use the official bf16 source and preserve the original.
- [OLMo 3 7B Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct): Apache 2.0 and useful training
  transparency for research. Establish its baseline behavior first to determine whether an edit adds value.

[Calibration evidence](CALIBRATION_V2.md) · [Current supported models](MODEL_STATUS.md) · [README](../README.md)

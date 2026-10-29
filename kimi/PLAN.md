# Abliteration, Phase B: Kimi-Linear-48B-A3B-Instruct (1x H200, cap $20)

Applies the Phase A method to a hybrid KDA/MLA MoE architecture. Main question beyond Phase A: what happens to the refusal direction if the shared expert's down_proj is NOT edited (the "noshared" variant).

## Model facts (from the HF config, verify in B0)
27 layers: 20 KDA + 7 MLA (full_attn_layers 4,8,12,16,20,24,27, 1-indexed). hidden 2304. Layer 0 is a dense MLP (first_k_dense_replace=1); layers 1-26 are MoE with 256 routed experts (8 per token, moe_intermediate 1024, full width, not latent) plus 1 shared expert. tie_word_embeddings=false, so lm_head is a reader and stays untouched. ~49B params, ~98 GB bf16. Needs trust_remote_code and fla-core >= 0.4.0. vLLM recipe: 0.11.2 recommended, avoid 0.12.0.

Expected residual writers: embed_tokens; 27 attention output projections (KDA and MLA o_proj); layer 0 mlp.down_proj; 26 x 256 routed expert down_proj; 26 shared expert down_proj; plus any biases. About 6,711 matrices.

## Variants
- base
- full: every writer orthogonalized against r-hat
- noshared: every writer except the 26 shared-expert down_projs

## Steps
B0. Create the pod per kimi/CLAUDE.md, then check it (first 15 min): nvidia-smi, disk, `runpodctl version`, `echo $RUNPOD_POD_ID`. Start the runaway guard: `mkdir -p /workspace/logs && nohup bash -c 'sleep 25200; runpodctl stop pod $RUNPOD_POD_ID' > /workspace/logs/watchdog.log 2>&1 &` (7 hours). After each step, copy that step's small outputs back to kimi/ on the Mac so nothing is lost if the pod stops. pip install fla-core, transformers, accelerate, vllm (try the current release; if Kimi Linear fails to load, pin vllm==0.11.2). hf download the model with hf_transfer. Copy abl/data/*.json from the Mac. Load in HF transformers (bf16, one GPU), list every module whose output is d_model=2304 and save writers.json; confirm the count and that lm_head is untied. Record the chat template's post-instruction tokens.

B1. Directions (HF). Non-system default chat template. Residual activations at each post-instruction token, every layer, 128 harmful + 128 harmless train prompts. r(l,pos) = mean difference. Record the base model's first-token distribution on harmful train to pick the refusal tokens for the refusal score.

B2. Selection (HF, val split only). Candidates: first 80% of layers x post-instruction positions. Arditi criteria: bypass score with the direction ablated at every layer (hooks), induce score adding it at its layer must be > 0, first-token KL on harmless val <= 0.1. Pick the lowest bypass score. One pick, pre-registered: this is THE direction. Save r_hat.npy and selection.json.

B3. In-memory checks (HF). Edit weights in place on the GPU: first all writers except shared experts (= noshared), measure, then the shared experts (= full), measure. For each of base / noshared / full: per-layer mean |h . r-hat| on val prompts (projection.json, the main Phase B finding), and first-token KL vs base on the 100 held-out harmless prompts. Free the HF model.

B4. Generation evals (vLLM, greedy, identical settings per variant, max-model-len 8192). Preferred path, verify with a smoke test first: load base once, evaluate, then apply the noshared edit in place with LLM.apply_model / collective_rpc, evaluate, then edit the shared experts, evaluate. vLLM stores routed experts fused, so project the fused w2 tensor per expert along the hidden dim. Smoke test: after the in-place edit, a known prompt's first-token logprobs must match the HF in-memory edit closely. If the in-place path fails, save each edited checkpoint to /workspace one at a time (edit shard by shard on CPU), evaluate, delete, then do the next.
   Sets: JBB harmful 100 (max 256 tokens), JBB benign 100, GSM8K 1,319 (same zero-shot prompt and last-number extraction as Phase A, max 512 tokens). Refusal markers: Phase A's list.

B5. Judge. Free Kimi, load Qwen3Guard-Gen-4B in vLLM on the pod, judge JBB harmful completions for each variant.

B6. Bring back to kimi/ on the Mac: writers.json, selection.json, r_hat.npy, projection.json, metrics.json (all numbers), every script, logs (no completions). Verify sizes/hashes, then terminate the pod.

B7. Write kimi/RESULTS.md: one table with rows base / full / noshared and columns JBB refusals, unsafe rate (Qwen3Guard), JBB benign refusals, KL, GSM8K; the per-layer projection ratios for full vs noshared; elapsed pod time x price; anything that deviated from this plan.

## Expected cost
~2.5-3.5 h of H200 time; Secure Cloud H200 was $5.30/h on Oct 29, so ~$13-19. No hard cap; the watchdog stops the pod at 7 h (~$37) as a runaway guard. Record actual hours x price in RESULTS.md.

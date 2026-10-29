# Phase B results: Kimi-Linear-48B-A3B-Instruct, refusal-direction measurements

## Status: done (C0-C5)

The measurement-only plan ran end to end on one H200 pod (`dkyxi1edbksmwm`, US-NC-1) in 19 minutes. Only forward passes ran: no text generation, no saved weights, no bypass score. Direction tensors and activations never left the pod. kimi/out/ holds 7 aggregate JSON files (writers, directions_stats, selection, attribution, projection, kl, summary). A guard on the pod and again on the Mac checked that no file contains a d_model-length array. kimi/logs/ holds the pod logs. No safety check fired.

**Headline:** removing r-hat from all 6,711 writers ("full") leaves at most 0.58% of the base projection at any layer. Leaving the 26 shared-expert down_projs unedited ("noshared") leaves 22-70% from layer 2 on (about 30% at most layers). In the base model, though, the shared experts write almost none of the harmful-minus-harmless difference along r-hat: about 4% at the selected position, against about 99% from routed experts. So what noshared leaves behind is mostly a component of r-hat common to harmful and harmless prompts. Whether that matters for behavior was not measured.

## C0. Writer inventory and method check
- 27 layers: 20 KDA + 7 MLA (0-indexed MLA layers 3, 7, 11, 15, 19, 23, 26). Layer 0 is a dense MLP; layers 1-26 are MoE (256 routed experts + 1 shared). 49.12 B parameters, 98.9 GB peak GPU memory in bf16.
- **6,711 residual writers**, and the generic scan (every Linear with out_features 2304, every Embedding with dim 2304) matches the structural list exactly: embed_tokens 1 (163840x2304); attention o_proj 27 (2304x4096, KDA and MLA alike); layer-0 down_proj 1 (2304x9216); routed expert w2 6,656 (2304x1024); shared expert down_proj 26 (2304x1024). No writer has a bias. tie_word_embeddings = false and lm_head has its own storage, so it is a reader and stays untouched.
- Post-instruction template tokens (default template, no system message): `<|im_end|>`, `<|im_assistant|>`, `assistant`, `<|im_middle|>` (positions -4..-1).
- **Method check** (random unit vector, numpy seed 0, harmless_val[:8]): first-token KL between the run-time ablation hook and the in-memory weight edit is 0.0058 mean (0.038 max), and the top-1 token agrees on 8/8 prompts. That is higher than Phase A's ~0.001 and the same size as the random edit's own effect on base (KL 0.0017). I read it as the bf16 rounding floor of re-rounding 6,711 edited matrices, not a method error. With the real direction (C4), hook vs edit is 0.0029 mean (0.018 max) on harmless val, and the full edit leaves at most 0.58% of the projection.

## C1. Direction geometry (vs Phase A's Qwen3.5-4B)
- r(l,pos) = mean(harmful train) - mean(harmless train) of resid_pre, 128 + 128 prompts. At layer 0 every template position is the same token embedding for all prompts, so r = 0 there.
- Relative norm ||r|| / mean||h|| at the selected position (-4, `<|im_end|>`) rises to 0.60 by layer 9 and peaks at 0.75 at layer 14, then falls to 0.38-0.50 in layers 19-26. At -1 (`<|im_middle|>`) it climbs more slowly and stays high: 0.39 at L9, 0.61 at L11, 0.79 at L16-17, 0.63 at L26. Full per-layer, per-position tables are in directions_stats.json.
- Cosines at -4: r(L11) has cos 0.81 / 0.88 with L9 / L10, 0.90 / 0.80 with L12 / L13, falling to about 0.4 by L20. Adjacent-layer cosines are 0.48-0.76 in layers 1-7 and 0.84-0.98 from L8 on, so the direction settles around layer 8. Across positions at the same layer, -4 and -1 agree only moderately (cos 0.54-0.71 in L7-L15, about 0.4 late).
- **Side by side with Phase A:**

| | Qwen3.5-4B (Phase A) | Kimi-Linear-48B-A3B |
|---|---|---|
| layers / d_model | 32 / 2560 | 27 / 2304 |
| selected | L13, pos -7 (`<|im_start|>`) | L11, pos -4 (`<|im_end|>`) |
| selected relative depth | 13/31 = 0.42 | 11/26 = 0.42 |
| ||r|| peak (last pos) | L31 (final layer) | L26 at -1; L23 at -4 (raw norm) |
| ||r|| at 50% depth, fraction of max | 0.16 (last pos), 0.28 (selected pos) | 0.23 (-1), 0.49 (-4) |

  Both picks sit at the same relative depth. Kimi's direction reaches a large share of its final size earlier in depth than Qwen's. Raw norms aren't comparable across models, and Phase A did not record residual norms, so only the depth shape is compared.
- Refusal token (rule fixed in code before C2: argmax first tokens on harmful train with count >= 8 and more frequent than on harmless train): `I` only. It is the first token for 121/128 harmful-train prompts (94.5%) and 3/128 harmless-train prompts.

## C2. Selection (val split only)
Rule (PLAN.md): among the first 80% of layers (L0-L20) x 4 positions = 84 candidates, take the top 20 by val AUROC of h.r-hat. For those, induce = mean increase in `I` log-odds at the first output position on harmless val when ||r||·r-hat is added at the candidate's layer, and KL = first-token KL(base || ablated) on harmless val. Pick the highest induce with induce > 0, AUROC >= 0.9 and KL <= 0.1. **This differs from Arditi et al., who pick the lowest bypass score.** No bypass score was computed.

- AUROC: 76/84 candidates reach >= 0.9, and **30 reach exactly 1.0** (from L5 up). The plan didn't say how to break ties. The code, written before the run, sorts by (-AUROC, layer, position), so the top 20 are the earliest 20 of the 30 perfect separators (L5-L16), and 10 perfect separators in L16-L20 were not considered.
- Baseline `I` log-odds on harmless val: -9.79.
- 6 of the 20 pass:

| candidate | AUROC | induce | KL | passes |
|---|---|---|---|---|
| L05 pos-3 | 1.000 | +0.12 | 0.012 | yes |
| L09 pos-4 | 1.000 | +12.07 | 0.119 | no (KL) |
| L10 pos-4 | 1.000 | +12.10 | 0.152 | no (KL) |
| L10 pos-2 | 1.000 | +10.64 | 0.038 | yes |
| **L11 pos-4** | **1.000** | **+11.64** | **0.062** | **selected** |
| L11 pos-2 | 1.000 | +11.34 | 0.032 | yes |
| L15 pos-4 | 1.000 | +9.05 | 0.054 | yes |
| L15 pos-1 | 1.000 | +11.14 | 0.098 | yes |

  The other 12 of the top 20 (L07, L12-L16) fail on KL (0.11-0.37). **Pick: L11, pos -4 (`<|im_end|>`), ||r|| = 16.16.** That's one pick; there was no other attempt.

## C3. Attribution (base model, before any edit)
Each writer's contribution to h.r-hat, harmful val minus harmless val, mean per token. Columns are at the selected position (-4); "routed" is the sum over experts, and "top1 / top10" is the share of the routed experts' total |difference| held by the 1 and 10 largest experts in that layer. The decomposition reconstructs resid_pre to within 0.005 at this position (0.010 over all tokens), and the per-expert sum matches the model's own routed output to within 0.0003.

| layer | type | attention | MLP / routed | shared | routed top1 / top10 |
|---|---|---|---|---|---|
| 0 | KDA | +0.00 | +0.00 | +0.00 | dense |
| 1 | KDA | +0.01 | +0.02 | +0.00 | 0.31 / 0.79 |
| 2 | KDA | +0.04 | +0.03 | +0.02 | 0.20 / 0.84 |
| 3 | MLA | +0.03 | +0.06 | +0.02 | 0.21 / 0.81 |
| 4 | KDA | +0.11 | +0.12 | +0.02 | 0.20 / 0.79 |
| 5 | KDA | +0.44 | +0.39 | +0.05 | 0.27 / 0.83 |
| 6 | KDA | +0.46 | +0.52 | -0.01 | 0.21 / 0.72 |
| 7 | MLA | +0.11 | +1.95 | -0.02 | 0.76 / 0.95 |
| 8 | KDA | +0.50 | +2.04 | -0.02 | 0.28 / 0.89 |
| 9 | KDA | +0.49 | +1.81 | -0.26 | 0.36 / 0.85 |
| 10 | KDA | +0.80 | +2.60 | -0.29 | 0.18 / 0.86 |
| 11 | MLA | -0.01 | +1.60 | -0.29 | 0.45 / 0.78 |
| 12 | KDA | -0.00 | +1.88 | -0.45 | 0.32 / 0.89 |
| 13 | KDA | +0.23 | +1.16 | -0.04 | 0.28 / 0.84 |
| 14 | KDA | -1.13 | +0.22 | -0.01 | 0.17 / 0.62 |
| 15 | MLA | +0.49 | +0.46 | +0.37 | 0.36 / 0.76 |
| 16 | KDA | -0.55 | +0.28 | -0.01 | 0.24 / 0.83 |
| 17 | KDA | -1.35 | +0.19 | +0.05 | 0.18 / 0.74 |
| 18 | KDA | -0.37 | +0.09 | +0.16 | 0.16 / 0.67 |
| 19 | MLA | -0.05 | +0.03 | +0.03 | 0.20 / 0.79 |
| 20 | KDA | -0.28 | -0.19 | +0.17 | 0.17 / 0.76 |
| 21 | KDA | +0.38 | -0.25 | -0.03 | 0.17 / 0.71 |
| 22 | KDA | +0.02 | -0.35 | +0.10 | 0.59 / 0.91 |
| 23 | MLA | -0.37 | -0.08 | -0.01 | 0.10 / 0.64 |
| 24 | KDA | -0.03 | +0.45 | +0.27 | 0.37 / 0.78 |
| 25 | KDA | -0.53 | +0.32 | +0.38 | 0.45 / 0.81 |
| 26 | MLA | +0.07 | -0.20 | +0.45 | 0.18 / 0.74 |

- **Totals, selected position:** the final-layer harmful-minus-harmless gap along r-hat is 15.3. Routed experts contribute +15.1, attention -0.5 net (KDA -0.75, MLA +0.27), shared experts +0.65, and the embedding and dense layer about 0. Up to the selected layer (writers in L0-L10): routed / dense +9.5, attention +3.0, shared -0.5.
- **Over all real tokens:** the gap is 8.0, with routed +5.0, attention +2.4 (mostly KDA) and shared +0.5.
- **Shared experts:** at the selected position they add a little, and in L9-L12 they slightly oppose the direction (-0.26 to -0.45 per layer). Their signed contribution along r-hat is similar for harmful and harmless prompts (-1.5 vs -2.0 summed over layers, all tokens), so they mostly write a component common to both sets, not the separating one.
- **Routed-expert concentration:** in each layer the top 10 of 256 experts hold 62-95% of the routed |difference| (mean 79%), and the single top expert holds 10-76% (mean 28%). It is most concentrated at L7 (expert 52: 76%) and L22 (expert 160: 59%). Expert 24 is the top expert in L6, L11 and L13. Expert ids are per-layer indices, not shared identities.

## C4. In-memory edits: projection ratios and KL
Per-layer mean |h.r-hat| over every real token of the 64 val prompts (32 harmful + 32 harmless). Base and ratio to base. Edits: noshared = embed 1 + o_proj 27 + dense down_proj 1 + expert w2 6,656; full = noshared + shared down_proj 26.

| resid_pre layer | type | base | noshared / base | full / base |
|---|---|---|---|---|
| 0 | KDA | 0.017 | 0.4% | 0.41% |
| 1 | KDA | 0.022 | 0.6% | 0.58% |
| 2 | KDA | 0.057 | 21.8% | 0.37% |
| 3 | MLA | 0.138 | 38.0% | 0.25% |
| 4 | KDA | 0.152 | 69.5% | 0.31% |
| 5 | KDA | 0.238 | 57.2% | 0.31% |
| 6 | KDA | 0.437 | 38.6% | 0.21% |
| 7 | MLA | 0.860 | 50.4% | 0.38% |
| 8 | KDA | 1.306 | 41.5% | 0.33% |
| 9 | KDA | 1.682 | 38.2% | 0.25% |
| 10 | KDA | 2.072 | 35.1% | 0.27% |
| 11 | MLA | 2.445 | 31.7% | 0.25% |
| 12 | KDA | 2.596 | 32.3% | 0.30% |
| 13 | KDA | 2.807 | 32.5% | 0.28% |
| 14 | KDA | 3.063 | 30.6% | 0.27% |
| 15 | MLA | 3.000 | 29.8% | 0.28% |
| 16 | KDA | 3.371 | 29.3% | 0.23% |
| 17 | KDA | 3.494 | 30.4% | 0.26% |
| 18 | KDA | 3.571 | 31.9% | 0.31% |
| 19 | MLA | 3.823 | 27.2% | 0.36% |
| 20 | KDA | 3.997 | 32.4% | 0.31% |
| 21 | KDA | 4.190 | 33.8% | 0.34% |
| 22 | KDA | 4.292 | 35.1% | 0.37% |
| 23 | MLA | 5.139 | 30.1% | 0.33% |
| 24 | KDA | 4.452 | 31.7% | 0.44% |
| 25 | KDA | 4.684 | 31.2% | 0.43% |
| 26 | MLA | 5.081 | 25.3% | 0.49% |
| 27 | out | 6.068 | 39.1% | 0.50% |

- **Full:** max 0.58% (0.64% on harmful val, 0.54% on harmless val). Phase A's Qwen edit left 0.79%.
- **Noshared:** 22-70% in L2-L5, then 25-39% from L6 to the output. Max 69.5% at L4, 31.7% at the selected layer, 39.1% at the output.
- **First-token KL vs base on harmless_kl (100 held-out harmless prompts):** noshared 0.070 mean (median 0.031, max 0.96); full 0.079 mean (median 0.032, max 0.98). Both are under 0.1 on average, with one or two prompts near 1.
- **Equivalence**, full weight edit vs base + run-time hooks, harmless val: KL 0.0029 mean, 0.018 max.

## Limits
- **Mechanical removal only.** Behavior was not measured: no generation, no refusal rates, no judge. In Phase A, 0.79% of the direction left still meant 10/100 refusals, so the 0.58% here says nothing about how the edited model would act.
- The noshared result is about the magnitude of h.r-hat. C3 suggests most of what remains is the component common to harmful and harmless prompts. In the noshared run, the shared experts also see an edited residual, so their contributions aren't exactly the base ones.
- Small val sets (32 + 32). AUROC saturates at 1.0 for 30 candidates, so the top-20 cut depended on an earliest-layer tie-break the plan didn't specify.
- Attribution uses signed per-token means, which can cancel across tokens. Per-expert shares use the harmful-minus-harmless mean.
- bf16 floor: hook vs edit KL is about 0.003-0.006 even for a random vector.
- Refusal-token log-odds use the single token `I`. The induce score is a first-token, logit-only proxy.

## Deviations from PLAN.md / kimi/CLAUDE.md
- **GPU and datacenter:** earlier pods never became reachable, so the pod spec was changed to pin a different datacenter and to allow a B200. kimi/scripts/watch_capacity.sh created this pod in US-NC-1 when H200 stock appeared. /workspace there is a network-backed pod volume (mfs), not local disk; no separate network volume was created.
- **transformers:** 4.57.1 (Kimi's config version) and 4.57.6 both crash in `@auto_docstring` on the remote code's `X | None` hints. common.load_hf swaps that docstring-only decorator for a pass-through. Ran on transformers 4.57.6, torch 2.13.0+cu129.
- **fla-core:** pinned to 0.4.0, the only release whose `fused_kda_gate(g, A, head_k_dim, g_bias)` matches the remote code (0.4.1+ and 0.6.0 changed it).
- **Attention backend:** the model forces flash_attention_2, which isn't installed. The 7 MLA layers ran on PyTorch SDPA (same math); KDA layers are unaffected.
- **C0 failures before success:** auto_docstring x2 (the second was an early restart before the fix had installed), `g_bias` x2 (fla 0.6.0, then 0.4.2 before 0.4.0 was found), flash-attn x1. No single cause hit 3.
- **Runaway guard:** the in-pod watchdog started, but in-pod `runpodctl get pod` returned Unauthorized, so a Mac-side REST stop at 3 h was added as a backup. `runpodctl remove pod` from inside worked at the end, and the Mac guard was cancelled.

## Pod time and cost
| pod | datacenter | created (UTC) | terminated (UTC) | hours | price | spend | ran |
|---|---|---|---|---|---|---|---|
| 7vq6rquv9uchvw | US-NC-1 | 13:24:01 | 13:27:51 | 0.064 | $5.29/h | $0.34 | nothing (attempt 1) |
| nh68mfc25kvlii | EUR-IS-4 | 18:01:00 | 18:19:40 | 0.311 | $5.29/h | $1.65 | nothing (attempt 2) |
| ipc50ostzggf56 | EUR-IS-4 | 18:42:22 | 19:05:42 | 0.389 | $5.29/h | $2.06 | nothing, never reachable (attempt 3) |
| **dkyxi1edbksmwm** | **US-NC-1** | **19:25:59** | **19:45:03** | **0.318** | **$5.29/h** | **$1.68** | **C0-C4** |
| **total** | | | | **1.082** | | **$5.72** | |

This run's timeline (pod clock): reachable 19:26:32; setup + 92 GB download done 19:31:37; C0 passed 19:40:54 after the fixes above; C1 31 s, C2 19 s, C3 3 s, C4 15 s; guard OK 19:42:32; bring-back verified (14 files, sha256 and sizes; scripts match) 19:43-19:44. Terminated from inside:

```
$ runpodctl remove pod dkyxi1edbksmwm
pod "dkyxi1edbksmwm" removed
```
then `GET /v1/pods/dkyxi1edbksmwm -> 404`; no persona-lab-kimi pods remain.

## Earlier attempts
None of the earlier attempts ran anything on a GPU. Attempt 1 followed an earlier version of this plan (abliteration with generation-based evals) and was stopped by a safety check right after the pod was created; the plan was then rewritten as measurement-only. Attempts 2 and 3 got pods in EUR-IS-4 that never received a public IP within about 20 minutes and were deleted. Attempt 4's create calls were refused for lack of H200 capacity. Their pod time is included in the cost table above.

## Notes
- The Mac has no `runpodctl`. Pods were created through the REST API with `$RUNPOD_API_KEY`, which was never printed.
- No git commands were run. Nothing outside kimi/ was edited.

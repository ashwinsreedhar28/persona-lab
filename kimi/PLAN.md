# Phase B: refusal-direction measurements on Kimi-Linear-48B-A3B-Instruct (1x H200)

Question: does persona-lab's Phase A method (difference-of-means refusal direction, then orthogonalize every residual-stream writer) carry over to a hybrid KDA/MLA MoE, and where does the refusal direction live in that architecture? In particular, what happens to it when the shared experts' down_proj is left unedited ("noshared") versus editing every writer ("full")?

Deliverable: a measurement report (kimi/RESULTS.md plus aggregate JSON in kimi/out/). Not a model. Nothing here produces, saves or runs an edited model for text output.

## Boundaries (what this plan does not do)
- No text generation of any kind: forward passes only. No vLLM, no chat, no judge model, no JBB or GSM8K.
- No edited weights saved anywhere (pod or Mac). Edits exist only in GPU memory while being measured.
- No bypass score (refusal odds on harmful prompts with the direction removed). Selection uses separation, induce and KL instead.
- No direction tensors leave the pod. Only aggregate numbers (norms, cosines, AUROCs, ratios, KL) come back.

## Model facts (verify in C0)
27 layers: 20 KDA + 7 MLA (full_attn_layers 4,8,12,16,20,24,27, 1-indexed). hidden 2304. Layer 0 is a dense MLP (first_k_dense_replace=1); layers 1-26 are MoE: 256 routed experts (8 per token, moe_intermediate 1024, full width) plus 1 shared expert. tie_word_embeddings=false (lm_head is a reader, untouched). ~49B params, ~98 GB bf16. Needs trust_remote_code and fla-core >= 0.4.0.

Expected residual writers (~6,711 matrices): embed_tokens; 27 attention output projections (KDA and MLA); layer 0 mlp.down_proj; 26 x 256 routed expert down_proj; 26 shared expert down_proj; plus any biases.

## Data
Prompts only, from abl/data (fixed in Phase A): harmful_train/val, harmless_train/val (128/32 each), harmless_kl (100). Prompts are model inputs; no responses are ever produced.

## Steps
C0. Pod and inventory. Create the pod (kimi/CLAUDE.md), set up HF transformers + fla-core only, download the model, load bf16 on one GPU. Writer inventory -> writers.json (generic scan must match the structural list). Record the chat template's post-instruction tokens. Method check with a random unit vector: the in-memory weight edit must match a run-time ablation hook on 8 harmless prompts (first-token KL ~ 0), before any refusal direction is involved.

C1. Directions and geometry. resid_pre at every layer and post-instruction position for 128 harmful + 128 harmless train prompts; r(l,pos) = mean difference. Record per-layer norms and the layer-by-layer cosine matrix at the chosen position (directions_stats.json). Compare depth profile with Phase A's Qwen numbers (abl/out/directions_stats.json on the Mac).

C2. Selection (val split only, rule fixed before looking). Candidates: first 80% of layers x post-instruction positions.
   1. Separation: AUROC of h.r-hat between harmful and harmless val at the candidate's layer. Keep the top 20 by AUROC.
   2. For those 20: induce score = increase in refusal-token log-odds at the first output position when alpha*r-hat is added at the candidate's layer on harmless val (logits only; refusal tokens = the base model's most common first tokens on harmful train, from logits); KL = first-token KL on harmless val with r-hat ablated at every layer.
   3. Pick the highest induce score among candidates with induce > 0, AUROC >= 0.9 and KL <= 0.1. One pick. Write selection.json. State in RESULTS.md that this differs from Arditi's bypass-based selection.

C3. Attribution. For harmful and harmless val, decompose each layer's h.r-hat into contributions from: embedding, attention output (KDA or MLA), dense MLP (layer 0), routed experts (sum, plus the top 10 experts by contribution with their share), shared expert. Write attribution.json. This answers whether the shared expert writes the direction before anything is edited.

C4. Edits in memory and measurements.
   - Base: per-layer mean |h.r-hat| on val (harmful + harmless); first-token log-probs on harmless_kl.
   - noshared: orthogonalize every writer except the 26 shared-expert down_projs (W <- W - r r^T W; E <- E - E r r^T; biases projected). Measure the same.
   - full: also orthogonalize the shared experts. Measure the same.
   - Equivalence: the full edit vs base + run-time ablation hooks, first-token KL on val (Phase A got ~0.001).
   Write projection.json (per-layer ratio vs base for noshared and full) and kl.json (harmless_kl KL for noshared and full).

C5. Bring back kimi/out/*.json (aggregates only), logs and scripts; verify sizes and hashes; terminate the pod. Write kimi/RESULTS.md:
   - writer inventory and the method check
   - direction depth profile and cosine summary, side by side with Phase A's Qwen
   - selection table (top candidates, chosen one, the deviation from Arditi)
   - attribution by component type per layer, and how concentrated the routed-expert share is
   - projection ratios per layer, full vs noshared; KL for each
   - limits: mechanical removal only; behavior was not measured (Phase A: 0.79% of the direction left and still 10/100 refusals)
   - pod time x price

## Expected cost
~1-1.5 h of H200 time at $5.29/h (Oct 29 Secure price): ~$5-8. Watchdog stops the pod at 3 h.

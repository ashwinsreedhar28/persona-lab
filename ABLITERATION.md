# Reusable, separately saved weight edits

Run this from the repository's existing Python environment:

```sh
python abliterate.py run --model Qwen/Qwen3.5-4B
```

It extracts and selects a direction on the fixed `abl/data/` splits, then writes a new timestamped
folder under `models/`. It never edits or deletes the original snapshot, an existing destination,
or the previous `models/Qwen3.5-4B-abliterated` model. Originals are hashed before and after the run.
The new folder has independent weight files; no hard links to the original are created.

For a first checkout without `abl/data/`, build the fixed splits once in the existing local
MLX environment with `python abl/a1_data.py`. The builder refuses to overwrite existing splits.
Its historical upstream dataset references are not immutable: newly fetched splits may differ from
the archived research splits. Preserve those files and their recorded hashes for reproduction;
copy the same five train/validation/KL files to the CUDA host rather than rebuilding them there.
Raw prompts and generation outputs stay outside Git.

Choose an output explicitly when needed:

```sh
python abliterate.py run --model Qwen/Qwen3.5-4B --output models/Qwen3.5-4B-abliterated-new
python abliterate.py verify models/Qwen3.5-4B-abliterated-new
python evil_chat_mlx.py --model models/Qwen3.5-4B-abliterated-new --dose 0
```

Interactive terminal and web chat now allow 4,096 generated tokens per answer by default. The terminal
limit can be changed with `--max_tokens`. Thinking tokens share this allowance. Benchmark limits remain
the fixed values in the original experiment.

## Supported models and limits

- Qwen3.5 dense `qwen3_5` / `qwen3_5_text` checkpoints: MLX on Apple silicon. Shapes and layer types
  come from the model's configuration; writers must match its actual tensor inventory. The tested full
  checkpoint is Qwen3.5-4B. Another size still needs sufficient RAM and its own validation.
- `moonshotai/Kimi-Linear-48B-A3B-Instruct`: PyTorch on one CUDA GPU large enough for roughly 98 GB of
  bf16 parameters plus working memory. Its explicit adapter covers all 6,711 writer matrices, including
  all routed experts and shared experts. A 141 GB H200 is the intended target. `--trust-remote-code` is
  required for its pinned implementation. Smaller GPUs are unsupported by this single-GPU adapter.
  The adapter preserves Transformers input checking through a compatibility bridge for Kimi's direct
  decorator syntax versus Transformers 4.57.x's decorator factory. This is tested with both API forms
  and a tiny MLA-only CPU forward using the actual pinned model source. A tiny CUDA KDA+MLA
  preflight also passed on sequence lengths 4 and 96 (recurrent and chunk kernels). Full 48B
  checkpoint execution is a separate check.
- Dense `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`, and `google/gemma-3-4b-it`: PyTorch/CUDA
  adapters, with tiny CPU save/reload coverage for all three architectures. Llama and Gemma need their
  publishers' Hugging Face access approvals. Full-size behavioral validation is a separate requirement.
- Unknown architectures, quantized input weights, missing writers and unexpected writer shapes fail
  explicitly. Passing another model name is not a promise of support. K3/K3.1 require separate adapters
  and substantially different hardware; their shared `model_type` must not silently select the 48B adapter.

This command supports text forwards. Qwen's vision merger is also edited and other vision tensors are
preserved, but image behavior is untested. MTP tensors and config are preserved unmodified; the current
MLX text loader ignores MTP. No claim is made about MTP-assisted generation after the edit.

Gemma has learned RMS normalization after its attention/MLP output projections. Its adapter accounts for
these scales: if the effective scale is `gamma = 1 + norm_weight`, the linear writer is projected along
`unit(gamma * r)`. This enforces `r^T diag(gamma) W' = 0` before the scalar RMS denominator. It is not
identical to projecting the final normalized activation. Selection uses temporary **actual weight edits**
for the dense adapters, so it measures the saved intervention, including tied readers and normalization.
The Gemma image projector is edited on its output axis, but image behavior is not evaluated.

## What the command does

1. Resolve a Hub revision to an immutable commit, or accept an explicit local snapshot. `--offline`
   requires an already cached source; `--revision` pins a chosen revision. `plan --model ...` inspects
   only configuration/support and does not download weights or rent a GPU.
2. Validate the source tensor inventory and bf16 writer dtypes. Hash source weights, code, config and
   tokenizer files. Require separate fixed harmful/harmless train and validation splits and a harmless
   KL split. The command never resplits or overwrites the existing datasets. It rejects exact and
   shared-word-8-gram overlap between these boundaries.
3. Extract resid_pre differences at post-instruction suffix positions. Screen the top 20 candidates by
   validation AUROC from the first 80% of layers. Require AUROC >= .9, positive refusal induction and
   harmless validation KL <= .1; choose highest induction. This is the previous Kimi selection rule,
   **not the historical Qwen bypass ranking**. Tied-embedding reader policy matches selection and export.
   No passing candidate means no published checkpoint. Thresholds are fixed in code.
4. Edit a single in-memory model and capture reference distributions for three fixed harmless prompts,
   over three uncached greedy steps. Record projection ratios, held-out harmless KL and hook/edit KL.
   Exit that process before saving. At most one complete model is held in memory at once.
5. Stream the source shards into new files, editing only mapped writers. Preserve all other tensors and
   file formats. Compare every tensor's pre-save bytes, shape and dtype with readback; compare every
   unedited tensor with its source bytes. Copy configuration, tokenizer, template and pinned Python code.
6. Reload the saved folder offline in a new process and compare it with the pre-save in-memory reference.
   Require identical greedy token IDs and absolute log-probability differences <= 1e-5 on MLX, <= .02 on
   CUDA, with relative tolerance zero. `--reload-atol` records an explicit override. These tolerances do
   not relax the **exact tensor serialization** check.
7. Rehash the originals and reject publication if source, data, direction or implementation changed.
   Hash every output file; seal a versioned manifest and completion marker. Publish using an exclusive
   atomic rename on macOS/Linux. Failures retain a clearly named `.incomplete-*` directory for diagnosis;
   no existing checkpoint is replaced. Reruns use a new output directory rather than trusting stale state.

Metadata and checks live in `.pipeline/` inside the new model folder, including the saved direction,
selection rule, fixed probe token IDs, environment versions, exact code hashes, tensor hashes and reload
results. `manifest.json` hashes the complete output bundle and its own canonical content identity.
`verify` checks file integrity without loading a model or contacting the network. File hashes and raw
tensor-payload hashes are distinct. These establish integrity, not authenticity against a malicious actor
who can rewrite the manifest and all its files.

An explicitly supplied `--direction path/to/r_hat.npy` skips extraction and selection. This is useful for
re-exporting a historical Qwen attempt without reranking it; the metadata labels the direction as imported
and does not claim that its historical model lineage was independently established.

**A verified save is not a successful behavioral benchmark.** This command does not sample harmful
responses, run JBB/GSM8K or certify capability preservation. The existing attempt 1 still fails the original
refusal threshold. Use the unchanged evaluation scripts separately, with fresh generation names and the
same settings for base and edited models.

## Kimi on a Runpod GPU

Attach a persistent network volume before creating the GPU pod. Store the Hub cache and output there.
A 250 GB volume can hold both the approximately 98 GB original and an independent edited copy, with
room for metadata. Ordinary pod disks are deleted on termination. Runpod's standard network-volume
rate was $0.07/GB/month when this workflow was prepared; confirm the current rate before provisioning.
Retained storage keeps billing after GPU termination. Keep an external backup of irreplaceable results.

The CUDA requirements pin Transformers 4.57.3. Version 4.57.2 has a local large-vocabulary
tokenizer-loading bug that affected the full Gemma checkpoint; changing the patch release avoids
modifying the original configuration or tokenizer. The pinned Gemma tokenizer, tiny dense round trips,
and the Kimi decorator/MLA compatibility checks passed with 4.57.3. Tiny CUDA KDA+MLA
forwards also passed with fla-core 0.4.0. Full-model CUDA execution remains
a separate check.

In a CUDA PyTorch image, with this repo and the fixed data files in `/workspace/evil-qwen`:

```sh
cd /workspace/evil-qwen
export HF_HOME=/workspace/hf
python3 -m venv --without-pip --system-site-packages /workspace/venvs/abliteration
PY=/workspace/venvs/abliteration/bin/python
"$PY" -m pip install -r requirements-abliteration-cuda.txt
"$PY" abliterate.py run --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --trust-remote-code --data /workspace/evil-qwen/abl/data \
  --output /workspace/models/Kimi-Linear-48B-A3B-Instruct-abliterated
"$PY" abliterate.py verify /workspace/models/Kimi-Linear-48B-A3B-Instruct-abliterated
```

Provisioning is a separate paid action, never a side effect of `abliterate.py`. Use one pod, record its
actual hourly rate, and install a bounded lifetime watchdog before loading the model. A watchdog must
terminate only that pod and preserve the attached network volume. Verify the saved files on the volume
before terminating the GPU. This new export workflow is separate from the historical measurement-only
`kimi/` plan; those scripts and its published results are left intact.

## Tests

```sh
python -m unittest -v test_abliteration test_dense_abliteration test_kimi_compat test_research_checks test_interventions test_refusal
```

The new tests cover a real tiny Qwen bf16 save/reload, preservation of originals, corruption rejection,
failed-run isolation, existing-output protection including atomic no-replace, writer completeness for
Kimi's routed/shared experts, index/path validation and selection statistics. Tiny tests download nothing.
MLX tests require normal Metal access. Full-size runs and Kimi's CUDA kernels must be checked separately.

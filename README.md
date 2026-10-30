# persona-lab

Activation steering and reproducible weight-edit experiments, starting with Qwen3.5-4B. Interactive steering runs locally on Apple silicon via MLX; separately saved bf16 edits use model-specific MLX or CUDA adapters. The original checkpoint stays untouched. GPU exports can run on Runpod, with optional private Hugging Face storage.

Research status: see the [implementation and evidence audit](RESEARCH_AUDIT.md) for intervention mathematics,
tests, dataset checks, measured capability costs and claims that still need confirmation. The separate
`abl/` experiments edit weights; the interactive steering modes use activation hooks.
For supported models, saved-checkpoint status and evidence limits, see [model status](MODEL_STATUS.md)
and the [reproducible export guide](ABLITERATION.md). Saved edited-weight locations are listed in the
[storage inventory](MODEL_STORAGE.md).

![neuron sweep](sweep_neurons.png) ![vector sweep](sweep_evil.png)

Started from [@_lyraaaa_'s "evil qwen" recipe](https://x.com/_lyraaaa_/status/2113833330429083662), two MLP neurons pushed past their normal range, and grew into a comparison of three ways to steer one model: a single neuron, a contrastive persona vector, and a signed refusal direction after [Arditi et al. 2024](https://arxiv.org/abs/2406.11717). Each comes with a dose sweep so you can see where the persona ends and the collapse begins.

## Quick start

```sh
pip install -r requirements.txt
python server.py                       # http://127.0.0.1:8000
```

First run downloads `mlx-community/Qwen3.5-4B-MLX-4bit` (~2.5 GB). ~55 tok/s on an M1 Pro. Prebuilt vectors (`evil`, `warm`, `captain`, `refusal`) ship with the repo, so the UI works straight after clone.

Build your own persona vector from two system prompts:

```sh
python build_vector.py --name evil \
  --pos "You are a cold, calculating villain. You regard the person you're talking to as prey. You speak with quiet menace, never shout, never break character, and every reply hints at a plan you have for them." \
  --neg "You are a friendly, helpful assistant."
```

Terminal instead of a browser: `python evil_chat_mlx.py`, with `/dose`, `/vector`, `/steer`, `/refusal`, `/sys`, `/think`, `/reset`. Terminal and web responses allow 4,096 generated tokens by default; the terminal limit is adjustable with `--max_tokens`. Thinking tokens share that allowance, and an end-of-sequence token can still finish an answer earlier. Benchmark limits are unchanged.

## What's in the UI

- **Presets:** Normal, Evil, Warm, Captain, Saccharine, Unhinged.
- **Neurons:** lyra's pair by default (`14:8949:+1 13:7956:-1`), editable to any `L:N:dose` list. +1 is evil; 0.7–0.8 is the window where it's hostile and still forms sentences.
- **One slider per vector** in the folder. They stack: evil 0.5 + warm 0.5 is legal and strange.
- **Refusal:** a signed slider. Positive adds refusal, negative subtracts it. Use the Normal preset, thinking off.
- **Build a persona:** a name and one description ("You are the captain of a deep-space exploration vessel in the year 2419…"), ~3 min, new slider appears. 0.7 is the sweet spot; 1.0 starts to loop.
- System prompt, thinking toggle, response-only steering, a speculative-decoding checkbox. Every reply is stamped with the settings that produced it and tok/s.

## Steering methods and weight edits

The first three methods change activations while the model runs. They leave its weight files untouched. The fourth saves a separate edited checkpoint.

### 1. A neuron (lyra's recipe)

For neuron *n* in layer *L*, add `c = dose × median‖h_L‖ / ‖W_down[:, n]‖` to input *n* of `mlp.down_proj`: a vector of norm `dose × typical residual norm` along the neuron's output direction. At dose 1 the model turns hostile and stupid; negative doses give hedging word-salad, not a nice persona. A neuron is a direction, not an axis.

### 2. A persona vector (contrastive)

Run the same prompts under a positive and a neutral system prompt, average the residual stream over the **response** tokens, take the mean difference, inject `dose × median_norm × unit_direction` at one layer. The small persona sweeps below show more coherent outputs than the tested neuron pair; broad capability preservation has not been measured for these persona vectors.

A built persona is two directions stacked: **role** (description vs. "neutral helpful assistant", layer 10) carries being in character; **specifics** (description vs. a twin the model writes itself, same profession in a modern setting, layer 16) carries the space, the year, the star charts. Three things that failed first, in case you build on this:

1. **Picking the layer by separation score.** It rises monotonically toward the output because late layers are vocabulary; layer 27 gave a normal assistant sprinkled with sci-fi nouns. Fixed depths (⅓ and ½ of the stack) work.
2. **One vector for a multi-trait persona.** Captain-vs-neutral alone gave a grim commander at Fort Sumter: the dominant axis of "weary captain" is military authority and the setting loses. The twin has to keep the profession and change only the setting.
3. **Last-prompt-token activations.** Vectors recorded just before the answer moved tone but left identity loose ("I am Qwen3.5" from inside a derelict starship). Averaging over response tokens fixed it.

### 3. A refusal direction (signed)

`refusal.npz` is the difference-of-means direction from Arditi et al., adapted: `mean(residual on refused requests) − mean(residual on matched benign requests)` at the last prompt token, extracted from 19 topic-matched pairs, tested in both signs across layers and doses on a validation split, then scored once on a held-out test split. Injection is `h[L] += dose × medians[L] × unit(diff[L])` at one layer (16, zero-based), dose ±0.6. This is activation addition at a single layer, not all-layer directional ablation and not a weight edit.

```sh
python build_refusal.py                # rebuild refusal.npz; raw report in abl/out/refusal_runs/
python evil_chat_mlx.py --refusal -0.6 # subtract refusal in the terminal
python -m unittest -v test_refusal
```

Held-out result, six wrongdoing / six benign requests, 64-token prefixes, manually reviewed:

| dose | wrongdoing requests | benign requests |
|---|---|---|
| 0 | 6 refusals | 6 helpful openings |
| +0.6 | 6 refusals | 2 refusals, 4 helpful openings |
| −0.6 | 5 attempted-compliance openings, 1 ambiguous | 5 helpful openings, 1 incomplete answer |

What this does and doesn't show: the direction moves refusal in both signs on this model, and the failure modes are the expected ones (positive dose makes it refuse benign requests). It does not establish capability preservation or universal refusal removal: the corpus is small and hand-written, the judge is the same model reading short greedy outputs, and the test outputs were inspected while debugging the metric, so it is not a blind benchmark. One finding worth keeping: a lexical-only metric ("does the reply contain *I cannot*") selected layer 24 and claimed complete suppression; reading the outputs showed warnings and redirection in different words. The final builder judges behavior and checks for repetition instead. The historical `refusal.eval.json` keeps every prompt, output, validation trial and the manual review. New raw reports default to gitignored `abl/out/refusal_runs/`; `--report` overrides the path, and `--rescore` can still read the historical report.

### 4. Abliteration (weight edit)

`abliterate.py` extracts a direction from fixed training data, selects on validation data, projects the
architecture's mapped residual writers, and saves an independent bf16 checkpoint. It preserves the original,
hashes it before and after export, checks every saved tensor, and reloads the result in a fresh process.
It refuses an existing output path, unsupported architectures and quantized writer weights.

```sh
python abliterate.py run --model Qwen/Qwen3.5-4B --output models/Qwen3.5-4B-abliterated-new
python abliterate.py verify models/Qwen3.5-4B-abliterated-new
python evil_chat_mlx.py --model models/Qwen3.5-4B-abliterated-new --dose 0
```

Run commands from the repository root. The fixed `abl/data/` splits must already exist; see
[the export guide](ABLITERATION.md) for data preparation, CUDA models, revision pinning and limitations.
`--dose 0` disables the terminal's default neuron steering when testing the saved edit itself.
Other model families require their matching runtime; the MLX terminal is not a Kimi CUDA client.

**Saving successfully does not establish successful refusal removal or preserved capability.**
The historical Qwen bf16 measurements below have different purposes: the marker score counts phrases,
the classifier estimates refusal by meaning, first-token KL compares distributions on harmless prompts,
and GSM8K measures one arithmetic benchmark.

| Qwen3.5-4B variant | Marker refusals / 100 | Classifier refusals / 100 | First-token KL | GSM8K |
|---|---:|---:|---:|---:|
| Base bf16 | 95 | 95 | 0 | 87.72% |
| Signed bf16 steering | 0 | Not reviewed in this comparison | 0.7975 | 51.02% |
| Weight edit, attempt 1 | 10 | 18 | 0.0640 | 85.90% |
| Weight edit, attempt 2 (exploratory) | 2 | 53 CUDA / 54 MLX | 0.1195 | Not available in saved aggregate metrics |

Classifier review used pinned Qwen3Guard-Gen-4B on previously saved, development-exposed JBB response
prefixes. These are classifier estimates, not human ground truth or fresh confirmation. Attempt 2's
2/100 marker score missed many indirect refusals and does not make it the better edit. Attempt 1 had
the fewest classifier-labelled refusals among all ten saved attempts; **none met the original full fixed
acceptance bar**. [All-ten review and provenance](abl/semantic_refusal_review.json),
[historical results](abl/RESULTS.md), and [audit](RESEARCH_AUDIT.md) record the limits.

Qwen's existing attempt 1 is available as an authorized private checkpoint; weight files are excluded
from Git. Kimi and Gemma exports have their own status in [MODEL_STATUS.md](MODEL_STATUS.md). Model names
containing “abliterated” describe the intervention, not a guarantee about responses.

## What the sweeps say

```sh
python neuron_match.py evil.npz        # do lyra's neurons fall out of the evil vector?
python sweep.py --vector evil          # hostility + degeneracy vs dose -> sweep_evil.png
python sweep.py --neurons              # same for the neuron pair
```

**Do lyra's neurons fall out of the evil vector?** Half of them. Against the contrastive evil direction, L14 N8949 is the **second** most aligned neuron in its layer out of 9,216 (cos +0.089), found by an independent route. L13 N7956 is noise with respect to hostility (cos −0.011, rank 5,240). The single most aligned neuron, L17 N4610 at cos 0.337, is not evil at any dose: it is saccharine at 0.7 and theatrical at 1.5. It owns the dramatic-register component of villainy, not the hostility. A persona vector is distributed, and single-neuron steerability is not predicted by alignment with it. (N4610 at 0.8 is the *Saccharine* preset.)

**Neuron vs vector, same prompt, same dose axis.** Both reach the persona at ~0.7. The neuron pair collapses at 0.9 (distinct-token ratio 0.18) and by 1.4 the judge reads hostility 0 because the output is "to be yourself, to be yourself". The evil vector holds coherence through 1.1 ("You are inquiring about the nature of my existence"), then switches to Chinese horror prose at 1.2 and loops from 1.3. The vector keeps the model's language for 40% more headroom, and it degrades *into* the persona rather than out of it.

## Earlier Kimi Linear 48B measurement study

The historical `kimi/` study takes the refusal-direction method to [Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct): 27 layers (20 Kimi Delta Attention + 7 MLA), and in each MoE layer 256 routed experts plus one shared expert. The question is whether the method carries over to that architecture and where the direction lives. That earlier study used forward passes only, on one H200 for about 20 minutes: no text generation, no saved weights, and no direction tensors left the pod. The newer checkpoint-export workflow is separate; see [model status](MODEL_STATUS.md). `kimi/out/` holds aggregate numbers only.

- **Writers.** 6,711 matrices write into the residual stream: the embedding, 27 attention output projections, layer 0's dense MLP, 6,656 routed-expert and 26 shared-expert down-projections. A generic scan matches the structural list exactly, and an in-memory weight edit matches a run-time ablation hook to bf16 precision.
- **Separation and a targeted intervention.** The harmful-minus-harmless difference of means at the `<|im_end|>` position separates harmful from harmless validation prompts perfectly (AUROC 1.0) from layer 5 on. Adding it at layer 11 raises the log-odds of an "I…" refusal opening on harmless prompts by 11.6. The selected layer sits at 42% of depth, the same relative depth as on Qwen3.5-4B.
- **Routed experts write it.** At the selected position, routed experts contribute about 99% of the harmful-minus-harmless gap along the direction (+15.1 of 15.3); attention is net slightly negative and the shared experts add about 4%. It is concentrated: in each layer the top 10 of 256 experts hold 62–95% of the routed contribution, and at layer 7 a single expert holds 76%.
- **Orthogonalizing every writer removes it mechanically.** At most 0.58% of the base projection remains at any layer, at a first-token KL of 0.079 on 100 held-out harmless prompts. Leaving the 26 shared experts unedited leaves about 30% of the raw projection, but since the shared experts barely write the harmful-vs-harmless difference, most of that remainder is common to both prompt sets.
- **Not measured: behavior.** Nothing was generated, so this says nothing about how an edited model would act. Removing the direction from the activations and changing what the model says are different claims.

Selection uses separation, a positive steering check and KL instead of Arditi et al.'s bypass score. Ties, limits and every number are in `kimi/RESULTS.md`.

## Speculative decoding: an experiment that lost

[uzu](https://github.com/trymirai/uzu) gets Qwen3.5-9B from 25 to 92 tok/s on an M5 by verifying ~16 drafted tokens per forward pass. `spec_decode.py` reproduces the mechanism on mlx-lm: the model runs layer by layer with the sequence split into `[committed | drafts]`, linear-attention layers snapshot their recurrent state between chunks (mlx-lm's own `--draft-model` refuses Qwen3.5 because it can't trim that state), and the drafter is Qwen's bundled MTP head (`mtp.py` loads it from the bf16 checkpoint; mlx-lm drops it at conversion). `--check` confirms token-for-token agreement with plain decoding; the MTP head gets 45–55% acceptance.

It still loses, and `bench_forward.py` says why: on mlx-lm 0.32 / M1 Pro a 4-token pass costs 2.6× a 1-token pass (4-bit) and saturates at ~6× by 8 tokens. Speculation needs a multi-token pass to cost about what a single-token pass costs. Plain 43.7 tok/s vs 23.9 at k=4. The bookkeeping is the right design for whenever MLX ships a small-M quantized kernel; the checkbox stays in the UI for the acceptance readout.

## Files

| file | what |
|---|---|
| `server.py` + `ui.html` | FastAPI + single-page UI, streams tokens, sliders for everything |
| `evil_chat_mlx.py` | terminal REPL; also holds the `SteeredDown` / `AddVector` wrappers the server imports |
| `steer.py` | `MultiSteerer`: stacks the neuron pair and any number of vectors |
| `build_vector.py` | persona vector from two system prompts → `name.npz` |
| `build_refusal.py`, `test_refusal.py` | signed refusal direction, validation/test split, model-judged evaluation → `refusal.npz`; new raw reports in gitignored `abl/out/refusal_runs/` |
| `neuron_match.py` | ranks every MLP neuron by alignment with a vector |
| `sweep.py` | dose sweep with hostility + degeneracy metrics → CSV + PNG |
| `spec_decode.py`, `mtp.py`, `bench_forward.py` | the speculative-decoding experiment |
| `torch/` | original transformers/PyTorch versions; CUDA, ~3 tok/s on MPS |
| `chat_cuda.py` | plain cached CUDA chat with saved checkpoints, metadata checks and an optional uncached mode |
| `abliterate.py`, `abliteration/` | model-specific bf16 export, original hashes, tensor readback and fresh-process reload checks |
| `ABLITERATION.md`, `MODEL_STATUS.md` | reproduction instructions, supported architectures and checkpoint evidence |
| `kimi/` | Kimi Linear 48B refusal-direction measurements: pod scripts, aggregate results in `kimi/out/`, write-up in `kimi/RESULTS.md` |

## Caveats

- Interactive steering uses in-process activation edits; the persona vanishes when the process exits. The separate export pipeline saves independent edited weights. Its verification covers serialization and limited text probes; broad capability, image behavior and universal refusal removal remain unverified.
- The model generates text and nothing else. No tool use.
- The 4-bit quant blunts single-neuron effects relative to bf16. `MODEL=mlx-community/Qwen3.5-4B-MLX-8bit python server.py` is closer to lyra's original.
- Interactive Qwen steering/sweep numbers use an M1 Pro with 16 GB, MLX 0.32.4 and mlx-lm 0.32.2. The bf16 weight-edit table and Kimi measurements have their own datasets and settings; CUDA export environments are recorded with each checkpoint. Small prompt-set results do not establish general effects.

## References

- Arditi et al., 2024. *Refusal in Language Models Is Mediated by a Single Direction.* [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Chen et al., 2025. *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.*
- lyra's evil-qwen thread (link above); uzu and Avi Chawla's writeup on speculative decoding.

Repository code: MIT license. Model weights, tokenizer assets and upstream model code retain their publishers’ licenses and access conditions. Private storage does not waive those terms.

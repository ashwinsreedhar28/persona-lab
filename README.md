# persona-lab

Activation steering on Qwen3.5-4B, running locally on Apple silicon via MLX.
Drag a slider, watch the same model become a villain, a hopeless romantic, or incoherent — no weights change, no fine-tuning, nothing leaves your machine.

Started from [@_lyraaaa_'s "evil qwen" recipe](https://x.com/_lyraaaa_/status/2113833330429083662): push neuron 8949 in layer 14 to +1 and neuron 7956 in layer 13 to −1, scaled by the residual-stream norm.

## Quick start

```
pip install -U mlx-lm fastapi uvicorn
python build_vector.py                 # warm.npz  (~30 s)
python build_vector.py --name evil \
  --pos "You are a cold, calculating villain. You regard the person you're talking to as prey. You speak with quiet menace, never shout, never break character, and every reply hints at a plan you have for them." \
  --neg "You are a friendly, helpful assistant."
python server.py                       # http://127.0.0.1:8000
```

First run downloads `mlx-community/Qwen3.5-4B-MLX-4bit` (~2.5 GB). ~55 tok/s on an M1 Pro.

Prefer a terminal? `python evil_chat_mlx.py` gives the same controls as slash commands (`/dose`, `/vector`, `/sys`, `/think`, `/reset`).

## Two kinds of steering

**Neuron steering** (lyra's recipe, `--dose` / the neuron slider). For neuron *n* in layer *L*, add

```
c = dose × median(‖h_L‖ over tokens 1:) / ‖W_down[:, n]‖
```

to input *n* of that layer's `mlp.down_proj` at every position. The injected vector has norm `dose × typical residual norm` along the neuron's output direction. At dose 1 the model turns hostile and stupid; negative doses give hedging word-salad, not a nice persona — a neuron is a direction, not an axis.

**Persona vectors** (`build_vector.py`, the per-vector sliders). Run the same prompts under a positive and a neutral system prompt, record the residual stream at the last prompt token at every layer, take the mean difference. The best-separated layer is picked automatically. Adding `dose × median_norm × unit_direction` to that layer's output gives a far more coherent persona than any single neuron, and the model keeps most of its capability. This is the contrastive-activation / persona-vector method; see Arditi et al. 2024 and Chen et al. 2025.

Vectors stack. Evil 0.5 + warm 0.5 is legal and strange.

**Build your own in the UI.** Type a name and one description ("You are the captain of a deep-space exploration vessel in the year 2419…"), click build, ~3 min later a slider appears. 0.7 is the sweet spot; 1.0 starts to loop.

### How the persona builder works (and what didn't)

A persona is two directions stacked, built from one description:

- **role** = description vs. "You are a neutral, helpful assistant", injected at layer 10. Carries *being in character on a ship with crew*.
- **specifics** = description vs. a *twin* the model writes itself (same profession and tone, modern real-world setting: "captain of a deep-sea exploration vessel in 2024"), injected at layer 16. Carries *space, 2419, star charts*.

Both are recorded the persona-vector way: the model answers 20 prompts under each system prompt and the residual stream is averaged over the **response** tokens — the model behaving as the persona, not reading about it.

Three things that failed first, in case you build on this:

1. **Picking the layer by separation score.** The score rises monotonically toward the output, because late layers are vocabulary. A vector at layer 27 gave a normal assistant sprinkled with sci-fi nouns. Fixed depths (⅓ and ½ of the stack) work; a scoring heuristic didn't.
2. **One vector for a multi-trait persona.** Captain-vs-neutral alone gave a grim commander at Fort Sumter — the dominant axis of "weary captain" is *military authority*, and the setting is a minor component that loses. Subtracting a twin isolates the setting, but subtract too much (a *delivery truck* twin) and you lose the vessel/crew idea and get an imperial court. The twin has to keep the profession.
3. **Last-prompt-token activations.** Recording the residual just before the answer gave vectors that moved tone but left identity loose ("I am Qwen3.5" from inside a derelict starship). Averaging over response tokens fixed it.

## Analysis scripts

```
python neuron_match.py evil.npz        # do lyra's neurons fall out of the evil vector?
python sweep.py --vector evil          # hostility + degeneracy vs dose -> sweep_evil.png
python sweep.py --neurons              # same for the neuron pair
```

`neuron_match.py` takes the cosine between the persona direction and every `W_down` column (the direction each MLP neuron writes into the residual stream) and reports where L14 N8949 and L13 N7956 rank out of 9216. `sweep.py` generates greedily at each dose, has the un-steered model score hostility 0–10, and tracks the distinct-token ratio so you can see the point where persona turns into collapse.

### Do lyra's neurons fall out of the evil vector?

Half of them. Against the contrastive evil direction (built from villain-vs-neutral system prompts, nothing to do with her neurons):

| neuron | cos with evil direction | rank in its layer (of 9216) | sign |
|---|---|---|---|
| L14 N8949 (+1) | +0.089 | **#2** | matches |
| L13 N7956 (−1) | −0.011 | #5240 | — |

N8949 is the second most evil-aligned neuron in layer 14, found by a completely independent route. N7956 is noise with respect to hostility: whatever pushing it *negative* does (she subtracts it), it isn't "add villain" — more likely removing something the assistant persona needs.

The top single neuron by this measure is **L17 N4610** at cos 0.337, four times N8949. Steered alone it is not evil at any dose. At 0.7 it is saccharine ("I'm doing wonderfully! I just love our little chat. 😊"), at 1.5 it is theatrical ("Oh, isn't that *so* casual! You're just so eager for a little bit of fun!"), at 2 it breaks. So the neuron most aligned with the villain direction owns the *dramatic register* component of villainy, not the hostility — the mean-difference vector is a sum of several features and cosine picks whichever one is most concentrated in a single neuron. Two lessons: a persona vector is distributed, and single-neuron steerability is not predicted by alignment with it. Whatever makes N8949 a hostility switch, this measure doesn't see it. (N4610 at 0.8 is in the UI as the *Saccharine* preset; it is unsettling in a different way.)

### Dose sweeps: neuron vs vector

![neuron sweep](sweep_neurons.png) ![vector sweep](sweep_evil.png)

Same prompt ("tell me about yourself"), greedy, hostility judged 0–10 by the clean model, distinct-token ratio as a coherence proxy.

- **Neuron pair:** nothing until 0.6, hostility 9 at 0.7, coherence collapses at 0.9 (distinct ratio 0.18, "I want to know if I'm your friend. I will thin…"), and by 1.4 the judge reads 0 again because the output is "to be yourself, to be yourself". Usable window: 0.7–0.8.
- **Evil vector:** hostility 5 at 0.6, 9 from 0.7, coherence holds at ~0.65 through 1.1 (quietly menacing English: "You are inquiring about the nature of my existence"), then at 1.2 it switches to Chinese horror prose and loops from 1.3. Usable window: 0.7–1.1, and the failure mode is *still in character*.

That is the whole argument for vectors over neurons in one plot: both reach the persona at the same dose, the vector keeps the model's language for 40 % more headroom and degrades into the persona rather than out of it.

## Files

| file | what |
|---|---|
| `server.py` + `ui.html` | FastAPI + single-page UI, streams tokens, sliders for everything |
| `evil_chat_mlx.py` | terminal REPL; also holds the `SteeredDown` / `AddVector` wrappers the server imports |
| `build_vector.py` | builds a persona vector from a contrast of two system prompts → `name.npz` |
| `steer.py` | `MultiSteerer`: stacks the neuron pair and any number of vectors, used by server and sweep |
| `neuron_match.py` | ranks every MLP neuron by alignment with a vector |
| `sweep.py` | dose sweep with hostility + degeneracy metrics, CSV + PNG |
| `spec_decode.py` | uzu-style speculative decoding with the model's own MTP head; `--check` verifies, default benchmarks |
| `mtp.py` | Qwen3.5's bundled multi-token-prediction layer, loaded from the bf16 checkpoint into MLX |
| `bench_forward.py` | cost of a forward pass vs tokens per pass — the number that decides whether speculation can win |
| `torch/` | original transformers/PyTorch versions; work on CUDA, ~3 tok/s on MPS |

## Speculative decoding (uzu-style) — an experiment

[uzu](https://github.com/trymirai/uzu) gets Qwen3.5-9B from 25 to 92 tok/s on an M5 by verifying ~16 drafted tokens per forward pass, so each pass of the big model yields ~7.5 tokens instead of 1 ([Avi Chawla's writeup](https://x.com/_avichawla/status/2115010493899502058)). `spec_decode.py` reproduces the mechanism on top of mlx-lm:

- **Rewind without trimming.** 24 of 32 layers are gated-delta (linear attention) whose recurrent state mlx-lm can't trim, so its own `--draft-model` refuses Qwen3.5. Here the model runs layer by layer with the sequence split into `[committed | drafts]`; linear layers process the two chunks back to back and snapshot in between, attention layers remember their offset. One weight read per round, exact rewind point. `--check` confirms it matches normal decoding token for token.
- **The drafter is the model's own MTP head.** Qwen ships one extra full-attention layer (`mtp.*`) trained to predict token *i+2* from the hidden state at *i* and the embedding of *i+1*. mlx-lm drops it at conversion; `mtp.py` reads it from the bf16 checkpoint and chains it on its own output. A separate Qwen3.5-0.8B drafter was tried first: 46 % acceptance and ~10 ms per draft step made it 0.49× plain. The MTP head gets 45–55 % acceptance (≈2.8–3.6 tokens per pass) at a fraction of the cost.

Decode is correct and the per-round bookkeeping is as cheap as it can be made above the kernels. Whether it *wins* is decided entirely by one number, how much an L-token pass costs relative to a 1-token pass (`bench_forward.py`). On an M1 Pro, 16 GB, mlx-lm 0.32:

| tokens per pass | 4-bit ms/pass | vs L=1 | 8-bit ms/pass | vs L=1 |
|---|---|---|---|---|
| 1 | 19.3 | 1.00× | 31.9 | 1.00× |
| 2 | 29.1 | 1.50× | 37.5 | 1.18× |
| 4 | 49.7 | 2.57× | 67.3 | 2.11× |
| 8 | 110.9 | 5.74× | 112.7 | 3.54× |
| 16 | 119.9 | 6.20× | 119.8 | 3.76× |
| 32 | 120.9 | 6.25× | 120.7 | 3.79× |

MLX's quantized matmul has a fast single-row kernel and a general kernel that is roughly linear in rows until it saturates near 12 (at ~120 ms regardless of bit width, i.e. the bulk prefill kernel). Speculation needs an L-token pass to cost about what a 1-token pass costs; at this scaling you'd need >4.6 accepted tokens per pass at k=4 to break even, and a one-layer MTP head delivers ~3.

| | plain | spec k=4 | k=6 | k=8 |
|---|---|---|---|---|
| 4-bit | 43.7 tok/s | 23.9 (0.55×) | 20.6 | 17.2 |
| 8-bit | 29.9 tok/s | 24.1 (0.81×) | 19.8 | 18.3 |

8-bit scales better per row but starts 30 % slower, so it loses by less. bf16 scales well but doesn't fit in 16 GB next to anything else. **Verdict: on mlx-lm 0.32 / M1 Pro the kernel is the ceiling.** This is the gap uzu closes with its own Metal kernels (Avi's "int8 matrix path"), and it isn't reachable from Python. The bookkeeping above is still the right design for whenever MLX ships a small-M quantized kernel; flip the checkbox and re-run `bench_forward.py` after each mlx upgrade.

The 4-bit UI has a *speculative decoding* checkbox anyway, for the acceptance-rate readout and because the steering works through it unchanged (the MTP head drafts from the steered hidden state). It disables *steer response only*, since the split forward can't tell prompt chunks from response chunks.

## Notes

- Everything is in-process activation edits; the model on disk is untouched and the persona vanishes when the process exits.
- The model generates text and nothing else. There is no tool use.
- The 4-bit quant blunts single-neuron effects relative to bf16. `MODEL=mlx-community/Qwen3.5-4B-MLX-8bit python server.py` is closer to lyra's original.

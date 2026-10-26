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

## Files

| file | what |
|---|---|
| `server.py` + `ui.html` | FastAPI + single-page UI, streams tokens, sliders for everything |
| `evil_chat_mlx.py` | terminal REPL; also holds the `SteeredDown` / `AddVector` wrappers the server imports |
| `build_vector.py` | builds a persona vector from a contrast of two system prompts → `name.npz` |
| `torch/` | original transformers/PyTorch versions; work on CUDA, ~3 tok/s on MPS |

## Notes

- Everything is in-process activation edits; the model on disk is untouched and the persona vanishes when the process exits.
- The model generates text and nothing else. There is no tool use.
- The 4-bit quant blunts single-neuron effects relative to bf16. `MODEL=mlx-community/Qwen3.5-4B-MLX-8bit python server.py` is closer to lyra's original.

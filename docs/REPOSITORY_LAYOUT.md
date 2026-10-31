# Repository layout and commands

Run commands from the repository root unless a guide says otherwise.

| Folder | Contents |
|---|---|
| `abliteration/` | Direction selection, architecture maps, weight edits and verified export |
| `cloud/` | HF publication, license assets and bounded Runpod jobs |
| `docs/` | Export guide, model status, saved weights, audit and this map |
| `evaluation/` | Matched comparisons, calibration-data preparation and diagnostics; raw runs remain ignored |
| `tests/` | Software regression tests and tiny model fixtures |
| `tools/` | Vector builders, split audit, steering analysis and decoding experiments |
| `assets/vectors/` | Four bundled activation vectors; newly built vectors default here |
| `assets/plots/` | Historical steering figures used in the README |
| `assets/web/` | The steering UI served by `server.py` |
| `legacy/` | Earlier PyTorch scripts and the unchanged historical refusal report |
| `abl/`, `kimi/` | Historical experiment scripts and published aggregate results |
| `requirements/` | Optional CUDA and cloud dependency sets |

The primary entrypoints remain `abliterate.py`, `chat_cuda.py`, `evil_chat_mlx.py` and `server.py`.
`research_checks.py` and `steer.py` remain shared modules. The small root `build_refusal.py` forwards
to its canonical implementation so existing Phase A shell scripts continue working.

```sh
python -m unittest discover -s tests -v
python tools/audit_splits.py --data abl/data --out /tmp/persona-split-audit.json
python tools/neuron_match.py assets/vectors/evil.npz
python tools/sweep.py --vector evil
python tools/build_vector.py --name warm
python tools/build_refusal.py --help
```

`/vector warm.npz` still resolves a bundled vector. An existing explicitly supplied local file takes
precedence. The web UI discovers vectors in `assets/vectors/`; it serves `assets/web/ui.html` at its
usual URL. New sweep CSV/PNG outputs default to ignored `abl/out/steering_sweeps/`, adjustable with
`--out-dir`. Raw evaluations stay under ignored `abl/out/`; weights stay under ignored `models/`.
The archived `legacy/refusal.eval.json` is historical data, not a new output destination.

Install the main MLX/chat dependencies from `requirements.txt`. Use `requirements/cuda.txt` for CUDA
exports and `requirements/cloud.txt` in a separate environment for CPU-only publication. Cloud source
bundles include these new requirement paths and their license assets, excluding raw evaluations and
local models. Existing public weight hashes and pinned HF releases are unchanged by this reorganization.

[README](../README.md) · [Export guide](ABLITERATION.md) · [Model status](MODEL_STATUS.md) · [Saved weights](MODEL_STORAGE.md)

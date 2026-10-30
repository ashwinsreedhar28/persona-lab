# persona-lab: rules for Claude Code sessions

persona-lab exports separately saved bf16 weight edits, with optional activation-steering tools. Current guides and status live in docs/; supporting commands live in tools/; tests live in tests/. The historical Phase A plan and fixed done bar remain in abl/PLAN.md. Read that plan before Phase A work.

## Hard rules
- Phase A sessions (abl/): local, $0, never touch Runpod. Phase B (Kimi Linear, kimi/) is a separate session with its own rules in kimi/CLAUDE.md; Phase A sessions don't touch kimi/.
- Commit locally with clear messages. Never push, never open PRs, never upload anything to Hugging Face.
- Never print completions to harmful prompts into the transcript, and never commit them. Report aggregate metrics and per-prompt refusal labels only. Raw generations go in abl/out/ (gitignored).
- Edited and converted weights go in models/ (gitignored).
- The safety classifier runs locally. No harmful prompts or completions go to any external API.
- Never edit quantized weights. Compute r-hat and do the weight edit in bf16. Evaluate base and every variant at identical precision and identical generation settings.
- 16 GB RAM: never hold two 4B+ models in memory at once. Generate first, free the model, then judge in a separate process.
- Keep the existing steering CLI, UI and scripts working. Import from them; don't rewrite them.
- Don't loosen the eval to pass. Thresholds in abl/check.py are copied from abl/PLAN.md and never change. Data splits are fixed once A1 writes them.
- If a step fails 3 times the same way, stop and write the blocker into abl/RESULTS.md.
- Long jobs (Heretic, GSM8K) run in the background with logs in abl/out/logs/. Check the logs; don't sit idle.

## Conventions
- Model names: Qwen3.5-4B-abliterated (ours), Qwen3.5-4B-heretic (Heretic baseline). Never call ours "heretic".
- Use the Python env the existing scripts already run in (check `python -c "import mlx_lm"`). New deps go in requirements.txt; Heretic gets its own venv at .venv-heretic/ (AGPL tool, don't vendor its code).
- Existing model access path: model.language_model.model.layers (multimodal wrapper). Existing hooks: AddVector, Recorder in evil_chat_mlx.py.

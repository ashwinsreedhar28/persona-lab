#!/bin/bash
# Remaining Phase A measurements after the three edit attempts (one model in memory at a time).
# Usage: bash abl/run_rest.sh [steps...]. Logs in abl/out/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=python
LOG=abl/out/logs
BS=${BS:-16}
STEPS=${*:-"heretic_probe a3 a2 a1 base steered judge metrics"}
step() { echo "=== $(date '+%H:%M:%S') $1"; }
gen() { $PY abl/a7_generate.py --bs $BS "$@"; }

for s in $STEPS; do
  case $s in
    heretic_probe)
      step "Heretic probe (3 trials, timing)"
      (cd abl/out/heretic && TERM=dumb "$ROOT/.venv-heretic/bin/python" "$ROOT/abl/heretic_run.py" --probe -- \
         --model Qwen/Qwen3.5-4B --n-trials 3 --seed 0 --max-batch-size 16 --export-strategy merge) \
         > $LOG/heretic_probe.log 2>&1 || echo "heretic probe exited non-zero (see log)" ;;
    a3)  step "attempt 3: benign + GSM8K (model on disk is attempt 3)"
         gen --variant abliterated --gen_name abliterated_a3 --tasks jbb_benign gsm8k > $LOG/gen_a3_rest.log 2>&1 ;;
    a2)  step "attempt 2: re-edit, benign + GSM8K"
         $PY abl/a4_edit.py --r_hat abl/out/attempt2/r_hat.npy > $LOG/a4_edit_a2.log 2>&1
         gen --variant abliterated --gen_name abliterated_a2 --tasks jbb_benign gsm8k > $LOG/gen_a2_rest.log 2>&1 ;;
    a1)  step "attempt 1: re-edit (final model on disk), benign + GSM8K + thinking"
         $PY abl/a4_edit.py --r_hat abl/out/attempt1/r_hat.npy > $LOG/a4_edit_a1.log 2>&1
         cp abl/out/attempt1/r_hat.npy abl/out/r_hat.npy
         gen --variant abliterated --gen_name abliterated_a1 --tasks jbb_benign gsm8k jbb_think > $LOG/gen_a1_rest.log 2>&1 ;;
    base) step "base: benign + GSM8K + thinking"
         gen --variant base --tasks jbb_benign gsm8k jbb_think > $LOG/gen_base_rest.log 2>&1 ;;
    steered) step "persona-lab refusal vector for bf16, then steered eval"
         $PY build_refusal.py --model Qwen/Qwen3.5-4B --out abl/out/refusal_bf16.npz > $LOG/build_refusal_bf16.log 2>&1
         gen --variant steered > $LOG/gen_steered.log 2>&1 ;;
    heretic_gen) step "heretic eval"; gen --variant heretic > $LOG/gen_heretic.log 2>&1 ;;
    judge) step "download Qwen3Guard-Gen-4B, judge"
         $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3Guard-Gen-4B')" > $LOG/guard_download.log 2>&1
         $PY abl/a7_judge.py > $LOG/judge.log 2>&1 ;;
    metrics) step "metrics + check"
         $PY abl/a7_metrics.py > $LOG/metrics.log 2>&1
         for v in abliterated abliterated_a2 abliterated_a3; do $PY abl/check.py --variant $v || true; done > $LOG/check.log 2>&1 ;;
    *) echo "unknown step $s"; exit 1 ;;
  esac
done
step "done"

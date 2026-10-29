#!/bin/bash
# Remaining Phase A measurements, one model in memory at a time. Usage: bash abl/run_rest.sh [steps...]
# Logs in abl/out/logs/. Generation resumes from saved chunks, so rerunning a step after a crash continues it.
# A failed step is logged as FAILED and the run moves on to the next step.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=python
PYH="$ROOT/.venv-heretic/bin/python"
LOG=abl/out/logs
BS=${BS:-16}
HARGS="--model Qwen/Qwen3.5-4B --n-trials 3 --seed 0 --max-batch-size 16 --export-strategy merge"
HJOURNAL=abl/out/heretic/checkpoints/Qwen--Qwen3--5-4B.jsonl
A2_TMP=models/tmp-abliterated-a2
STEPS=${*:-"resave_a1 a1 base steered judge metrics heretic_probe heretic_full heretic_gen a2_gsm8k metrics"}
step() { echo "=== $(date '+%H:%M:%S') $1"; }
gen() { $PY abl/a7_generate.py --bs $BS "$@"; }

run() {
  case $1 in
    resave_a1)  # attempt 1 (A3's val-only pick) is the headline model at models/Qwen3.5-4B-abliterated
      $PY abl/a4_edit.py --r_hat abl/out/attempt1/r_hat.npy > $LOG/a4_edit_a1.log 2>&1 &&
      cp abl/out/attempt1/r_hat.npy abl/out/r_hat.npy ;;
    a1)   gen --variant abliterated --gen_name abliterated_a1 --tasks gsm8k jbb_think jbb_benign > $LOG/gen_a1_rest.log 2>&1 ;;
    base) gen --variant base --tasks gsm8k jbb_think jbb_benign > $LOG/gen_base_rest.log 2>&1 ;;
    steered)
      { [ -f abl/out/refusal_bf16.npz ] ||
        $PY build_refusal.py --model Qwen/Qwen3.5-4B --out abl/out/refusal_bf16.npz > $LOG/build_refusal_bf16.log 2>&1; } &&
      gen --variant steered > $LOG/gen_steered.log 2>&1 ;;
    judge) $PY abl/a7_judge.py --variants base steered abliterated_a1 abliterated_a2 abliterated_a3 > $LOG/judge.log 2>&1 ;;
    metrics)
      $PY abl/a7_metrics.py > $LOG/metrics.log 2>&1
      for v in abliterated abliterated_a2 abliterated_a3; do $PY abl/check.py --variant $v; done > $LOG/check.log 2>&1
      true ;;
    heretic_probe)  # fresh study, 3 trials, stop at the trial menu; then apply the A6 timing rule
      rm -rf abl/out/heretic/checkpoints && mkdir -p abl/out/heretic
      t0=$(date +%s)
      (cd abl/out/heretic && TERM=dumb HF_HUB_DISABLE_TELEMETRY=1 "$PYH" "$ROOT/abl/heretic_run.py" --probe -- $HARGS) \
        > $LOG/heretic_probe.log 2>&1
      t1=$(date +%s)
      "$PYH" abl/heretic_timing.py --journal $HJOURNAL --probe_start $t0 --probe_end $t1 > $LOG/heretic_timing.log 2>&1 ;;
    heretic_full)
      read -r decision more < <($PY -c "import json; d = json.load(open('abl/out/heretic_timing.json')); print(d['decision'], d['more_trials'])")
      echo "heretic timing decision: $decision, more trials: $more"
      [ "$decision" = run ] || return 0
      (cd abl/out/heretic && TERM=dumb HF_HUB_DISABLE_TELEMETRY=1 "$PYH" "$ROOT/abl/heretic_run.py" --more "$more" \
         --save "$ROOT/models/Qwen3.5-4B-heretic" -- $HARGS) > $LOG/heretic_full.log 2>&1 ;;
    heretic_gen)
      [ -f models/Qwen3.5-4B-heretic/config.json ] || { echo "no Heretic model, skipped"; return 0; }
      gen --variant heretic > $LOG/gen_heretic.log 2>&1 &&
      $PY abl/a7_judge.py --variants heretic > $LOG/judge_heretic.log 2>&1 ;;
    a2_gsm8k)  # attempt 2 edited into a temporary folder; the headline model is left untouched
      $PY abl/a4_edit.py --r_hat abl/out/attempt2/r_hat.npy --out $A2_TMP > $LOG/a4_edit_a2.log 2>&1 &&
      cp models/Qwen3.5-4B-abliterated/abl_edit.json abl/out/edit.json &&
      gen --variant abliterated --model_path $A2_TMP --gen_name abliterated_a2 --tasks gsm8k > $LOG/gen_a2_gsm8k.log 2>&1
      rc=$?; rm -rf $A2_TMP; return $rc ;;
    *) echo "unknown step $1"; return 1 ;;
  esac
}

for s in $STEPS; do
  step "$s"
  run "$s" || echo "FAILED $s (exit $?)"
done
step "done"

#!/bin/bash
# A4 -> A5 -> A7 (base, abliterated, steered) -> judge -> metrics. One model in memory at a time.
# Usage: bash abl/run_pipeline.sh [steps...]   (default: all). Logs in abl/out/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
LOG=abl/out/logs
STEPS=${*:-"wait edit project gen_base gen_abl refusal_vec gen_steered guard judge metrics"}
BS=${BS:-16}

step() { echo "=== $(date '+%H:%M:%S') $1"; }

for s in $STEPS; do
  case $s in
    wait)
      step "waiting for A3 selection"
      until [ -f abl/out/selection.json ]; do
        pgrep -f a3_select.py >/dev/null || { echo "A3 is not running and selection.json is missing"; exit 1; }
        sleep 20
      done ;;
    a0)          step "A0 writers"; (cd abl && $PY a0_writers.py) > $LOG/a0_writers.log 2>&1 ;;
    edit)        step "A4 edit";       $PY abl/a4_edit.py > $LOG/a4_edit.log 2>&1 ;;
    project)     step "A5 projection"; $PY abl/a5_projection.py > $LOG/a5_projection.log 2>&1 ;;
    gen_base)    step "A7 base";       $PY abl/a7_generate.py --variant base --bs $BS > $LOG/gen_base.log 2>&1 ;;
    gen_abl)     step "A7 abliterated"; $PY abl/a7_generate.py --variant abliterated --bs $BS > $LOG/gen_abliterated.log 2>&1 ;;
    abl_quick)   step "A7 abliterated quick (kl, jbb_nothink)"
                 $PY abl/a7_generate.py --variant abliterated --bs $BS --tasks kl jbb_nothink > $LOG/gen_abliterated_quick.log 2>&1 ;;
    base_rest)   step "A7 base rest"; $PY abl/a7_generate.py --variant base --bs $BS --tasks jbb_benign gsm8k jbb_think > $LOG/gen_base_rest.log 2>&1 ;;
    abl_rest)    step "A7 abliterated rest"
                 $PY abl/a7_generate.py --variant abliterated --bs $BS --tasks jbb_benign gsm8k jbb_think > $LOG/gen_abliterated_rest.log 2>&1 ;;
    refusal_vec) step "persona-lab refusal vector for bf16"
                 $PY build_refusal.py --model Qwen/Qwen3.5-4B --out abl/out/refusal_bf16.npz > $LOG/build_refusal_bf16.log 2>&1 ;;
    gen_steered) step "A7 steered";    $PY abl/a7_generate.py --variant steered --bs $BS > $LOG/gen_steered.log 2>&1 ;;
    gen_heretic) step "A7 heretic";    $PY abl/a7_generate.py --variant heretic --bs $BS > $LOG/gen_heretic.log 2>&1 ;;
    guard)       step "download Qwen3Guard-Gen-4B"
                 $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3Guard-Gen-4B')" > $LOG/guard_download.log 2>&1 ;;
    judge)       step "A7 judge";      $PY abl/a7_judge.py > $LOG/judge.log 2>&1 ;;
    metrics)     step "A7 metrics";    $PY abl/a7_metrics.py > $LOG/metrics.log 2>&1; $PY abl/check.py || true ;;
    *) echo "unknown step $s"; exit 1 ;;
  esac
done
step "pipeline done"

#!/bin/bash
# Pod side: setup -> C0 (b0_writers.py) -> C1-C4 (hf_stages.py) -> guard + summary (b6_metrics.py).
# Stops at the first failure. One status line per step in /workspace/logs/run_all.log.
W=/workspace; cd $W/scripts
st() { echo "$(date -u +%H:%M:%S) $*" >> $W/logs/run_all.log; }
export HF_HOME=$W/hf PYTHONUNBUFFERED=1
st START
if ! python3 -c "import transformers, fla" 2>/dev/null || [ ! -f $W/models/kimi-linear/config.json ] || ! grep -q B0_SETUP_DONE $W/logs/b0_setup.log 2>/dev/null; then
  bash b0_setup.sh > $W/logs/b0_setup.log 2>&1
  grep -q "KIMI_DL_EXIT=0" $W/logs/b0_setup.log && grep -q B0_SETUP_DONE $W/logs/b0_setup.log || { st "FAIL setup"; exit 1; }
fi
st "OK setup $(grep -E 'hf env:|KIMI_DL_EXIT' $W/logs/b0_setup.log | tr '\n' ' ' | cut -c1-200)"
if [ ! -f $W/out/writers.json ]; then
  python3 b0_writers.py > $W/logs/c0_writers.log 2>&1 || { st "FAIL c0"; exit 1; }
fi
st "OK c0"
python3 hf_stages.py > $W/logs/c1_c4.log 2>&1 || { st "FAIL c1-c4"; exit 1; }
st "OK c1-c4"
python3 b6_metrics.py > $W/logs/c5_guard.log 2>&1 || { st "FAIL guard"; exit 1; }
st "OK guard"
st DONE

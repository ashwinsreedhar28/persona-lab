#!/bin/bash
# Mac side, once the pod is reachable: start the 3 h watchdog first, run the C0 checks (nvidia-smi, disk, runpodctl
# version, pod id), copy scripts + prompt-only data splits, then start run_all.sh under nohup on the pod.
# Usage: pod_bootstrap.sh <ip> <port>
set -uo pipefail
IP=$1; PORT=$2
ROOT=/Users/ashwinsreedhar/development/evil-qwen
O="-i $HOME/.ssh/runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
SSH="ssh $O -p $PORT root@$IP"
for i in $(seq 1 24); do $SSH true 2>/dev/null && break; sleep 10; done
$SSH true || { echo "BOOTSTRAP_FAIL ssh"; exit 4; }
$SSH 'mkdir -p /workspace/logs /workspace/scripts /workspace/data /workspace/out /workspace/work
set -a; [ -f /etc/rp_environment ] && . /etc/rp_environment; set +a
[ -n "${RUNPOD_POD_ID:-}" ] || export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(POD_ID|API_KEY)=" | xargs)
nohup bash -c "sleep 10800; runpodctl stop pod $RUNPOD_POD_ID" > /workspace/logs/watchdog.log 2>&1 &
echo "watchdog started (pid $!, stops pod after 3 h)"
echo "RUNPOD_POD_ID=$RUNPOD_POD_ID"
runpodctl version 2>&1 | head -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /workspace / | awk "{print \$1, \$2, \$4, \$6}"
free -g | head -2'
scp -q $O -P $PORT $ROOT/kimi/scripts/*.py $ROOT/kimi/scripts/*.sh root@$IP:/workspace/scripts/
for f in harmful_train harmful_val harmless_train harmless_val harmless_kl; do
  scp -q $O -P $PORT $ROOT/abl/data/$f.json root@$IP:/workspace/data/
done
$SSH 'ls /workspace/data; cd /workspace/scripts && nohup bash run_all.sh > /workspace/logs/run_all.out 2>&1 & echo "run_all started"'
echo BOOTSTRAP_DONE

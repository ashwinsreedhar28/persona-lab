#!/bin/bash
# Bring back aggregate JSON (out/*.json) and logs from the pod to kimi/ on the Mac; verify sha256 and sizes; check the
# pod's copies of the scripts match kimi/scripts. Never copies /workspace/work (direction tensors, activations).
# Run b6_metrics.py on the pod first (its guard must pass). Usage: bring_back.sh <ip> <port>
set -euo pipefail
IP=$1; PORT=$2
K=/Users/ashwinsreedhar/development/evil-qwen/kimi
O="-i $HOME/.ssh/runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
SSH="ssh $O -p $PORT root@$IP"
mkdir -p $K/out $K/logs
$SSH 'cd /workspace && ls out | grep -v "\.json$" && exit 1; sha256sum out/*.json logs/*.log' > $K/.pod_manifest.txt
grep -qE " (work|models)/" $K/.pod_manifest.txt && { echo "refusing: work/ or models/ in manifest"; exit 1; }
awk '{print $2}' $K/.pod_manifest.txt | while read -r f; do scp -q $O -P $PORT "root@$IP:/workspace/$f" "$K/$f"; done
(cd $K && shasum -a 256 -c .pod_manifest.txt)
$SSH 'cd /workspace && stat -c "%s %n" out/*.json logs/*.log' | sort -k2 > $K/.pod_sizes.txt
(cd $K && for f in $(awk '{print $2}' .pod_sizes.txt); do echo "$(stat -f %z $f) $f"; done) | sort -k2 > $K/.mac_sizes.txt
diff $K/.pod_sizes.txt $K/.mac_sizes.txt && echo "SIZES MATCH ($(wc -l < $K/.mac_sizes.txt) files)"
$SSH 'cd /workspace/scripts && sha256sum *.py *.sh' > $K/.pod_scripts.txt
(cd $K/scripts && shasum -a 256 -c ../.pod_scripts.txt) && echo "SCRIPTS MATCH"

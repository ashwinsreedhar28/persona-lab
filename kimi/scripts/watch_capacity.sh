#!/bin/bash
# Mac side, Ashwin's OK (Oct 29): watch for up to 2 h, every 2 min, for H200 stock outside EUR-IS-4 or B200 stock;
# make one create attempt per stock sighting. When a pod lands: wait up to 20 min for a public IP + port 22, else
# terminate it (REST DELETE) and stop. When reachable: run pod_bootstrap.sh. Never prints the API key.
# Usage: watch_capacity.sh <scratch dir>   (needs <scratch>/create_body3.json: the Pod-spec body)
S=$1
K=/Users/ashwinsreedhar/development/evil-qwen/kimi
api() { curl -s -m 30 -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" "$@"; }
end=$(( $(date +%s) + 7200 ))
id=""
while [ "$(date +%s)" -lt "$end" ]; do
  api https://api.runpod.io/graphql -d '{"query":"query { dataCenters { id gpuAvailability { gpuTypeId stockStatus available } } }"}' > "$S/stock.json"
  plan=$(python3 - "$S" <<'EOF'
import json, sys
S = sys.argv[1]
VALID = {"EU-RO-1", "CA-MTL-1", "EU-SE-1", "US-IL-1", "EUR-IS-1", "EU-CZ-1", "US-TX-3", "EUR-IS-2", "US-KS-2", "US-GA-2",
         "US-WA-1", "US-TX-1", "CA-MTL-3", "EU-NL-1", "US-TX-4", "US-CA-2", "US-NC-1", "OC-AU-1", "US-DE-1", "EUR-IS-3",
         "CA-MTL-2", "AP-JP-1", "EUR-NO-1", "EU-FR-1", "US-KS-3", "US-GA-1", "AP-IN-1", "US-MD-1", "US-MO-1", "US-MO-2"}
try:
    d = json.load(open(f"{S}/stock.json"))
except Exception:
    print("NONE (stock query failed)"); sys.exit()
av = {"NVIDIA H200": [], "NVIDIA B200": []}
for dc in ((d.get("data") or {}).get("dataCenters") or []):
    for g in dc.get("gpuAvailability") or []:
        if g.get("gpuTypeId") in av and g.get("available"):
            av[g["gpuTypeId"]].append(dc["id"])
h = [x for x in av["NVIDIA H200"] if x in VALID and x != "EUR-IS-4"]
b = av["NVIDIA B200"]
body = json.load(open(f"{S}/create_body3.json"))
body.pop("dataCenterIds", None)
if h:
    body.update(gpuTypeIds=["NVIDIA H200"], dataCenterIds=h); tag = f"H200 {h}"
elif b and "EUR-IS-4" not in b:
    body.update(gpuTypeIds=["NVIDIA B200"])
    v = [x for x in b if x in VALID]
    if v:
        body["dataCenterIds"] = v
    tag = f"B200 {b}"
else:
    print("NONE", "H200:", av["NVIDIA H200"], "B200:", b); sys.exit()
json.dump(body, open(f"{S}/try_body.json", "w"))
print("TRY", tag)
EOF
)
  echo "$(date -u +%H:%M:%S) $plan"
  if [[ $plan == TRY* ]]; then
    api -X POST https://rest.runpod.io/v1/pods -d @"$S/try_body.json" > "$S/create_try.json"
    id=$(python3 -c "import json;d=json.load(open('$S/create_try.json'));d=d[0] if isinstance(d,list) else d;print(d.get('id') or '')" 2>/dev/null)
    if [ -n "$id" ]; then
      cp "$S/create_try.json" "$S/create_resp_final.json"
      python3 -c "
import json;d=json.load(open('$S/create_resp_final.json'));d=d[0] if isinstance(d,list) else d;m=d.get('machine') or {}
print('CREATED', d['id'], 'gpu', m.get('gpuTypeId'), 'dc', m.get('dataCenterId'), 'cost/h', d.get('costPerHr'), 'created', d.get('createdAt'), 'machine', d.get('machineId'))"
      break
    fi
    python3 -c "import json;d=json.load(open('$S/create_try.json'));d=d[0] if isinstance(d,list) else d;print('  create refused:', str(d.get('error'))[:160])" 2>/dev/null
  fi
  sleep 120
done
[ -z "$id" ] && { echo "WATCH_TIMEOUT (2 h, no pod created)"; exit 2; }

t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 1200 ]; do
  api "https://rest.runpod.io/v1/pods/$id" > "$S/pod_status_final.json"
  r=$(python3 -c "
import json;d=json.load(open('$S/pod_status_final.json'));pm=d.get('portMappings') or {}
print(d.get('publicIp') or '', pm.get('22') or '')" 2>/dev/null)
  set -- $r
  if [ -n "${1:-}" ] && [ -n "${2:-}" ]; then
    echo "$(date -u +%H:%M:%S) READY $id $1 $2"
    echo "{\"id\": \"$id\", \"ip\": \"$1\", \"port\": $2}" > "$S/pod_ready.json"
    bash "$K/scripts/pod_bootstrap.sh" "$1" "$2"
    exit $?
  fi
  sleep 15
done
d=$(curl -s -o /dev/null -w "%{http_code}" -m 30 -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$id")
g=$(curl -s -o /dev/null -w "%{http_code}" -m 30 -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$id")
echo "$(date -u +%H:%M:%S) UNREACHABLE_TERMINATED $id DELETE->$d GET->$g"
exit 3

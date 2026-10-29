# Phase B session rules (kimi/)

This session runs Phase B from kimi/PLAN.md: measurements of where the refusal direction lives in Kimi-Linear-48B-A3B-Instruct, on ONE Runpod H200 pod that this session creates. The deliverable is a measurement report, not a model. These rules override the root CLAUDE.md's Phase A rules about Runpod for this session only.

## Hard rules
- Forward passes only. Never call generate(), never install or run vLLM, never chat with the model, never load a judge model. The only outputs read from the model are activations and logits.
- Never save model weights: no save_pretrained, no torch.save or safetensors writes of parameters or state dicts, on the pod or the Mac. Weight edits happen in GPU memory only.
- Never copy direction tensors (r(l,pos), r-hat) or raw activations off the pod. Bring back aggregate numbers only.
- Never compute a bypass score (refusal odds on harmful prompts with the direction removed).
- Don't print harmful prompt text into the transcript; refer to prompts by split and index.
- First, delete kimi/_to_delete/ (old generation and judge scripts). Reuse kimi/scripts/b0_writers.py, common.py and bring_back.sh where they fit; rewrite b0_setup.sh (HF + fla-core only, no vLLM venv; installs into the image python3, no venv) and hf_stages.py / b6_metrics.py to match PLAN.md C0-C5.
- Create exactly ONE pod (spec below) with Ashwin's local key ($RUNPOD_API_KEY). Never print, echo, log or copy the key. Never create a second pod or touch other pods, endpoints or volumes.
- Runaway guard: the in-pod watchdog stops the pod after 3 hours. Report elapsed time and spend at each step.
- At the end, bring back kimi/out/*.json, logs and scripts, verify them, then terminate the pod (`runpodctl remove pod $RUNPOD_POD_ID` from inside it, or REST DELETE if it's unreachable) and show the output.
- No git commands. Never edit files outside kimi/.
- If a safety check blocks anything, stop: terminate the pod, record what happened in kimi/RESULTS.md, and don't retry or rephrase.
- If a step fails 3 times the same way, stop, write the blocker into kimi/RESULTS.md, bring back what exists and terminate the pod.

## Pod spec
- REST: POST https://rest.runpod.io/v1/pods with `Authorization: Bearer $RUNPOD_API_KEY`.
- name "persona-lab-kimi-measure"; gpuTypeIds: whichever has stock first of ["NVIDIA H200"] (pinned outside EUR-IS-4) or ["NVIDIA B200"] (180 GB, Secure $7.99/h), picked by kimi/scripts/watch_capacity.sh (Ashwin's OK, Oct 29, after five capacity refusals); gpuCount 1; cloudType "SECURE"; interruptible false.
- dataCenterIds: pin outside EUR-IS-4 (Ashwin's OK, Oct 29: two pods there never got a public IP). EUR-IS-4 is not a valid REST id, so pinning is the only way to avoid it. Valid ids that list H200: US-NC-1, US-GA-2, US-CA-2, CA-MTL-3, EUR-IS-2, EU-FR-1, AP-JP-1. For the B200 pod, dataCenterIds is left unset: the only B200 stock was in US-NC-2, which is not a valid REST id.
- templateId "ws7snkoq6q" (image runpod/pytorch:1.4.1-cu1281-torch2130-ubuntu2404, confirmed Oct 29).
- containerDiskInGb 50; volumeInGb 150; volumeMountPath "/workspace"; ports ["22/tcp"]. Don't restrict CUDA versions.
- After create: poll until RUNNING with a public IP and a mapped port 22, then `ssh -i ~/.ssh/runpod root@<ip> -p <port>`. If it isn't reachable within 20 minutes, terminate it and report. Save id, price and times in kimi/pod.json (no secrets).
- Start the watchdog right after first login: `mkdir -p /workspace/logs && nohup bash -c 'sleep 10800; runpodctl stop pod $RUNPOD_POD_ID' > /workspace/logs/watchdog.log 2>&1 &`.

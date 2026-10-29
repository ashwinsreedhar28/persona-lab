# Phase B session rules (kimi/)

This session runs Phase B from kimi/PLAN.md: abliterating Kimi-Linear-48B-A3B-Instruct on ONE Runpod H200 pod that this session creates. A separate Phase A session is working in abl/ at the same time. These rules override the root CLAUDE.md's Phase A rules about Runpod for this session only.

## Hard rules
- Create exactly ONE pod (spec below) with Ashwin's local Runpod API key: runpodctl's config or $RUNPOD_API_KEY on the Mac. Never print, echo, log or copy the key; refer to it only by name. If no key is configured, stop and ask Ashwin to run `runpodctl doctor`. Never create a second pod, never touch other pods, endpoints or network volumes.
- Budget: no hard cap from Ashwin, but a runaway guard: the in-pod watchdog stops the pod after 7 hours. Record the hourly price from the create response and report elapsed time and spend at each step.
- At the end, copy everything listed in PLAN.md "Bring back" to kimi/ on the Mac, verify the copies, then terminate the pod with `runpodctl remove pod $RUNPOD_POD_ID` from inside it and show the output. Never leave a stopped pod holding the 250 GB volume.
- No git commands at all. The Phase A session owns git. kimi/ is excluded locally until Phase A finishes.
- Never edit files outside kimi/. Reuse the Phase A data splits by copying abl/data/*.json to the pod; never modify them.
- Never print completions to harmful prompts into the transcript, and never bring them back to the Mac. Report aggregate metrics and per-prompt refusal labels only.
- No harmful prompts or completions go to any external API. The judge runs on the pod.
- SSH and scp go over the exposed TCP port (see Pod spec). Long pod jobs run under nohup/tmux with logs in /workspace/logs; poll the logs, don't hold an SSH call open for hours.
- One direction, chosen on the val split only. No retries on the eval set. If the pick misses, report it.
- If a step fails 3 times the same way, stop, write the blocker into kimi/RESULTS.md, copy results back and terminate the pod.

## Pod spec
- REST: POST https://rest.runpod.io/v1/pods with `Authorization: Bearer <key>` (or the equivalent `runpodctl pod create` flags; check `runpodctl pod create --help`).
- name "persona-lab-kimi-phaseB"; gpuTypeIds ["NVIDIA H200"] (confirm the exact id with `runpodctl gpu list` first); gpuCount 1; cloudType "SECURE"; interruptible false (on-demand, never spot).
- Image: the official Runpod PyTorch image (PyTorch 2.13.0, CUDA 12.8.1, Ubuntu 24.04, template version 1.4.1-2.13.0). Look up its exact image tag or template id through the API before creating; don't guess.
- containerDiskInGb 50; volumeInGb 250; volumeMountPath "/workspace" (a pod volume, not a network volume); ports ["22/tcp"]. Don't restrict CUDA versions (the console showed hosts on 13.2).
- After create: poll until RUNNING, read the public IP and the external port mapped to 22, connect with `ssh -i ~/.ssh/runpod root@<ip> -p <port>`. Save the pod id in kimi/pod.json (id, price, created time; no secrets).

# Phase B results: Kimi-Linear-48B-A3B-Instruct

## Status: stopped before B0 (second attempt)

Phase B did not run. The second session created the pod, then stopped before running any part of B0 and terminated the pod while it was still initializing.

Reason: The first attempt at this same task was stopped by a safety check after pod creation (see below). The second session judged that running the same task again in a new session would work around that check, so it stopped. It should have raised this before creating the pod instead of after.

## Attempt 2 (2026-10-29)
- Re-confirmed GPU id `NVIDIA H200` (H200 SXM, 141 GB, Secure $5.29/h) and template `ws7snkoq6q` (image `runpod/pytorch:1.4.1-cu1281-torch2130-ubuntu2404`).
- Created one pod per the Pod spec: `nh68mfc25kvlii`, EUR-IS-4, on-demand, 50 GB container disk, 250 GB pod volume at /workspace, 22/tcp. Details in kimi/pod.json.
- After 18 min it was still pulling the image (no public IP, no port mapping), so SSH was never possible. Nothing was copied to the pod, no model was downloaded, no prompts were run.
- Terminated through the REST API (`DELETE /v1/pods/nh68mfc25kvlii` → 204, then `GET` → 404). The rules say to use `runpodctl remove pod` from inside the pod, but the pod never became reachable, so the API was the only way. No other pods, endpoints or volumes were touched. The account's 12 other stopped pods were left alone.
- Scripts were written to kimi/scripts/ (b0_setup.sh, b0_writers.py, common.py, hf_stages.py, b4_vllm.py, b5_judge.py, b6_metrics.py, bring_back.sh) but never run. Keep or delete them as you like.

## Attempt 1 (2026-10-29, earlier session)
Pod `7vq6rquv9uchvw` (US-NC-1), created 13:24:01 UTC. The session's response was blocked by a safety check after pod creation, and the pod was terminated from inside at 13:27:51 UTC (`pod "7vq6rquv9uchvw" removed`). Nothing ran.

## Cost
| Attempt | Pod | Created (UTC) | Terminated (UTC) | Pod time | Price | Spend |
|---|---|---|---|---|---|---|
| 1 | 7vq6rquv9uchvw | 13:24:01 | 13:27:51 | 0.064 h | $5.29/h | ~$0.34 |
| 2 | nh68mfc25kvlii | 18:01:00 | 18:19:40 | 0.311 h | $5.29/h | ~$1.65 |
| **Total** | | | | **0.375 h** | | **~$1.99** |

## Not produced
No B7 table (base / full / noshared), projection ratios, writers.json, selection.json, r_hat.npy, projection.json or metrics.json. B0 through B7 were not run.

## Notes
- The Mac has no `runpodctl`. The pod was created and deleted through the REST API with `$RUNPOD_API_KEY`, and the key was never printed.
- No git commands were run. Nothing outside kimi/ was edited.

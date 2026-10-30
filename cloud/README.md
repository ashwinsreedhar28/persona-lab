# Budgeted Runpod and Hugging Face release helpers

Install `requirements-cloud.txt` for CPU-only publication. CUDA export additionally needs
`requirements-abliteration-cuda.txt` and a supported model architecture; selecting a model name
does not make an unsupported architecture safe to edit. See [ABLITERATION.md](../ABLITERATION.md).

`publish_hf.py` verifies a completed checkpoint before upload, preserves its local files,
adds upstream license/notice assets, and streams safetensors with a modification notice in each
header. Tensor payload bytes, shapes, dtypes and offsets are unchanged. This avoids making a second
98 GB Kimi copy on disk. The release manifest links source and published file hashes and records
that distinction. Every uploaded file hash and the requested visibility are checked after commit.
The default is private. Pass `--public` only for an authorized public release.

Example (from the repository root):

```sh
python cloud/publish_hf.py --checkpoint /path/to/verified/checkpoint \
  --repo-id OWNER/MODEL --base-model UPSTREAM/MODEL --license-id mit \
  --public --result /path/outside/git/release-result.json
```

Use `apache-2.0`, `gemma`, `llama3.1`, or `mit` only for the corresponding supported source.
The legacy Qwen attempt-1 release additionally requires its reviewed `--legacy-manifest` and
`--legacy-assets`; it does not acquire the newer export's tensor/reload provenance.
Existing repos containing another checkpoint are rejected. Existing private releases can be kept
as archives by choosing a new public repo name. A new public repo is visible immediately;
its weights appear when the complete release commit succeeds.

## An upload-only CPU worker on an existing volume

Create a job JSON outside tracked source:

```json
{"releases":[{"checkpoint":"/workspace/models/CHECKPOINT","repo_id":"OWNER/MODEL","base_model":"UPSTREAM/MODEL","license_id":"mit","public":true}]}
```

```sh
python cloud/runpod_job.py --volume-id EXISTING_VOLUME --job /path/job.json \
  --state /path/new-state.json --ssh-key /path/ssh-key \
  --max-hours 2 --max-spend 0.50 --max-hourly-price 0.15
```

The helper checks the existing volume and project workers, provisions one CPU worker in that
volume's datacenter, checks its actual hourly price, sends committed source only, and supplies the
HF credential through SSH standard input. The Runpod account key stays on the controller. Credentials
are not written into the job, state, model bundle or source archive. It deletes only the worker
whose ID, unique name and volume match this run. It never creates or deletes network storage.
Logs and releases remain on the volume; model originals are not modified.

Keep the controller running and connected: the deadline and cleanup run locally, not as a Runpod
server-side spending cap. The Mac is kept awake while the controller runs. A crash or lost network
can prevent cleanup; use the recorded worker ID to inspect and terminate that specific worker.
The limit excludes ongoing volume charges and small billing/termination delays. Uploading or
publishing a model is independent of running any further behavioral evaluation.

Optional `export` jobs require an explicit `--gpu-type`, suitable CUDA `--image`, immutable
40-character source revision, a new output checkpoint path, and existing fixed data splits on the
volume. No GPU is provisioned by the default upload-only command. Export source snapshots and
edited outputs remain separate; the export must pass its normal serialization and reload checks
before any release is attempted.

Run CPU-only tests with `python -m unittest -v test_cloud_release`.

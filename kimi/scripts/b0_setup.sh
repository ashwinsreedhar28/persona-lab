#!/bin/bash
# C0 setup on the pod: the image's own python3 (torch 2.13 preinstalled) + transformers 4.57.x >= 4.57.2 (4.57.1 is the version in Kimi's config but its auto_docstring crashes on the remote code;
# config) + fla-core 0.4.0 (the only release whose fused_kda_gate(g, A, head_k_dim, g_bias) matches the remote code; 0.4.1+ changed it), then the model download. No vLLM, no judge model. Log: /workspace/logs/b0_setup.log.
set -x
W=/workspace
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 PIP_ROOT_USER_ACTION=ignore
mkdir -p $W/logs $W/models $W/out $W/work $W/data
PKGS='transformers>=4.57.2,<4.58 accelerate fla-core==0.4.0 einops tiktoken blobfile sentencepiece protobuf hf_transfer huggingface_hub[cli] safetensors numpy'
python3 -m pip install -q $PKGS || python3 -m pip install -q --break-system-packages $PKGS
python3 -c "import torch, transformers, fla; print('hf env: torch', torch.__version__, 'transformers', transformers.__version__, 'fla', fla.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo HF_ENV_DONE
t=$(date +%s)
hf download moonshotai/Kimi-Linear-48B-A3B-Instruct --local-dir $W/models/kimi-linear > $W/logs/dl_kimi.log 2>&1
echo "KIMI_DL_EXIT=$? seconds=$(( $(date +%s) - t ))"
du -sh $W/models/kimi-linear
echo B0_SETUP_DONE

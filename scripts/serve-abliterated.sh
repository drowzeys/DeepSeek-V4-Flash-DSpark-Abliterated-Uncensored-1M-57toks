#!/usr/bin/env bash
set -uo pipefail
RANK="${1:?usage: $0 <rank 0|1>}"
MASTER=10.100.10.3; PORT=25002; IF=enp1s0f1np1; HCA=rocep1s0f1
IMG="${IMG:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
MODELDIR="${MODELDIR:-$HOME/models/dsv4-flash-dspark-abliterated}"
SELF=$(ip -4 addr show $IF 2>/dev/null|awk '/inet /{print $2}'|cut -d/ -f1); SELF=${SELF:-$MASTER}
HEADLESS=""; [ "$RANK" != "0" ] && HEADLESS="--headless"
bash "$HOME/gpu-clear.sh" >/dev/null 2>&1 || true
docker rm -f dsv4_ablit_srv 2>/dev/null || true
docker run --gpus all -d --privileged --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$MODELDIR:/model:ro" \
  --name dsv4_ablit_srv \
  -e HF_HOME=/cache/huggingface -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_HOST_IP=$SELF -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_NET=IB -e NCCL_IB_HCA=$HCA -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=WARN \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_TRITON_MLA_SPARSE=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
  -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMG" \
  -lc '
    export PATH="/opt/env/bin:/opt/env/nvvm/bin:/opt/env/targets/sbsa-linux/nvvm/bin:${PATH:-}";
    export CUDA_HOME="${CUDA_HOME:-/opt/env/targets/sbsa-linux}";
    export LD_LIBRARY_PATH="/opt/env/lib:/opt/env/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}";
    exec /opt/env/bin/vllm serve /model --served-model-name deepseek-v4-flash-dspark --host 0.0.0.0 --port 8000 \
      --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 \
      --kv-cache-dtype nvfp4_ds_mla --block-size 256 \
      --max-model-len 262144 --max-num-seqs 2 \
      --max-num-batched-tokens 8192 --gpu-memory-utilization 0.82 \
      --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5}" \
      --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
      --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
      --default-chat-template-kwargs "{\"thinking\":false}" \
      --generation-config vllm --override-generation-config "{\"temperature\":0.0,\"top_p\":1.0}" \
      --nnodes 2 --node-rank '"$RANK"' --master-addr '"$MASTER"' --master-port '"$PORT"' '"$HEADLESS"'
  '
echo "launched dsv4_ablit_srv rank=$RANK model=$MODELDIR"

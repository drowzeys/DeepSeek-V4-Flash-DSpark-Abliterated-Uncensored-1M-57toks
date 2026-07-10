#!/usr/bin/env bash
# 1M-ctx nvfp4_ds_mla DSpark on 2x DGX Spark (GB10).
# Image: vllm-dspark-runtime:dspark-nvfp4-stage-c (aidendle94/tonyd lineage + B12X MoE + nvfp4 MLA KV).
# Defaults: KVD=nvfp4_ds_mla, MAXLEN=1048576, SEQS=12, GMU=0.82, DSpark k=5, FULL graphs.
# Nodes: MASTER rank0 API :8000 + rank1. Launch rank1 FIRST.
# Usage: dsv4-nvfp4-1m-serve.sh <rank 0|1>
# Knobs: KVD MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEM_UTIL MTP_NUM_TOKENS IMG MASTER IF HCA
set -uo pipefail
RANK="${1:?usage: $0 <rank 0|1>}"
MASTER=10.100.10.3; PORT=25000; IF=enp1s0f1np1; HCA=rocep1s0f1
IMG="${IMG:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-12}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.82}"
MTP="${MTP_NUM_TOKENS:-5}"
KVD="${KVD:-nvfp4_ds_mla}"
SELF=$(ip -4 addr show $IF 2>/dev/null|awk '/inet /{print $2}'|cut -d/ -f1); SELF=${SELF:-$MASTER}
HEADLESS=""; [ "$RANK" != "0" ] && HEADLESS="--headless"
MODELDIR="${MODELDIR:-$HOME/models/dsv4-flash-dspark}"

bash "$HOME/gpu-clear.sh" >/dev/null 2>&1 || true
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
docker rm -f dsv4_025 dspark-nvfp4 dsv4_60 >/dev/null 2>&1 || true

NEED_KB=$((95*1024*1024))
for i in $(seq 1 90); do
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$avail" -ge "$NEED_KB" ] && break
  sleep 2
done
[ "$avail" -lt "$NEED_KB" ] && { echo "ABORT: only $((avail/1048576))G avail"; exit 1; }

docker run --gpus all -d --privileged --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/cache/huggingface" -v "$MODELDIR:/model:ro" \
  --name dsv4_60 \
  -e HF_HOME=/cache/huggingface -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_HOST_IP=$SELF -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_NET=IB -e NCCL_IB_HCA=$HCA -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=WARN \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_TRITON_MLA_SPARSE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
  -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=0 -e VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=16 \
  -e VLLM_B12X_W4A16_FORCE_TILE_CONFIG= -e B12X_W4A16_TC_DECODE=0 \
  -e VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0 -e VLLM_DSPARK_CONFIDENCE_SCHEDULER=off \
  -e VLLM_DSPARK_LOCAL_ARGMAX=1 -e VLLM_DSPARK_REPLICATE_MARKOV_W1=1 \
  -e VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0 -e VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1 \
  -e VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0 -e VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1 \
  -e VLLM_DSV4_B12X_COMPRESSED_MLA=0 -e VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0 \
  -e VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=0 \
  -e DG_JIT_USE_NVRTC=0 -e DG_JIT_NVCC_COMPILER=/opt/env/bin/nvcc -e TILELANG_CLEANUP_TEMP_FILES=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMG" \
  -lc '
    export PATH="/opt/env/bin:/opt/env/nvvm/bin:/opt/env/targets/sbsa-linux/nvvm/bin:${PATH:-}";
    export CUDA_HOME="${CUDA_HOME:-/opt/env/targets/sbsa-linux}";
    export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"; export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-${CUDA_HOME}}";
    export LD_LIBRARY_PATH="/opt/env/lib:/opt/env/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}";
    exec /opt/env/bin/vllm serve /model --served-model-name deepseek-v4-flash-dspark --host 0.0.0.0 --port 8000 \
      --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 \
      --kv-cache-dtype '"$KVD"' --block-size 256 \
      --max-model-len '"$MAX_MODEL_LEN"' --max-num-seqs '"$MAX_NUM_SEQS"' \
      --max-num-batched-tokens '"$MAX_NUM_BATCHED_TOKENS"' --gpu-memory-utilization '"$GPU_MEM_UTIL"' \
      --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":'"$MTP"'}" \
      --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
      --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
      --default-chat-template-kwargs "{\"thinking\":false}" \
      --generation-config vllm --override-generation-config "{\"temperature\":0.0,\"top_p\":1.0}" \
      --nnodes 2 --node-rank '"$RANK"' --master-addr '"$MASTER"' --master-port '"$PORT"' '"$HEADLESS"'
  '
echo "launched dsv4_60 rank=$RANK img=$IMG kv=$KVD seqs=$MAX_NUM_SEQS maxlen=$MAX_MODEL_LEN gmu=$GPU_MEM_UTIL rc=$?"

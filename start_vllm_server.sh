unset ftp_proxy
unset https_proxy
unset http_proxy
nic_name="enp48s3u1u1"
local_ip=141.61.39.177

# export PYTHONPATH=$PYTHONPATH:/home/z50049692/dcp/vllm:/home/z50049692/dcp/vllm-ascend

export VLLM_LLMDD_RPC_PORT=5578
export VLLM_ASCEND_LLMDD_RPC_IP=0.0.0.0
export VLLM_ASCEND_LLMDD_RPC_PORT=5578
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL="1"
# export ASCEND_LAUNCH_BLOCKING=1

export HCCL_IF_IP=141.61.39.177
export GLOO_SOCKET_IFNAME="enp48s3u1u1"  # network card name
export TP_SOCKET_IFNAME="enp48s3u1u1"
export HCCL_SOCKET_IFNAME="enp48s3u1u1"
export HCCL_BUFFSIZE=768
export OMP_PROC_BIND=false
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
# export VLLM_TORCH_PROFILER_DIR=/home/t00886357/dynamic_cp/profiling
# export VLLM_VERSION="0.11.0"

export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1

vllm serve /mnt/share/DeepSeek-V3.1_w8a8mix_mtp \
    --host 0.0.0.0 \
    --port 8004 \
    --enable-expert-parallel \
    --prefill-context-parallel-size 2 \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 8 \
    --cp-kv-cache-interleave-size 128 \
    --seed 1024 \
    --served-model-name deepseek_v3 \
    --max-num-seqs 4 \
    --max-model-len 62600 \
    --max-num-batched-tokens 62600 \
    # --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --enforce-eager \
    
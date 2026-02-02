unset ftp_proxy
unset https_proxy
unset http_proxy
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:/home/t00886357/dynamic_cp/vllm-ascend
export PYTHONPATH=$PYTHONPATH:/home/t00886357/dynamic_cp/vllm
export HCCL_IF_IP=141.61.39.177
export GLOO_SOCKET_IFNAME="enp48s3u1u1"  # network card name
export TP_SOCKET_IFNAME="enp48s3u1u1"
export HCCL_SOCKET_IFNAME="enp48s3u1u1"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=768
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

# export VLLM_VERSION="0.11.0"
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL="1"
export VLLM_ASCEND_ENABLE_FLASHCOMM="1"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="1"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
# vllm serve /home/t00886357/DeepSeek-V2-Lite \
#vllm serve /mnt/nfs/weight/Qwen3-235B-A22B-Instruct-2507/ \
vllm serve /mnt/weight/DeepSeek-R1-0528_w8a8_mix_mtp \
  --host 0.0.0.0 \
  --port 8004 \
  --api-server-count 1 \
  --data-parallel-size 1 \
  --decode-context-parallel-size 8 \
  --prefill-context-parallel-size 2 \
  --cp-kv-cache-interleave-size 128 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enforce-eager \
  --distributed-executor-backend mp \
  --served-model-name deepseek_v3 \
  --quantization ascend \
  --seed 1024 \
  --max-num-seqs 4 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 2,
                    "tp_size": 8
             }
      }
  }' \
  2>&1 | tee /home/t00886357/dynamic_cp/mooncake_test/disaggregate_prefill_lite.log

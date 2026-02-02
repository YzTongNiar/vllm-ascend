unset ftp_proxy
unset https_proxy
unset http_proxy
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTHONPATH=$PYTHONPATH:/home/t00886357/dynamic_cp/vllm-ascend
export PYTHONPATH=$PYTHONPATH:/home/t00886357/dynamic_cp/vllm
export HCCL_IF_IP=141.61.39.177
export GLOO_SOCKET_IFNAME="enp48s3u1u1"  # network card name
export TP_SOCKET_IFNAME="enp48s3u1u1"
export HCCL_SOCKET_IFNAME="enp48s3u1u1"
export VLLM_USE_V1=1
export VLLM_ASCEND_LLMDD_RPC_PORT=5578
export HCCL_BUFFSIZE=384
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL="1"

#  --max-model-len 62600 \
#vllm serve /mnt/nfs/weight/Qwen3-235B-A22B-Instruct-2507/ \
#/mnt/nfs/weights/Qwen3-30B-A3B-W8A8
# vllm serve /mnt/nfs/weights/DeepSeek-R1_w8a8_vllm
# vllm serve /home/t00886357/DeepSeek-V2-Lite \
# vllm serve /mnt/share/weight/Qwen3-235B-A22B-Instruct-2507-w8a8 \
vllm serve /mnt/share/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port 8005 \
  --api-server-count 1 \
  --data-parallel-size 2 \
  --decode-context-parallel-size 1 \
  --prefill-context-parallel-size 1\
  --cp-kv-cache-interleave-size 128 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --served-model-name deepseek_v3 \
  --quantization ascend \
  --seed 1024 \
  --max-model-len 62600 \
  --max-num-batched-tokens 62600 \
  --max-num-seqs 4 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "2",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 2,
                    "tp_size": 8
             }
      }
  }'  \
  2>&1 | tee /home/t00886357/dynamic_cp/mooncake_test/disaggregate_decode_lite.log

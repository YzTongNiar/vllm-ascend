#!/bin/bash
# FIA v2 Sink 双算子独立构建壳（variant-1 迁移后，2026-08-31；a5_port 增 SoC 档 2026-09-02）
# 构建载体：upstream/pr15336（vllm-ascend csrc 布局，PR #15336 @ 34f94ca34）
# 产物：vllm-ascend/vllm_ascend/{vllm_ascend_C*.so, libvllm_ascend_kernels.so, _cann_ops_custom/}
#
# 用法:  bash build.sh                     # A3 (910_93) 全量重建（F2 清缓存）
#        SOC_VERSION=ascend950 bash build.sh  # A5 (950PR) 全量重建
#        bash build.sh --fast              # 跳过清目录，增量重建（同 SoC 才安全）
set -e

TREE="$(cd "$(dirname "$0")/vllm-ascend" && pwd)"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=${SOC_VERSION:-ascend910_9382}
export MAX_JOBS=${MAX_JOBS:-64}
export COMPILE_CUSTOM_KERNELS=1

case "$SOC_VERSION" in
  ascend950*) # 顶层 host_config.cmake 只认具体型号串（ascend950pr_9599 等）；
              # build_aclnn.sh 的 ^ascend950 正则两种写法均匹配，SOC_ARG 恒为 ascend950
    KERNEL_SOC_DIR=ascend950 ;;
  *)         KERNEL_SOC_DIR=ascend910_93 ;;
esac

cd "$TREE"
if [ "$1" != "--fast" ]; then
  echo "[build.sh] F2: clean csrc/build (stale-cache discipline)"
  rm -rf csrc/build
fi
python3 setup.py build_ext --inplace

echo "[build.sh] done (SOC_VERSION=$SOC_VERSION). artifacts:"
ls -la vllm_ascend/*.so "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/${KERNEL_SOC_DIR}/fused_infer_attention_score_v2_sink/" | tail -4

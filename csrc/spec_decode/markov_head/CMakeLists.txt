# ----------------------------------------------------------------------------
# DSpark markov-head fused AscendC custom op (spec-decode draft chain).
#
# Source-ported from the standalone dspark_markov_head wheel project with zero
# semantic changes (device kernels byte-identical; op_host adapted only in the
# generated launch-header include prefix). Runtime entry point for callers is
# the Python wrapper vllm_ascend/ops/markov_head.py.
#
# Build layout (mirrors the mla_preprocess / batch_matmul_transpose direct
# AscendC ops): device kernels get their own ascendc_library linked into
# vllm_ascend_C, while the host wrappers below are compiled into the
# vllm_ascend_C module itself and registered on torch.ops._C_ascend in
# csrc/torch_binding.cpp.
#
# ccec must stay at -O1 for this kernel: -O3 miscompiles it on AscendC 9.1
# (kept from the upstream kernel project; see op_kernel/dspark_mh_ops.cpp
# header notes). The dedicated library target keeps this flag scoped here so
# other kernels in vllm_ascend_kernels are unaffected.
# ----------------------------------------------------------------------------

ascendc_library(dspark_markov_head_kernels SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/dspark_markov_head.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/dspark_mh_ops.cpp
)
ascendc_compile_options(dspark_markov_head_kernels PRIVATE -O1)
ascendc_include_directories(dspark_markov_head_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

# Host-side torch wrappers compiled into vllm_ascend_C by the root
# CMakeLists. ascendc.cmake emits the auto-generated launch headers into the
# per-target dir ${CMAKE_BINARY_DIR}/include/dspark_markov_head_kernels/,
# which is the #include prefix used by op_host/*.cpp.
set(DSPARK_MARKOV_HEAD_OP_HOST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/dspark_markov_head.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/dspark_mh_ops.cpp
    PARENT_SCOPE
)
set(DSPARK_MARKOV_HEAD_OP_HOST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_BINARY_DIR}/include
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include
    PARENT_SCOPE
)

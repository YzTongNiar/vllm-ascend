# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.models.glm5next.recoverssm_config import use_glm_recoverssm
from vllm_ascend.models.qwen3_5.recoverssm_config import use_qwen_gdn_recoverssm


def patch_kda_recoverssm(vllm_config):
    if use_glm_recoverssm(vllm_config):
        import vllm.models.kimi_k3.nvidia.ops.recoverssm as recoverssm

        from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext

        recoverssm.KDARecoverSSMCommitContext = KDARecoverSSMCommitContext

    if use_qwen_gdn_recoverssm(vllm_config):
        # The Ascend metadata builder binds its own context, but patch the
        # upstream module as well for callers that construct the builder before
        # the Ascend backend selector runs.
        import vllm.model_executor.layers.mamba.gdn.recoverssm_gdn as gdn_recoverssm
        import vllm.v1.attention.backends.ple_recoverssm as ple_recoverssm

        from vllm_ascend.ops.triton.gdn.recoverssm import GDNRecoverSSMCommitContext
        from vllm_ascend.ops.triton.kda.recoverssm import (
            _compact_conv_state_kernel,
            _prepare_commit_plan_kernel,
        )

        gdn_recoverssm.GDNRecoverSSMCommitContext = GDNRecoverSSMCommitContext
        ple_recoverssm.compact_conv_state_kernel = _compact_conv_state_kernel
        ple_recoverssm.prepare_commit_plan_kernel = _prepare_commit_plan_kernel

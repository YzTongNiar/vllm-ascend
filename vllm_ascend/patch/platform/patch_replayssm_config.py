# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extend the upstream ReplaySSM selector to Ascend's GLM KDA implementation."""

from functools import wraps

from pydantic.dataclasses import rebuild_dataclass
from vllm.config import VllmConfig

from vllm_ascend.models.glm5next.recoverssm_config import is_glm_recoverssm_model, validate_glm_recoverssm
from vllm_ascend.models.qwen3_5.recoverssm_config import (
    is_qwen_gdn_recoverssm_model,
    validate_qwen_gdn_recoverssm,
)

_original_validate = VllmConfig.validate_mamba_cached_kernel


@wraps(_original_validate)
def validate_mamba_cached_kernel(config):
    if config.cache_config.use_replayssm and is_glm_recoverssm_model(config):
        validate_glm_recoverssm(config)
        config.cache_config.use_kda_recoverssm = True
        return config
    if config.cache_config.use_replayssm and is_qwen_gdn_recoverssm_model(config):
        validate_qwen_gdn_recoverssm(config)
        # #58863 renamed the derived selector to use_recoverssm. Keep both
        # names populated so this plugin works with the paired vLLM revision
        # and with the older compatibility contract used by #19133.
        if hasattr(config.cache_config, "use_recoverssm"):
            config.cache_config.use_recoverssm = True
        if hasattr(config.cache_config, "use_kda_recoverssm"):
            config.cache_config.use_kda_recoverssm = True
        return config
    return _original_validate(config)


VllmConfig.validate_mamba_cached_kernel = validate_mamba_cached_kernel
VllmConfig.__pydantic_decorators__.model_validators["validate_mamba_cached_kernel"].func = validate_mamba_cached_kernel
rebuild_dataclass(VllmConfig, force=True)

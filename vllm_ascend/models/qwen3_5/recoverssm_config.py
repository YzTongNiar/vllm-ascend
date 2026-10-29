# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen GDN RecoverSSM configuration helpers for Ascend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


_QWEN_GDN_MODEL_TYPES = {
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_next",
    "qwen3_next_text",
    "qwen4_exp",
    "qwen4_exp_text",
}


def is_qwen_gdn_recoverssm_model(vllm_config: VllmConfig) -> bool:
    model_config = vllm_config.model_config
    if model_config is None:
        return False
    text_config = getattr(model_config, "hf_text_config", None)
    return getattr(text_config, "model_type", None) in _QWEN_GDN_MODEL_TYPES


def use_qwen_gdn_recoverssm(vllm_config: VllmConfig) -> bool:
    cache = vllm_config.cache_config
    return is_qwen_gdn_recoverssm_model(vllm_config) and bool(getattr(cache, "use_replayssm", False))


def validate_qwen_gdn_recoverssm(vllm_config: VllmConfig) -> None:
    if vllm_config.speculative_config is None:
        raise ValueError("Qwen GDN RecoverSSM requires speculative decoding")

    parallel = vllm_config.parallel_config
    if parallel.pipeline_parallel_size != 1:
        raise ValueError("Qwen GDN RecoverSSM currently requires PP=1")

    mamba = getattr(vllm_config, "mamba_config", None)
    if mamba is not None and getattr(mamba, "enable_stochastic_rounding", False):
        raise ValueError("Qwen GDN RecoverSSM does not support stochastic rounding")

    transfer = vllm_config.kv_transfer_config
    if transfer is not None and transfer.kv_connector not in (
        None,
        "MooncakeConnectorV2",
        "MooncakePullConnector",
    ):
        raise ValueError("Qwen GDN RecoverSSM KV transfer requires MooncakeConnectorV2 or MooncakePullConnector")

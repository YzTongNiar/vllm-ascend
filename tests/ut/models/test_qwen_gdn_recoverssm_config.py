# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.models.qwen3_5.recoverssm_config import (
    is_qwen_gdn_recoverssm_model,
    use_qwen_gdn_recoverssm,
    validate_qwen_gdn_recoverssm,
)


def make_config(model_type="qwen3_5_text"):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)),
        speculative_config=SimpleNamespace(),
        cache_config=SimpleNamespace(use_replayssm=True, mamba_cache_mode="align", enable_prefix_caching=True),
        mamba_config=SimpleNamespace(enable_stochastic_rounding=False),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        kv_transfer_config=None,
    )


@pytest.mark.parametrize(
    "model_type",
    ["qwen3_5", "qwen3_5_text", "qwen3_5_moe_text", "qwen3_next", "qwen3_next_text", "qwen4_exp"],
)
def test_qwen_gdn_model_types_are_supported(model_type):
    config = make_config(model_type)
    assert is_qwen_gdn_recoverssm_model(config)
    assert use_qwen_gdn_recoverssm(config)


def test_disabled_switch_keeps_native_path():
    config = make_config()
    config.cache_config.use_replayssm = False
    assert not use_qwen_gdn_recoverssm(config)


def test_align_and_prefix_cache_are_allowed():
    validate_qwen_gdn_recoverssm(make_config())


def test_speculative_decoding_is_required():
    config = make_config()
    config.speculative_config = None
    with pytest.raises(ValueError, match="requires speculative decoding"):
        validate_qwen_gdn_recoverssm(config)


def test_pipeline_parallelism_is_rejected():
    config = make_config()
    config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="PP=1"):
        validate_qwen_gdn_recoverssm(config)

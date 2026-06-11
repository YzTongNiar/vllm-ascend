# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VWN-Eagle3 model components.

Tests cover PreVwnLayerV1, VwnLlamaDecoderLayer, VwnLlamaModel, and
Eagle3VwnLlamaForCausalLM using CPU-only execution with mocked VllmConfig.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from vllm.config import CacheConfig, CompilationMode, VllmConfig, set_current_vllm_config

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.models.llama_eagle3_vwn import (
    Eagle3VwnLlamaForCausalLM,
    PreVwnLayerV1,
    VwnLlamaDecoderLayer,
    VwnLlamaModel,
)

# Default dimensions matching the real VWN checkpoint config.json
_HIDDEN = 2048
_INTERMEDIATE = 6144
_VOCAB = 151936
_DRAFT_VOCAB = 35000
_NUM_HEADS = 32
_NUM_KV_HEADS = 4
_HEAD_DIM = 128
_RMS_EPS = 1e-6


# ---------------------------------------------------------------------------
# CPU mocks: passthrough modules + autouse fixtures replacing NPU-only ops
# ---------------------------------------------------------------------------


class _Passthrough(nn.Module):
    """Replaces self_attn / mlp for CPU tests — returns hidden_states as-is."""

    def forward(self, hidden_states=None, **kwargs):
        return hidden_states


def _mock_npu_ops_on_layer(layer):
    """Replace self_attn and mlp with passthrough modules for CPU testing."""
    layer.self_attn = _Passthrough()
    layer.mlp = _Passthrough()


class _MockTPGroup:
    """Minimal mock for get_tp_group() when TP=1."""

    rank_in_group = 0
    world_size = 1

    def all_reduce(self, *args, **kwargs):
        pass

    def all_gather(self, x, *args, **kwargs):
        return x.unsqueeze(0)

    def reduce_scatter(self, x, *args, **kwargs):
        return x


@pytest.fixture(autouse=True)
def _mock_tp_group():
    """Patch get_tp_group so CustomLinearOp.tp_rank/tp_size work on CPU."""
    mock = _MockTPGroup()
    with patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=mock), \
         patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock), \
         patch("vllm_ascend.ops.vocab_parallel_embedding.get_tp_group", return_value=mock):
        yield


@pytest.fixture(autouse=True)
def _mock_ascend_config():
    """Patch get_ascend_config so Ascend linear ops work on CPU."""
    cfg = MagicMock()
    cfg.configure_mock(
        enable_flashcomm2_parallel_size=0,
        enable_context_parallel=False,
        enable_flashcomm1=False,
        enable_matmul_allreduce=False,
        weight_nz_mode=1,
        enable_mlapo=True,
        enable_fused_mc2=0,
        msmonitor_use_daemon=False,
        enable_transpose_kv_cache_by_block=True,
        # All finegrained_tp fields default to 0 (disabled)
        **{f"finegrained_tp_config.{name}_tensor_parallel_size": 0
           for name in ("lmhead", "embedding", "oproj", "olora", "mlp")},
    )
    with patch("vllm_ascend.utils.get_ascend_config", return_value=cfg):
        yield


@pytest.fixture(autouse=True)
def _mock_gemm_op():
    """Patch NPU-only vllm custom ops to work on CPU."""
    with patch.object(torch.ops.vllm, "unquantized_gemm",
                      lambda input, weight, bias=None: nn.functional.linear(input, weight, bias)), \
         patch.object(torch.ops.vllm, "maybe_calc_kv_scales", lambda *a, **kw: None), \
         patch.object(torch.ops.vllm, "maybe_pad_and_reduce", lambda x, *a, **kw: x):
        yield


# ---------------------------------------------------------------------------
# Helpers: VllmConfig / hf config / decoder layer builders
# ---------------------------------------------------------------------------


def _make_hf_config(num_hidden_layers=1, vwn_m=4, vwn_r=1.5):
    """Create a real LlamaConfig with VWN attributes.

    Using a real config object instead of MagicMock avoids whack-a-mole
    with missing attributes that the deep init chain of LlamaDecoderLayer
    expects (rope_parameters, max_position_embeddings, etc.).
    """
    from transformers import LlamaConfig

    cfg = LlamaConfig(
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        num_hidden_layers=num_hidden_layers,
        vocab_size=_VOCAB,
        rms_norm_eps=_RMS_EPS,
        max_position_embeddings=40960,
    )
    # VWN-specific attributes (not in LlamaConfig schema)
    cfg.vwn_m, cfg.vwn_r, cfg.draft_vocab_size = vwn_m, vwn_r, _DRAFT_VOCAB
    return cfg


def _create_vllm_config_for_vwn(vwn_m=4, vwn_r=1.5, num_hidden_layers=1):
    """Create a fully mocked VllmConfig for VWN model instantiation on CPU."""
    hf_config = _make_hf_config(num_hidden_layers, vwn_m, vwn_r)
    vllm_config = MagicMock(spec=VllmConfig)

    spec = vllm_config.speculative_config = MagicMock()
    spec.method = "eagle3"
    spec.num_speculative_tokens = 3
    spec.draft_tensor_parallel_size = 1
    spec.disable_padded_drafter_batch = False
    spec.parallel_drafting = False
    spec.speculative_token_tree = str([(i + 1) * (0,) for i in range(3)])

    draft = spec.draft_model_config = MagicMock()
    draft.hf_config = hf_config
    draft.get_hidden_size.return_value = _HIDDEN
    draft.get_inputs_embeds_size.return_value = _HIDDEN
    draft.uses_mrope = False
    draft.uses_xdrope_dim = 0
    draft.quantization = None

    cache = vllm_config.cache_config = MagicMock(spec=CacheConfig)
    cache.block_size = 16
    cache.kv_cache_dtype_skip_layers = None
    cache.cache_dtype = "auto"

    sched = vllm_config.scheduler_config = MagicMock()
    sched.max_num_batched_tokens = 1024
    sched.max_num_seqs = 32
    sched.async_scheduling = True

    mc = vllm_config.model_config = MagicMock()
    mc.dtype = torch.float32  # float32 for CPU testing
    mc.max_model_len = 2048
    mc.uses_mrope = False
    mc.uses_xdrope_dim = 0
    mc.enforce_eager = True
    mc.hf_text_config = MagicMock(spec=[])
    mc.hf_text_config.to_dict = MagicMock(return_value={})
    mc.hf_config = hf_config
    mc.get_num_layers.return_value = 48  # num target layers

    comp = vllm_config.compilation_config = MagicMock()
    comp.mode = CompilationMode.NONE
    comp.pass_config.enable_sp = False
    comp.custom_ops = ["none"]  # Required by CustomOp.default_on()

    par = vllm_config.parallel_config = MagicMock()
    par.tensor_parallel_size = 1
    par.data_parallel_rank = 0
    par.data_parallel_size = 1
    par.prefill_context_parallel_size = 1
    par.decode_context_parallel_size = 1
    par.enable_expert_parallel = False

    vllm_config.additional_config = None

    init_ascend_config(vllm_config)
    return vllm_config


def _hf(vllm_config):
    """Shortcut to the draft model's HF config."""
    return vllm_config.speculative_config.draft_model_config.hf_config


def _make_layer(vllm_config, layer_idx=0):
    """Instantiate a VwnLlamaDecoderLayer (call inside set_current_vllm_config)."""
    return VwnLlamaDecoderLayer(
        vllm_config=vllm_config,
        prefix="model.layers.48",
        config=_hf(vllm_config),
        layer_idx=layer_idx,
    )


# ---------------------------------------------------------------------------
# Test: PreVwnLayerV1
# ---------------------------------------------------------------------------


class TestPreVwnLayerV1:
    """Tests for the pre-VWN projection layer — init + forward merged."""

    @pytest.mark.parametrize("vwn_m,vwn_r", [(1, 1.5), (4, 1.5), (4, 1.0), (1, 1.0)])
    def test_init_and_forward(self, vwn_m, vwn_r):
        """Verify layer init (dims, submodules) and forward output shape."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=vwn_m, vwn_r=vwn_r)
        hs, wd, batch = _HIDDEN, int(_HIDDEN * vwn_r), 4

        with set_current_vllm_config(vllm_config):
            layer = PreVwnLayerV1(
                vllm_config=vllm_config, prefix="test_prevwn", config=_hf(vllm_config),
            )
            assert (layer.hidden_size, layer.wider_dim, layer.m) == (hs, wd, vwn_m)
            for attr in ("input_layernorm", "hidden_norm", "fc", "upward"):
                assert hasattr(layer, attr)

            out = layer(torch.randn(batch, hs), torch.randn(batch, hs))

        assert out.shape == (batch, wd)


# ---------------------------------------------------------------------------
# Test: VwnLlamaDecoderLayer
# ---------------------------------------------------------------------------


class TestVwnLlamaDecoderLayer:
    """Tests for the VWN-augmented Llama decoder layer."""

    @pytest.mark.parametrize("vwn_m,vwn_r", [(4, 1.5), (4, 1.0), (1, 1.5)])
    def test_init_vwn_projections(self, vwn_m, vwn_r):
        """VWN-specific submodules and dimension bookkeeping."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=vwn_m, vwn_r=vwn_r)

        with set_current_vllm_config(vllm_config):
            layer = _make_layer(vllm_config)

        assert isinstance(layer.pre_vwn_layer, PreVwnLayerV1)
        for attr in ("downward_and_forgot", "upward_after_attn",
                     "upward_after_mlp", "downward"):
            assert hasattr(layer, attr)
        assert (layer.m, layer.wider_dim) == (vwn_m, int(_HIDDEN * vwn_r))

    @pytest.mark.parametrize("vwn_m,vwn_r,batch", [
        (4, 1.5, 4), (4, 1.0, 4), (1, 1.5, 4), (4, 1.5, 1),
    ])
    def test_forward_layer0(self, vwn_m, vwn_r, batch):
        """VWN forward with various m/r configs and batch sizes."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=vwn_m, vwn_r=vwn_r)

        with set_current_vllm_config(vllm_config):
            layer = _make_layer(vllm_config)
            _mock_npu_ops_on_layer(layer)

            out_hidden, _ = layer(
                torch.arange(batch, dtype=torch.long),  # positions
                torch.randn(batch, _HIDDEN),            # embeds
                torch.randn(batch, _HIDDEN),            # hidden
                None,
            )

        assert out_hidden.shape == (batch, _HIDDEN)

    def test_forward_layer_nonzero_passthrough(self):
        """Non-zero layer_idx returns hidden_states unchanged."""
        vllm_config = _create_vllm_config_for_vwn()
        batch = 2

        with set_current_vllm_config(vllm_config):
            layer = _make_layer(vllm_config, layer_idx=1)
            out_hidden, _ = layer(
                torch.arange(batch, dtype=torch.long),
                torch.randn(batch, _HIDDEN),
                torch.randn(batch, _HIDDEN),
                None,
            )

        assert out_hidden.shape == (batch, _HIDDEN)

    def test_qkv_proj_input_size_layer0(self):
        """VWN layer 0 restores qkv_proj input to hidden_size (not 2*hs).

        This is the critical VWN-vs-Eagle3 difference: the parent class
        overrides qkv_proj to accept 2*hidden_size (concatenating embeds +
        hidden), but VWN feeds hidden_size into attention via the
        downward_and_forgot projection instead.
        """
        vllm_config = _create_vllm_config_for_vwn()

        with set_current_vllm_config(vllm_config):
            layer = _make_layer(vllm_config)

        assert layer.self_attn.qkv_proj.input_size == _HIDDEN


# ---------------------------------------------------------------------------
# Test: VwnLlamaModel
# ---------------------------------------------------------------------------


class TestVwnLlamaModel:
    """Tests for the full VWN model body."""

    @pytest.mark.parametrize("num_hidden_layers", [1, 2])
    def test_init_and_forward(self, num_hidden_layers):
        """Verify layer count/type and forward output shapes."""
        vllm_config = _create_vllm_config_for_vwn(num_hidden_layers=num_hidden_layers)
        num_tokens = 4

        with set_current_vllm_config(vllm_config):
            model = VwnLlamaModel(vllm_config=vllm_config, prefix="model", start_layer_id=48)

            assert len(model.layers) == num_hidden_layers
            for i, layer in enumerate(model.layers):
                assert isinstance(layer, VwnLlamaDecoderLayer)
                assert layer.layer_idx == i
                _mock_npu_ops_on_layer(layer)

            postnorm, prenorm = model(
                torch.randint(0, _VOCAB, (num_tokens,)),
                torch.arange(num_tokens, dtype=torch.long),
                torch.randn(num_tokens, _HIDDEN),
            )

        assert postnorm.shape == (num_tokens, _HIDDEN)
        assert prenorm.shape == (num_tokens, _HIDDEN)

    def test_forward_with_input_embeds(self):
        """Forward with explicit input_embeds bypasses embedding lookup."""
        vllm_config = _create_vllm_config_for_vwn()
        num_tokens = 3

        with set_current_vllm_config(vllm_config):
            model = VwnLlamaModel(vllm_config=vllm_config, prefix="model", start_layer_id=48)
            for layer in model.layers:
                _mock_npu_ops_on_layer(layer)

            postnorm, prenorm = model(
                torch.randint(0, _VOCAB, (num_tokens,)),
                torch.arange(num_tokens, dtype=torch.long),
                torch.randn(num_tokens, _HIDDEN),
                input_embeds=torch.randn(num_tokens, _HIDDEN),
            )

        assert postnorm.shape == (num_tokens, _HIDDEN)
        assert prenorm.shape == (num_tokens, _HIDDEN)


# ---------------------------------------------------------------------------
# Test: Eagle3VwnLlamaForCausalLM
# ---------------------------------------------------------------------------


class TestEagle3VwnLlamaForCausalLM:
    """Tests for the top-level VWN-Eagle3 CausalLM model."""

    @pytest.mark.parametrize("vwn_m", [4, 1])
    def test_init_and_forward(self, vwn_m):
        """Init creates VwnLlamaModel; forward returns (postnorm, prenorm)."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=vwn_m)
        num_tokens = 3

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            assert isinstance(model.model, VwnLlamaModel)
            for layer in model.model.layers:
                _mock_npu_ops_on_layer(layer)

            postnorm, prenorm = model(
                torch.randint(0, _VOCAB, (num_tokens,)),
                torch.arange(num_tokens, dtype=torch.long),
                torch.randn(num_tokens, _HIDDEN),
            )

        assert postnorm.shape == (num_tokens, _HIDDEN)
        assert prenorm.shape == (num_tokens, _HIDDEN)

    def test_embed_input_ids(self):
        vllm_config = _create_vllm_config_for_vwn()
        num_tokens = 3

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            embeds = model.embed_input_ids(torch.randint(0, _VOCAB, (num_tokens,)))

        assert embeds.shape == (num_tokens, _HIDDEN)

    def test_compute_logits_output_shape_and_mapping(self):
        """compute_logits maps draft vocab logits to target vocab positions."""
        vllm_config = _create_vllm_config_for_vwn()
        batch = 2

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            model.draft_id_to_target_id.data.copy_(
                torch.arange(_DRAFT_VOCAB, dtype=torch.long))

            # Patch logits_processor.forward to bypass ParallelLMHead (needs
            # real TP group coordinator).
            with patch.object(type(model.logits_processor), "forward",
                              return_value=torch.randn(batch, _DRAFT_VOCAB)):
                logits = model.compute_logits(torch.randn(batch, _HIDDEN))

        assert logits.shape == (batch, _VOCAB)
        # Mapped positions should have finite values
        mapped = torch.arange(_DRAFT_VOCAB) + model.draft_id_to_target_id
        assert logits[0, mapped[0]].item() != float("-inf")
        # Unmapped positions should be -inf
        assert logits[0, _VOCAB - 1].item() == float("-inf")

    @pytest.mark.parametrize("use_aux", [True, False])
    def test_combine_hidden_states(self, use_aux):
        """combine_hidden_states: FC projection when aux=True, identity when False."""
        vllm_config = _create_vllm_config_for_vwn()
        batch = 2

        if not use_aux:
            _hf(vllm_config).eagle_config = {"use_aux_hidden_state": False}

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            assert model.model.use_aux_hidden_state == use_aux

            if use_aux:
                combined = model.combine_hidden_states(torch.randn(batch, _HIDDEN * 3))
                assert combined.shape == (batch, _HIDDEN)
            else:
                hidden = torch.randn(batch, _HIDDEN)
                assert torch.equal(model.combine_hidden_states(hidden), hidden)

    def test_vwn_parameters_and_count(self):
        """Verify VWN parameters exist and total count matches expectation."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=4, vwn_r=1.5)

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")

        params = dict(model.named_parameters())

        # VWN-specific prefixes must all have registered parameters
        for prefix in ("pre_vwn_layer", "downward_and_forgot", "upward_after_attn",
                       "upward_after_mlp", "downward"):
            assert any(prefix in n for n in params), (
                f"No parameters found for VWN prefix '{prefix}'"
            )

        # draft_id_to_target_id exists and does not require grad
        assert "draft_id_to_target_id" in params
        assert not params["draft_id_to_target_id"].requires_grad

        # Total parameter count (update if architecture changes)
        total = sum(p.numel() for p in model.parameters())
        assert total == 454_607_032, (
            f"Parameter count mismatch: got {total}, expected 454_607_032. "
            "Update if model architecture changed."
        )


# ---------------------------------------------------------------------------
# Test: Weight loading integration
# ---------------------------------------------------------------------------


def _make_synthetic_weights(vwn_r):
    """Synthetic checkpoint weights (m=4) with real shapes for the given r.

    r=1.0 checkpoints carry no `downward.weight` / `pre_vwn_layer.upward.weight`
    because wider_dim == hidden_size makes those projections identity-sized.
    """
    hs, m, f32 = _HIDDEN, 4, torch.float32
    wd = int(hs * vwn_r)
    qd, kvd = _NUM_HEADS * _HEAD_DIM, _NUM_KV_HEADS * _HEAD_DIM
    weights = {
        "d2t": torch.zeros(_DRAFT_VOCAB, dtype=torch.long),
        "embed_tokens.weight": torch.randn(_VOCAB, hs, dtype=f32),
        "fc.weight": torch.randn(hs, hs * 3, dtype=f32),
        "layers.0.downward_and_forgot.weight":
            torch.randn((hs + wd) // m, wd // m, dtype=f32),
        "layers.0.downward_and_forgot_after_attn.weight":
            torch.randn((hs + wd) // m, wd // m, dtype=f32),
        "layers.0.mlp.gate_proj.weight": torch.randn(_INTERMEDIATE, hs, dtype=f32),
        "layers.0.mlp.up_proj.weight": torch.randn(_INTERMEDIATE, hs, dtype=f32),
        "layers.0.mlp.down_proj.weight": torch.randn(hs, _INTERMEDIATE, dtype=f32),
        "layers.0.post_attention_layernorm.weight": torch.randn(hs, dtype=f32),
        "layers.0.pre_attention_layernorm.weight": torch.randn(hs, dtype=f32),
        "layers.0.pre_vwn_layer.fc.weight": torch.randn(hs, 2 * hs, dtype=f32),
        "layers.0.pre_vwn_layer.hidden_norm.weight": torch.randn(hs, dtype=f32),
        "layers.0.pre_vwn_layer.input_layernorm.weight": torch.randn(hs, dtype=f32),
        "layers.0.self_attn.q_proj.weight": torch.randn(qd, hs, dtype=f32),
        "layers.0.self_attn.k_proj.weight": torch.randn(kvd, hs, dtype=f32),
        "layers.0.self_attn.v_proj.weight": torch.randn(kvd, hs, dtype=f32),
        "layers.0.self_attn.o_proj.weight": torch.randn(hs, qd, dtype=f32),
        "layers.0.upward_after_attn.weight": torch.randn(wd // m, hs // m, dtype=f32),
        "layers.0.upward_after_mlp.weight": torch.randn(wd // m, hs // m, dtype=f32),
        "lm_head.weight": torch.randn(_DRAFT_VOCAB, hs, dtype=f32),
        "norm.weight": torch.randn(hs, dtype=f32),
    }
    if vwn_r != 1.0:
        weights["layers.0.downward.weight"] = torch.randn(hs // m, wd // m, dtype=f32)
        weights["layers.0.pre_vwn_layer.upward.weight"] = \
            torch.randn(wd // m, hs // m, dtype=f32)
    return weights


class TestWeightLoadingIntegration:
    """Weight loading tests using synthetic weight dicts with real shapes."""

    def test_load_weights_r15(self):
        """Load all weights for m=4, r=1.5: d2t remap, t2d skip, fc loaded."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=4, vwn_r=1.5)
        weights = _make_synthetic_weights(1.5)
        # Non-trivial d2t; t2d should be skipped; distinctive fc.weight
        weights["d2t"] = torch.arange(_DRAFT_VOCAB, dtype=torch.long)
        weights["t2d"] = torch.zeros(_VOCAB, dtype=torch.bool)
        weights["fc.weight"] = torch.ones(_HIDDEN, _HIDDEN * 3, dtype=torch.float32)

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            model.load_weights(weights.items())

        # d2t remapped to draft_id_to_target_id
        dit = dict(model.named_parameters())["draft_id_to_target_id"]
        assert torch.equal(dit.data, torch.arange(_DRAFT_VOCAB, dtype=torch.long))

        # fc weight loaded
        assert model.model.fc.weight.data.abs().sum() > 0
        assert model.model.fc.weight.shape == (_HIDDEN, _HIDDEN * 3)

        # Key parameters loaded (not all-zero)
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                assert param.data.abs().sum() > 0, f"{name} not loaded"

    def test_load_weights_r10(self):
        """Load weights for m=4, r=1.0: no downward/pre_vwn upward in checkpoint.

        When vwn_r=1.0, wider_dim == hidden_size, so downward and pre_vwn
        projections are identity-sized (hs//m, hs//m).
        """
        vllm_config = _create_vllm_config_for_vwn(vwn_m=4, vwn_r=1.0)
        hs, m = _HIDDEN, 4

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            model.load_weights(_make_synthetic_weights(1.0).items())

        layer = model.model.layers[0]
        # wd == hs when r=1.0, so dimensions are identity-sized
        assert layer.downward.weight.shape == (hs // m, hs // m)
        assert layer.pre_vwn_layer.upward.weight.shape == (hs // m, hs // m)
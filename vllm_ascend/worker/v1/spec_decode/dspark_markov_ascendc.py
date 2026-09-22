# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AscendC fused DSpark Markov-head proposal chain (host-side orchestration).

Replaces the per-step PyTorch Markov loop in
``vllm_ascend/spec_decode/llm_base_proposer.py``

    for idx in range(K):
        markov_emb = model.markov_embed(draft_token_ids[:, idx])   # w1 lookup (+AR)
        logits_bias = model.markov_bias(markov_emb)                # GEMV (+AG)
        logits[:, idx].add_(logits_bias)
        draft_token_ids[:, idx + 1] = logits[:, idx].argmax(-1)

with the ``dspark_markov_head`` AscendC operator family, accessed through the
unified wrapper ``vllm_ascend/ops/markov_head.py`` (prefers the in-repo csrc
build registered on ``torch.ops._C_ascend.*``, falls back to the installed
whl's ``torch.ops.npu.*``):

* TP=1  -> **L3 composition without communication**: per step one
  ``dspark_mh3_step`` (w1 lookup -> aclnnMatMulV2 GEMV over the full-vocab w2
  -> biased argmax -> packed candidate keys), one ``dspark_mh3_merge`` per
  forward. (The single-card fused op ``dspark_markov_head`` is deliberately
  NOT used: its in-kernel GEMV is only bit-exact against vLLM's
  ``F.linear``/aclnnMatMulV2 at its calibrated folded width Vd=19360 — at the
  full Vd==V=154880 its stripe-dependent fp32 accumulation order diverges,
  see the mismatch numbers in this module's validation notes.)

* TP>1  -> **L3 chain**: the same per-step composition over each rank's w2
  shard plus one int64 all-gather of the [W, 48*16] candidate-key buffer per
  step. The row-sharded w1 is replicated once at first use via a blocking
  all-gather (the embedding then becomes rank-local, which is numerically
  identical to vLLM's masked lookup + AllReduce(SUM): non-owner ranks
  contribute exact +0.0). The sharded w2 is used in place (ParallelLMHead
  shard == the operator's [vs, 256] shard with the same rank-ordered global
  vocab layout that ``LogitsProcessor`` gathers).

Both chains produce out_ids int64 bit-identical to the PyTorch deployment
chain (first-max argmax, bf16 materialization points preserved) when the
deployment dtype is bf16; the operator never writes its input tensors.

bf16 deployments (e.g. a quantized main model locking ``--dtype bfloat16``
while the speculative dtype follows the target model) are served natively:
the operator's L3 chain is bf16-only (fp16 retired from L3; see the operator
task's ``docs/bf16_native_migration_report.md``) and bit-identical to the
bf16 PyTorch deployment chain (materialization point 1 = aclnnMatMulV2 bf16,
point 2 = CAST_RINT fp32->bf16 RNE, bf16 mono packing), so the acceptance
profile matches the torch loop exactly. fp16 weights/logits keep the torch
loop (the bf16-only operator would reject them).

The path is opt-in via ``VLLM_ASCEND_DSPARK_MARKOV_IMPL=ascendc`` (default
``torch`` keeps the original loop with zero impact). Any unsupported shape /
head layout / batch size keeps the original loop, and a runtime failure falls
back to it for the current step (the draft buffer is only written by the
final copy-back, so the fallback recomputes cleanly from the seed column).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger

from vllm_ascend.ops.markov_head import MarkovHeadOps, markov_head_ops

# Kernel-side hard limits (see the operator's op_host checks):
#   dspark_markov_head: B <= 8 (B_PAD), w1/w2 rank must be 256;
#   dspark_mh3_merge:   B in [1, 8]; mh3 tok lines are [16] int64.
_MAX_BATCH = 8
_MARKOV_RANK = 256
_WS_LINE = 16  # int64 lanes per tok line / per-core ws slot

_ENABLED: Optional[bool] = None
_OPS: Optional[MarkovHeadOps] = None


def dspark_markov_ascendc_enabled() -> bool:
    """Runtime switch, read once (lazily so tests can set the env var)."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("VLLM_ASCEND_DSPARK_MARKOV_IMPL", "torch").strip().lower() == "ascendc"
    return _ENABLED


def _load_kernel_ops() -> bool:
    global _OPS
    try:
        _OPS = markov_head_ops()
    except RuntimeError as e:  # pragma: no cover - environment dependent
        logger.warning("[dspark-markov-ascendc] kernel ops unavailable (%s); keeping torch loop", e)
        return False
    return True


def _find_markov_head(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Locate the module owning both markov weights, across model flavors.

    Covers deepseek_v4 (``model.model.markov_head``: VocabParallelEmbedding +
    ParallelLMHead), upstream qwen3/kimi k3 (``model.model.markov_head``:
    nn.Embedding + disable_tp ParallelLMHead) and future variants that expose
    ``markov_w1`` / ``markov_w2`` on the same submodule.
    """
    if model is None:
        return None
    if hasattr(model, "markov_w1") and hasattr(model, "markov_w2"):
        return model
    for _, child in model.named_children():
        found = _find_markov_head(child)
        if found is not None:
            return found
    return None


def gather_w1_full(
    w1_shard: torch.Tensor,
    V: int,
    world: int,
    pg=None,
) -> torch.Tensor:
    """One-time rank-row all-gather of the sharded w1 into a full [V, R]."""
    vs = V // world
    w1_full = torch.empty((V, w1_shard.shape[1]), dtype=w1_shard.dtype, device=w1_shard.device)
    chunks = [w1_full[r * vs : (r + 1) * vs] for r in range(world)]
    dist.all_gather(chunks, w1_shard, group=pg)
    return w1_full


def run_l3_chain(
    logits: torch.Tensor,
    ids: torch.Tensor,
    w1_full: torch.Tensor,
    w2_shard: torch.Tensor,
    rank: int,
    world: int,
    bufs: "L3Bufs | None",
    pg=None,
) -> tuple[torch.Tensor, "L3Bufs"]:
    """L3 chain, verbatim op order from the operator-side acceptance harness
    (dspark_markov_head_multicard/test/tp2_golden_check.py::l3_forward).

    Returns (out_ids [B, K+1] int64, reusable buffers). Inputs are read-only.
    """
    B, K, V = logits.shape
    if bufs is None or bufs.tok_hist.shape[0] != K + 1 or bufs.ws_ag.numel() != world * 48 * _WS_LINE:
        bufs = L3Bufs(K, world, logits.device)
    ops = _OPS if _OPS is not None else markov_head_ops()
    bufs.tok_hist[0, :B] = ids[:, 0]
    for k in range(K):
        ops.step(
            logits,
            w1_full,
            w2_shard,
            bufs.ws_ag,
            bufs.tok_hist[0],
            bufs.tok_hist[k],
            rank,
            world,
            k,
            1 if k else 0,
        )
        if world > 1:
            dist.all_gather(bufs.ws_views, bufs.ws_views[rank], group=pg)
    ops.merge(bufs.ws_ag, bufs.tok_hist[K], B, world)
    out = bufs.out_ids[:B]
    out[:, 0] = ids[:, 0]
    out[:, 1:] = bufs.tok_hist[1 : K + 1, :B].t()
    return out, bufs


class L3Bufs:
    """Persistent L3 chain buffers (stable addresses; eager + capture safe)."""

    def __init__(self, K: int, world: int, device: torch.device) -> None:
        self.tok_hist = torch.zeros(K + 1, _WS_LINE, dtype=torch.int64, device=device)
        self.ws_ag = torch.zeros(world * 48 * _WS_LINE, dtype=torch.int64, device=device)
        line = 48 * _WS_LINE  # per-rank packed-candidate slot in the gather
        self.ws_views = [self.ws_ag[r * line : (r + 1) * line] for r in range(world)]
        self.out_ids = torch.empty(_MAX_BATCH, K + 1, dtype=torch.int64, device=device)


class DSparkMarkovAscendC:
    """Lazily-initialized AscendC runner bound to one DSpark draft model."""

    def __init__(self, model: torch.nn.Module, tp_process_group=None) -> None:
        self.supported = False
        self.reason = ""
        if not _load_kernel_ops():
            self.reason = "kernel whl/ops unavailable"
            return
        head = _find_markov_head(model)
        if head is None:
            self.reason = "markov head (markov_w1/markov_w2) not found on draft model"
            return
        w1_mod, w2_mod = head.markov_w1, head.markov_w2
        # LoRA-style added-vocabulary embedding shards have a different layout.
        if getattr(w1_mod, "num_added_embeddings", 0):
            self.reason = "markov_w1 has added (lora) vocab rows"
            return
        w1, w2 = w1_mod.weight, w2_mod.weight
        if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
            # The operator's L3 chain is bf16-only (fp16 retired; the previous
            # bf16->fp16 cast path lost draft quality at argmax near-ties).
            # fp16 deployments keep the torch loop instead of a lossy cast.
            self.reason = f"markov weights must be bf16 (operator is bf16-only), got {w1.dtype}"
            return
        if not w1.is_contiguous() or not w2.is_contiguous():
            self.reason = "markov weights must be contiguous"
            return
        if w1.shape[1] != _MARKOV_RANK:
            self.reason = f"markov rank must be {_MARKOV_RANK}, got {w1.shape[1]}"
            return
        self._w1 = w1
        self._w2 = w2
        self._w1_full: Optional[torch.Tensor] = None  # lazy (TP>1 sharded only)
        self._l3_bufs: Optional[L3Bufs] = None
        self.world = get_tensor_model_parallel_world_size()
        self.rank = get_tensor_model_parallel_rank()
        self._pg = tp_process_group  # None -> vllm TP group (resolved lazily)
        # NOTE: the L1 fused op (dspark_markov_head) is bit-exact only at its
        # calibrated folded w2 width (Vd=19360); at full Vd==V it diverges from
        # vLLM's F.linear numerics (49/56 out_ids entries differ at B=7,K=7),
        # so every path uses the L3 composition (TP=1 simply skips the AG).
        self.mode = "l3"
        self.supported = True
        self._logged = False

    # -- helpers -----------------------------------------------------------

    def _tp_group(self):
        if self._pg is None:
            from vllm.distributed.parallel_state import get_tp_group

            self._pg = get_tp_group().device_group
        return self._pg

    def _w1_for_V(self, V: int) -> Optional[torch.Tensor]:
        """Full-vocab w1 [V, 256] for this rank (None -> unsupported)."""
        w1 = self._w1
        if w1.shape[0] == V:  # replicated head (nn.Embedding / disable_tp)
            return w1
        world = self.world
        if world > 1 and V % world == 0 and w1.shape[0] == V // world:
            if self._w1_full is None or self._w1_full.shape[0] != V:
                # One-time blocking gather (~V*R*2 bytes, 79MB at V=154880).
                # NOTE: a later in-place weight reload would not be reflected;
                # restart (or keep the torch impl) when hot-reloading weights.
                self._w1_full = gather_w1_full(w1, V, world, pg=self._tp_group())
            return self._w1_full
        return None

    def _w2_shard_for_V(self, V: int) -> Optional[torch.Tensor]:
        """This rank's w2 shard [vs, 256] covering global rows [r0*vs, (r0+1)*vs)."""
        w2 = self._w2
        if self.world == 1:
            return w2 if w2.shape[0] == V else None
        if V % self.world != 0:
            return None
        vs = V // self.world
        if w2.shape[0] == vs:  # VocabParallel sharded (no vocab padding)
            return w2
        if w2.shape[0] == V:  # replicated full-vocab head: slice this rank's rows
            return w2[self.rank * vs : (self.rank + 1) * vs]
        return None

    def _log_once(self, msg: str) -> None:
        if not self._logged:
            self._logged = True
            logger.info("[dspark-markov-ascendc] %s", msg)

    # -- main entry ---------------------------------------------------------

    def run(self, logits: torch.Tensor, draft_token_ids: torch.Tensor) -> bool:
        """Fill draft_token_ids[:, 1:] via the AscendC chain.

        Returns False (untouched inputs) when anything is unsupported so the
        caller can keep the original PyTorch loop.
        """
        if not self.supported:
            return False
        if logits.dim() != 3 or logits.dtype != torch.bfloat16:
            return False
        if not logits.is_contiguous():
            return False
        B, K, V = logits.shape
        if not 1 <= B <= _MAX_BATCH or K < 1:
            return False
        if draft_token_ids.dtype != torch.int64 or tuple(draft_token_ids.shape) != (B, K + 1):
            return False
        if not draft_token_ids.is_contiguous():
            return False
        w1 = self._w1_for_V(V)
        if w1 is None:
            self._log_once(f"unsupported vocab layout (V={V}, w1 rows={self._w1.shape[0]})")
            return False
        w2 = self._w2_shard_for_V(V)
        if w2 is None:
            self._log_once(f"unsupported w2 layout (V={V}, w2 rows={self._w2.shape[0]})")
            return False
        out, self._l3_bufs = run_l3_chain(
            logits,
            draft_token_ids,
            w1,
            w2,
            self.rank,
            self.world,
            self._l3_bufs,
            pg=self._tp_group() if self.world > 1 else None,
        )
        draft_token_ids.copy_(out)
        self._log_once(f"L3 chain active (world={self.world}, V={V})")
        return True


def maybe_run_dspark_markov_ascendc(proposer, logits: torch.Tensor, draft_token_ids: torch.Tensor) -> bool:
    """Fast-path entry used by llm_base_proposer's dspark branch.

    Caches the runner on the proposer; never raises (any failure permanently
    degrades to the original PyTorch loop for this engine).
    """
    if not dspark_markov_ascendc_enabled():
        return False
    runner = getattr(proposer, "_dspark_markov_ascendc_runner", None)
    if runner is None:
        try:
            runner = DSparkMarkovAscendC(proposer.model)
        except Exception as e:
            logger.warning("[dspark-markov-ascendc] init failed (%s); keeping torch loop", e)
            runner = None
        if runner is None or not runner.supported:
            # WARNING, not INFO: the env var explicitly requested ascendc, so a
            # silent disable would make a passing benchmark a false positive
            # (the op never runs and the torch loop answers every request).
            logger.warning(
                "[dspark-markov-ascendc] disabled: %s "
                "(VLLM_ASCEND_DSPARK_MARKOV_IMPL=ascendc was requested; "
                "decode is served by the torch loop)",
                runner.reason if runner else "init exception",
            )
            proposer._dspark_markov_ascendc_runner = False
            return False
        logger.info(
            "[dspark-markov-ascendc] enabled: mode=%s world=%d w1=%s w2=%s dtype=bf16(native)",
            runner.mode,
            runner.world,
            tuple(runner._w1.shape),
            tuple(runner._w2.shape),
        )
        proposer._dspark_markov_ascendc_runner = runner
    if runner is False:
        return False
    try:
        return runner.run(logits, draft_token_ids)
    except Exception:
        logger.exception("[dspark-markov-ascendc] runtime failure; falling back to torch loop")
        proposer._dspark_markov_ascendc_runner = False
        return False

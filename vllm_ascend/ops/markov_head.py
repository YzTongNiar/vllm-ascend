# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified access layer for the DSpark markov-head AscendC operator family.

Engineering boundary between the markov-head callers and the backing kernel
library: upper layers (``vllm_ascend/worker/v1/spec_decode/dspark_markov_ascendc.py``)
call the semantic functions here (:func:`markov_step` / :func:`markov_merge`)
instead of touching ``torch.ops`` namespaces directly, so the backing library
can evolve (in-repo csrc build -> torch_npu API -> CANN aclnn) without
touching callers.

Backend resolution order (first available wins):

1. **in-repo build** — the operator family compiled into ``vllm_ascend_C`` at
   install time (``csrc/spec_decode/markov_head``) and registered on
   ``torch.ops._C_ascend.dspark_mh3_step`` / ``dspark_mh3_merge``.
2. **standalone wheel** — ``import dspark_markov_head`` registers the same op
   family on ``torch.ops.npu.*`` (the pre-integration delivery form).

The two registrations deliberately use different torch library namespaces so
both libraries can be loaded in the same process without dispatcher
conflicts (a second ``TORCH_LIBRARY_IMPL(npu, PrivateUse1)`` registration of
the same op would fail at load time).

:func:`markov_head_ops` raises ``RuntimeError`` with the per-backend reasons
when neither backend can serve the ops; the semantic functions propagate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from vllm.logger import logger

_REQUIRED_OPS = ("dspark_mh3_step", "dspark_mh3_merge")


@dataclass(frozen=True)
class MarkovHeadOps:
    """Resolved markov-head op backend (handles are bound callables)."""

    backend: str  # "csrc" (in-repo vllm_ascend_C) | "wheel"
    step: Callable[..., None]  # dspark_mh3_step
    merge: Callable[..., None]  # dspark_mh3_merge


_OPS: Optional[MarkovHeadOps] = None


def _missing_ops(namespace: object) -> list[str]:
    return [name for name in _REQUIRED_OPS if not hasattr(namespace, name)]


def _try_inrepo() -> tuple[Optional[MarkovHeadOps], str]:
    ns = torch.ops._C_ascend
    missing = _missing_ops(ns)
    if missing:
        return None, (
            f"in-repo csrc build unavailable: torch.ops._C_ascend missing {missing} "
            "(vllm_ascend_C not built or not loaded)"
        )
    return MarkovHeadOps("csrc", ns.dspark_mh3_step, ns.dspark_mh3_merge), ""


def _try_wheel() -> tuple[Optional[MarkovHeadOps], str]:
    try:
        import dspark_markov_head  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        return None, f"dspark_markov_head whl unavailable ({e})"
    ns = torch.ops.npu
    missing = _missing_ops(ns)
    if missing:
        return None, f"dspark_markov_head whl imported but torch.ops.npu missing {missing}"
    return MarkovHeadOps("wheel", ns.dspark_mh3_step, ns.dspark_mh3_merge), ""


def markov_head_ops() -> MarkovHeadOps:
    """Resolve the markov-head op backend (cached on success).

    Raises:
        RuntimeError: neither the in-repo csrc build nor the standalone wheel
            can provide ``dspark_mh3_step``/``dspark_mh3_merge``; the message
            carries the reason for each backend.
    """
    global _OPS
    if _OPS is not None:
        return _OPS
    ops, why = _try_inrepo()
    if ops is None:
        ops, why_wheel = _try_wheel()
        if ops is None:
            raise RuntimeError(f"{why}; {why_wheel}")
    logger.info(
        "[dspark-markov-head] using %s backend for markov-head ops",
        "in-repo csrc build (torch.ops._C_ascend)" if ops.backend == "csrc"
        else "standalone whl (torch.ops.npu)",
    )
    _OPS = ops
    return _OPS


def markov_step(
    logits: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    ws_ag: torch.Tensor,
    seed_line: torch.Tensor,
    tok_out: torch.Tensor,
    r0: int,
    world: int,
    step_k: int,
    mode: int,
) -> None:
    """One L3 markov step on this rank's vocab shard (``dspark_mh3_step``).

    In-place / in-out args follow the operator contract verbatim; see
    ``csrc/spec_decode/markov_head/op_host/dspark_mh_ops.cpp``.
    """
    markov_head_ops().step(logits, w1, w2, ws_ag, seed_line, tok_out, r0, world, step_k, mode)


def markov_merge(ws_ag: torch.Tensor, tok_out: torch.Tensor, B: int, world: int) -> None:
    """Final-column merge of the gathered candidate keys (``dspark_mh3_merge``)."""
    markov_head_ops().merge(ws_ag, tok_out, B, world)

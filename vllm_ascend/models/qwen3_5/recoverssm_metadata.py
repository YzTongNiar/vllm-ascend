# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend metadata builder for upstream Qwen GDN RecoverSSM."""

from dataclasses import fields
from typing import Any

import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.gdn_recoverssm import (
    GDNRecoverSSMMetadata,
    GDNRecoverSSMCommitMetadata,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    recoverssm_request_indices,
    recoverssm_spec_rows,
)

from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder


class AscendGDNRecoverSSMMetadataBuilder(AscendGDNHostMetadataBuilder):
    """Use Ascend host descriptors while retaining upstream GDN replay semantics."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self._ones = torch.ones(max_num_reqs, dtype=torch.int32, device=device)
        self._recover_context: Any = None
        self._recover_capturing = False
        self._recover_logged = False

    def _get_recover_context(self):
        if self._recover_context is None:
            from vllm_ascend.ops.triton.gdn.recoverssm import GDNRecoverSSMCommitContext

            static_context = self.vllm_config.compilation_config.static_forward_context
            layers = [static_context[name] for name in self.layer_names]
            self._recover_context = GDNRecoverSSMCommitContext.create(
                layers,
                spec_query_len=1 + self.num_spec,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
        return self._recover_context

    @staticmethod
    def _normalize_graph_padding(m, num_actual_reqs, draft_counts):
        if num_actual_reqs is None or num_actual_reqs >= m.num_reqs:
            return m, draft_counts
        starts_cpu = m.query_start_loc_cpu.clone()
        starts_cpu[num_actual_reqs + 1 :] = starts_cpu[num_actual_reqs]
        starts = torch.minimum(m.query_start_loc, m.query_start_loc[num_actual_reqs])
        m = m.replace(
            query_start_loc_cpu=starts_cpu,
            query_start_loc=starts,
            num_actual_tokens=int(starts_cpu[num_actual_reqs]),
        )
        if draft_counts is not None:
            draft_counts = draft_counts.clone()
            draft_counts[num_actual_reqs:] = -1
        return m, draft_counts

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
        num_actual_reqs=None,
    ):
        m, draft_counts = self._normalize_graph_padding(
            common_attn_metadata, num_actual_reqs, num_decode_draft_tokens_cpu
        )
        rows = None
        if self.use_spec_decode and draft_counts is not None:
            if self._recover_capturing and m.is_prefilling is None:
                m = m.replace(is_prefilling=torch.zeros(m.num_reqs, dtype=torch.bool))
            if m.is_prefilling is None or m.is_prefilling.device.type != "cpu":
                raise ValueError("Qwen GDN RecoverSSM needs the CPU is_prefilling mask")
            draft_counts, rows = recoverssm_spec_rows(
                m.is_prefilling,
                m.query_start_loc_cpu,
                draft_counts,
                m.num_reqs,
            )

        meta = super().build(
            common_prefix_len,
            m,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=draft_counts,
            fast_build=fast_build,
            num_actual_reqs=num_actual_reqs,
        )
        if meta.num_spec_decodes == 0:
            return meta
        if rows is None or meta.spec_state_indices_tensor is None:
            raise ValueError("Qwen GDN RecoverSSM: spec rows without draft counts")

        base = {f.name: getattr(meta, f.name) for f in fields(meta)}
        state_indices = meta.spec_state_indices_tensor
        base["spec_state_indices_tensor"] = state_indices[:, :1]
        base["num_accepted_tokens"] = self._ones[: state_indices.shape[0]]
        n = meta.num_spec_decodes
        request_indices = recoverssm_request_indices(rows, n, m.query_start_loc.device)
        align = self.vllm_config.cache_config.mamba_cache_mode == "align"
        commit = GDNRecoverSSMCommitMetadata(
            state_indices=base["spec_state_indices_tensor"],
            query_start_loc=meta.spec_query_start_loc,
            request_indices=request_indices,
            block_table=m.block_table_tensor if align else None,
            num_computed_tokens=m.compute_num_computed_tokens() if align else None,
            block_size=self.kv_cache_spec.block_size if align else None,
        )
        if not self._recover_logged:
            self._recover_logged = True
        return GDNRecoverSSMMetadata(
            **base,
            recoverssm_commit=commit,
            recoverssm_context=self._get_recover_context(),
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self._recover_capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._recover_capturing = False


__all__ = ["AscendGDNRecoverSSMMetadataBuilder"]

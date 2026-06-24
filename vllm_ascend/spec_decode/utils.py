# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def compute_sliding_window_block_table(
    full_block_table: torch.Tensor,
    full_seq_lens: torch.Tensor,
    *,
    num_speculative_tokens: int,
    window_size: int,
    block_size: int,
    max_window_blocks: int,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized, sync-free computation of the cropped sliding-window block table.

    For each request, the ``ceil((seq_len + K - start) / B)`` most recent blocks
    of ``full_block_table`` are gathered into ``out[:num_reqs]``, zero-padded to
    ``max_window_blocks`` columns. This reproduces the original per-row loop:

        window[i, :needed] = full_block_table[i, start_idx : start_idx + needed]

    (indices beyond the full table, or beyond ``needed``, are left as 0 -- those
    trailing block ids are never read because draft attention is bounded by
    ``seq_lens``).

    Two properties matter for ACL graph (FULL) mode, both guaranteed here:

    1. **No host sync / no Python data-dependent loop** -- the gather uses pure
       tensor ops (``torch.arange`` + ``torch.gather`` + masking), so it is safe
       to run on-device without ``.item()``.
    2. **Stable buffer** -- the result is written into the caller-provided
       ``out`` (a pre-allocated buffer) rather than freshly allocated, so the
       caller can keep an identical ``data_ptr`` across graph capture and replay.

    Args:
        full_block_table: ``[num_reqs, full_cols]`` block table, int32.
        full_seq_lens: ``[num_reqs]`` sequence lengths (before the K draft steps).
        num_speculative_tokens: K, the number of speculative tokens.
        window_size: W, the sliding window size in tokens.
        block_size: B, the block size.
        max_window_blocks: fixed column count of the output window table
            (constant across graph capture and runtime).
        out: pre-allocated ``[>= num_reqs, max_window_blocks]`` buffer; the
            gathered window table is written into ``out[:num_reqs]`` in place.

    Returns:
        ``(start_block_indices, needed_blocks_per_req)`` -- ``start_block_indices``
        is reused by the caller for slot-mapping; ``needed_blocks_per_req`` is
        returned for completeness/testing.
    """
    num_reqs = full_seq_lens.shape[0]
    k = num_speculative_tokens
    w = window_size
    b = block_size

    final_seq_lens = full_seq_lens + k
    start_tokens = (final_seq_lens - w).clamp(min=0)
    start_blocks = (start_tokens // b) * b
    start_block_indices = (start_blocks // b).to(torch.int64)

    tokens_to_cover = final_seq_lens - start_blocks
    needed_blocks_per_req = ((tokens_to_cover + b - 1) // b).to(torch.int64)

    full_cols = full_block_table.shape[1]
    # column offset grid [1, max_window_blocks]
    cols = torch.arange(max_window_blocks, device=full_block_table.device).unsqueeze(0)
    # source column per (row, col): start_block_indices[:, None] + cols
    src_cols = start_block_indices.unsqueeze(1) + cols
    # clamp to the valid full-block-table column range so gather never goes OOB
    src_cols_clamped = src_cols.clamp(max=full_cols - 1)

    gathered = torch.gather(full_block_table, 1, src_cols_clamped)
    needed = torch.clamp(needed_blocks_per_req, max=max_window_blocks).unsqueeze(1)
    # keep only columns within `needed` and within the full table; zero the rest
    valid_mask = (cols < needed) & (src_cols < full_cols)
    out[:num_reqs].copy_(gathered * valid_mask.to(gathered.dtype))

    return start_block_indices, needed_blocks_per_req


def update_num_computed_tokens_for_batch_change(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    """Correct num_computed_tokens for async spec decode drift.

    Requests that had drafts: corrected = prev_gpu + valid_count.
    New requests or non-draft (e.g. prefills): use CPU value directly.
    """
    # Clamp because prev_positions can be -1 for new requests
    gather_indices = prev_positions.clamp(min=0)

    valid_counts = valid_sampled_token_count[gather_indices]
    prev_computed = num_computed_tokens[gather_indices]
    prev_drafts = prev_num_draft_tokens[gather_indices]

    participating = (prev_positions >= 0) & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.int()

    n = prev_positions.shape[0]
    num_computed_tokens[:n].copy_(torch.where(participating, corrected, cpu_num_computed_tokens))
    num_accepted_tokens.copy_(torch.where(participating, valid_counts, num_accepted_tokens))

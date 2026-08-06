# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_initial_states_kernel(
    state_ptr,
    indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    stride_state_batch,
    stride_indices,
    stride_has_initial_state,
    row_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    has_initial_state = tl.load(
        has_initial_state_ptr + batch_idx * stride_has_initial_state
    ).to(tl.int1)
    state_idx = tl.load(indices_ptr + batch_idx * stride_indices).to(tl.int64)
    state_idx = tl.where(has_initial_state, state_idx, 0)
    values = tl.load(
        state_ptr + state_idx * stride_state_batch + offsets,
        mask=mask & has_initial_state,
        other=0.0,
    )
    tl.store(output_ptr + batch_idx * row_size + offsets, values, mask=mask)


def gather_initial_states(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Gather dense state rows, replacing uninitialized rows with zeros."""
    assert state.ndim >= 2
    assert state.is_cuda
    assert indices.ndim == 1 and has_initial_state.ndim == 1
    assert indices.shape == has_initial_state.shape
    assert indices.device == state.device
    assert has_initial_state.device == state.device
    assert indices.dtype in (torch.int32, torch.int64)
    assert has_initial_state.dtype == torch.bool

    row_size = state[0].numel()
    # Mamba pages may pad stride(0), but each state row remains dense.
    assert state[0].is_contiguous()
    # k3-kda guard: an out-of-range state index makes the kernel form an OOB GPU
    # address (state_ptr + idx*stride) -> "Memory access fault", even where the
    # value load is has_initial_state-masked. Under 2P/2D disagg the producer
    # prefill has been observed to carry indices >= state.shape[0]; clamp the
    # *effective* index to 0 wherever has_initial_state is False (a fresh prefill
    # has no prior state to gather anyway) and hard-clamp any stray index into
    # range so the address stays valid. Logs once if it fires.
    _n_state_blocks = int(state.shape[0])
    _safe_idx = torch.where(
        has_initial_state,
        indices.to(torch.int64).clamp_(0, _n_state_blocks - 1),
        torch.zeros_like(indices, dtype=torch.int64),
    )
    if bool((indices >= _n_state_blocks).any()) or bool((indices < 0).any()):
        import logging as _lg
        _bad = indices[(indices >= _n_state_blocks) | (indices < 0)]
        _lg.getLogger(__name__).warning(
            "[k3-kda gather] clamped %d out-of-range state idx (n_blocks=%d, "
            "sample=%s); disagg producer prefill likely mis-flagged initial state.",
            int(_bad.numel()), _n_state_blocks, _bad[:8].tolist(),
        )
    indices = _safe_idx
    output = torch.empty(
        (indices.numel(), *state.shape[1:]),
        dtype=state.dtype,
        device=state.device,
    )
    block_size = min(triton.next_power_of_2(row_size), 1024)
    grid = (triton.cdiv(row_size, block_size), indices.numel())
    _gather_initial_states_kernel[grid](
        state,
        indices,
        has_initial_state,
        output,
        state.stride(0),
        indices.stride(0),
        has_initial_state.stride(0),
        row_size=row_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return output

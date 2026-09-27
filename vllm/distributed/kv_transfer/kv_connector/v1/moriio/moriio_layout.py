import os  # GLM53_INDEXER_KBPB
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Mapping
from typing import NamedTuple

import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)


class LayerTransferGeometry(NamedTuple):
    num_blocks: int
    block_size: int
    block_len: int
    slot_size_bytes: int
    block_stride: int
    local_kv_stride: int | None
    remote_kv_stride: int | None
    transfers_per_block: int
    regions_per_block: int
    split_kv_regions: bool


def build_layer_to_spec(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    layer_to_spec: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            layer_to_spec.update(
                {
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in group.layer_names
                }
            )
        else:
            layer_to_spec.update(
                {layer_name: group_spec for layer_name in group.layer_names}
            )
    return layer_to_spec


def is_mla_cache_layer(
    layer_to_spec: Mapping[str, KVCacheSpec], layer_name: str
) -> bool:
    try:
        spec = layer_to_spec[layer_name]
    except KeyError as e:
        raise ValueError(f"Missing KV cache spec for layer {layer_name}") from e
    return isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))


def _spec_dim_matches(value: int, expected: int | None) -> bool:
    return expected is None or value == expected


def _kernel_layout_matches(
    spec: KVCacheSpec, kernel_block_size: int, num_kv_heads: int, head_dim: int
) -> bool:
    if kernel_block_size <= 0 or spec.block_size % kernel_block_size != 0:
        return False
    return _spec_dim_matches(
        num_kv_heads, getattr(spec, "num_kv_heads", None)
    ) and _spec_dim_matches(head_dim, getattr(spec, "head_size", None))


def _select_kernel_block_layout(
    layer_name: str, shape: torch.Size, spec: KVCacheSpec
) -> tuple[int, int, int]:
    axis2_matches = _kernel_layout_matches(spec, shape[2], shape[3], shape[4])
    axis3_matches = _kernel_layout_matches(spec, shape[3], shape[2], shape[4])

    if axis2_matches and axis3_matches and shape[2] != shape[3]:
        raise ValueError(
            f"Ambiguous MoRIIO kernel-block K/V cache shape for layer "
            f"{layer_name}: {tuple(shape)}"
        )
    if axis2_matches:
        return shape[2], shape[3], shape[4]
    if axis3_matches:
        return shape[3], shape[2], shape[4]

    raise ValueError(
        f"Unsupported MoRIIO K/V cache shape for layer {layer_name}: "
        f"{tuple(shape)} does not contain block size {spec.block_size}"
    )


def get_layer_transfer_geometry(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
    remote_num_blocks: int | None = None,
) -> LayerTransferGeometry:
    shape = kv_cache.shape
    stride = kv_cache.stride()
    element_size = kv_cache.element_size()
    spec = layer_to_spec[layer_name]
    is_mla_cache = is_mla_cache_layer(layer_to_spec, layer_name)

    if not is_mla_cache and len(shape) == 5 and shape[0] == 2:
        _, num_blocks = shape[:2]
        kernel_blocks_per_block = 1
        if shape[2] == spec.block_size:
            block_size, num_kv_heads, head_dim = shape[2:]
        elif shape[3] == spec.block_size:
            num_kv_heads, block_size, head_dim = shape[2:]
        else:
            kernel_num_blocks = num_blocks
            kernel_block_size, num_kv_heads, head_dim = _select_kernel_block_layout(
                layer_name, shape, spec
            )
            kernel_blocks_per_block = spec.block_size // kernel_block_size
            if kernel_num_blocks % kernel_blocks_per_block != 0:
                raise ValueError(
                    f"Unsupported MoRIIO K/V cache shape for layer {layer_name}: "
                    f"{tuple(shape)} has {kernel_num_blocks} kernel blocks, "
                    f"not divisible by {kernel_blocks_per_block}"
                )
            num_blocks = kernel_num_blocks // kernel_blocks_per_block
            block_size = spec.block_size
        slot_size_bytes = num_kv_heads * head_dim * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[1] * kernel_blocks_per_block,
            local_kv_stride=stride[0],
            remote_kv_stride=(
                stride[1] * kernel_blocks_per_block * (remote_num_blocks or num_blocks)
            ),
            transfers_per_block=2,
            regions_per_block=1,
            split_kv_regions=True,
        )

    if not is_mla_cache and len(shape) == 5 and shape[1] == 2:
        num_blocks = shape[0]
        if shape[2] == spec.block_size:
            block_size, num_kv_heads, head_dim = shape[2:]
            slot_size_bytes = num_kv_heads * head_dim * element_size
            block_len = block_size * slot_size_bytes
            return LayerTransferGeometry(
                num_blocks=num_blocks,
                block_size=block_size,
                block_len=block_len,
                slot_size_bytes=slot_size_bytes,
                block_stride=stride[0],
                local_kv_stride=stride[1],
                remote_kv_stride=stride[1],
                transfers_per_block=2,
                regions_per_block=2,
                split_kv_regions=False,
            )
        elif shape[3] == spec.block_size:
            num_kv_heads, block_size, head_dim = shape[2:]
        else:
            kernel_num_blocks = num_blocks
            kernel_block_size, _, _ = _select_kernel_block_layout(
                layer_name, shape, spec
            )
            kernel_blocks_per_block = spec.block_size // kernel_block_size
            if kernel_num_blocks % kernel_blocks_per_block != 0:
                raise ValueError(
                    f"Unsupported MoRIIO K/V cache shape for layer {layer_name}: "
                    f"{tuple(shape)} has {kernel_num_blocks} kernel blocks, "
                    f"not divisible by {kernel_blocks_per_block}"
                )
            num_blocks = kernel_num_blocks // kernel_blocks_per_block
            block_size = spec.block_size
            block_stride = stride[0] * kernel_blocks_per_block
            block_len = block_stride * element_size
            slot_size_bytes = block_len // block_size
            return LayerTransferGeometry(
                num_blocks=num_blocks,
                block_size=block_size,
                block_len=block_len,
                slot_size_bytes=slot_size_bytes,
                block_stride=block_stride,
                local_kv_stride=None,
                remote_kv_stride=None,
                transfers_per_block=1,
                regions_per_block=1,
                split_kv_regions=False,
            )
        slot_size_bytes = num_kv_heads * head_dim * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[0],
            local_kv_stride=stride[1],
            remote_kv_stride=stride[1],
            transfers_per_block=2,
            regions_per_block=2,
            split_kv_regions=False,
        )

    if (
        isinstance(spec, AttentionSpec)
        and len(shape) == 4
        and shape[1] == spec.num_heads
        and shape[2] == spec.num_states
        and shape[3] * element_size == spec.state_content_size_bytes
    ):
        # Standardized per-layer [B, H, N, C] view (MLA is just H == 1).
        num_blocks, num_heads, num_states, content_dim = shape
        slot_size_bytes = num_heads * content_dim * element_size
        block_len = num_states * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=spec.block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[0],
            local_kv_stride=None,
            remote_kv_stride=None,
            transfers_per_block=1,
            regions_per_block=1,
            split_kv_regions=False,
        )

    if len(shape) == 4 and shape[1] == 1:
        # Single-head-slot packed 4-D cache fallback for hybrid / packed caches
        # whose physical [B, H=1, N, C] layout the standardized branch above
        # rejects. Two such caches occur in GLM-5.3-Flash / DeepSeek-V3.2 DSA on
        # ROCm, and the earlier standardized branch fails on each for a reason
        # unrelated to the actual byte layout:
        #   * the sparse-indexer k_cache, e.g. shape (42453, 1, 32, 132),
        #     stride (4224, 132, 132, 1), MLAAttentionSpec whose *logical*
        #     num_states (288) != the physically packed shape[2] (32), so
        #     ``shape[2] == spec.num_states`` fails; and
        #   * the linear-attention (Mamba / gated-delta / KDA) recurrent-state
        #     cache, e.g. shape (4717, 1, 1, 1085440), stride
        #     (1179648, 1085440, 1085440, 1), a non-AttentionSpec MambaSpec
        #     whose page is padded (here by 8.68%, so stride[0] > prod(shape[1:]))
        #     to match the attention page, and which the AttentionSpec-typed
        #     branch never matches at all.
        # Both are one contiguous (optionally padded) region per block, so
        # transfer the meaningful bytes per block over the natural block stride:
        # block_len counts only the real per-block content (prod(shape[1:]))
        # while block_stride keeps the padded stride[0] so per-block offsets and
        # the registered span both skip the inter-block padding. This is
        # byte-identical to the standardized [B, H, N, C] branch for an
        # unpadded, spec-matching MLA tensor, so it is a safe superset.
        # ``shape[1] == 1`` (single head slot) is the structural gate: genuine
        # dense K/V caches that need two regions / split-KV transfers are 5-D
        # (shape[0] == 2 or shape[1] == 2) and are handled above, never reaching
        # here; real multi-head [B, H>1, N, C] MLA also matches the standardized
        # branch above first.
        num_blocks = shape[0]
        slot_size_bytes = shape[2] * shape[3] * element_size
        block_len = shape[1] * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=spec.block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[0],
            local_kv_stride=None,
            remote_kv_stride=None,
            transfers_per_block=1,
            regions_per_block=1,
            split_kv_regions=False,
        )

    cache_kind = "MLA" if is_mla_cache else "K/V"
    raise ValueError(
        f"Unsupported MoRIIO {cache_kind} cache shape for layer "
        f"{layer_name}: {tuple(shape)}"
    )


def iter_layer_registration_regions(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
) -> list[tuple[torch.Tensor, int]]:
    geometry = get_layer_transfer_geometry(layer_name, kv_cache, layer_to_spec)
    region_len = geometry.num_blocks * geometry.regions_per_block * geometry.block_len
    if geometry.regions_per_block == 1:
        # With padded or interleaved pages the block stride exceeds the
        # meaningful block_len; register the strided span. The span ends with
        # the last block's meaningful bytes, not a whole stride, so it never
        # runs past the backing allocation of a strided layer view.
        block_stride_bytes = geometry.block_stride * kv_cache.element_size()
        region_len = max(
            region_len,
            (geometry.num_blocks - 1) * block_stride_bytes + geometry.block_len,
        )
    if geometry.split_kv_regions:
        return [(cache, region_len) for cache in kv_cache]
    return [(kv_cache, region_len)]


def merge_contiguous_offsets(
    offsets_local: list[int],
    offsets_remote: list[int],
    sizes: list[int],
) -> tuple[list[int], list[int], list[int]]:
    if not offsets_local:
        return [], [], []
    if not (len(offsets_local) == len(offsets_remote) == len(sizes)):
        raise ValueError("Input list lengths mismatch")

    rows = sorted(zip(offsets_local, offsets_remote, sizes), key=lambda row: row[0])
    merged: list[list[int]] = []
    for local, remote, size in rows:
        if (
            merged
            and local == merged[-1][0] + merged[-1][2]
            and remote == merged[-1][1] + merged[-1][2]
        ):
            merged[-1][2] += size
        else:
            merged.append([local, remote, size])

    return (
        [row[0] for row in merged],
        [row[1] for row in merged],
        [row[2] for row in merged],
    )


def _kernel_blocks_per_group_block(num_kernel_blocks, ref_group_blocks) -> int:
    """GLM53_INDEXER_KBPB: kernel-blocks-per-group-block (kbpb).
    Returns num_kernel_blocks/ref_group_blocks when it divides cleanly and
    num_kernel_blocks>ref (the finely-paged DSA indexer k_cache); else 1
    (byte-identical no-op for every 1:1 layer).
    """
    try:
        if (ref_group_blocks and ref_group_blocks > 0 and num_kernel_blocks
                and num_kernel_blocks > ref_group_blocks
                and num_kernel_blocks % ref_group_blocks == 0):
            return max(1, num_kernel_blocks // ref_group_blocks)
    except Exception:
        pass
    return 1


def compute_block_transfer_offsets(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
    local_block_ids: list[int],
    remote_block_ids: list[int],
    remote_num_blocks: int,
    merge_fn: Callable[
        [list[int], list[int], list[int]], tuple[list[int], list[int], list[int]]
    ] = merge_contiguous_offsets,
    local_num_blocks: int | None = None,
    remote_ref_blocks: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    # A shorter (or empty) local list is the READ-mode "drop the transfer, just
    # free the prefill blocks" case (full-prefix-hit / aborted-before-scheduled):
    # decode pulls fewer blocks than the prefill holds. The zip loop below pairs
    # local[i]<->remote[i] and sizes by len(local), so a short local transfers
    # only what decode allocated and an empty local is a no-op. A longer local
    # list is a genuine bug and still fails loudly.
    if len(local_block_ids) > len(remote_block_ids):
        raise ValueError(
            "local_block_ids longer than remote_block_ids: "
            f"{len(local_block_ids)} > {len(remote_block_ids)}"
        )
    geometry = get_layer_transfer_geometry(
        layer_name, kv_cache, layer_to_spec, remote_num_blocks
    )
    element_size = kv_cache.element_size()
    transfer_size_byte = geometry.block_len
    per_block = geometry.transfers_per_block

    # GLM53_INDEXER_KBPB: expand GROUP block-ids into kernel sub-blocks for the
    # finely-paged single-region DSA sparse-indexer k_cache. Per-side because
    # local/remote kernel-page counts differ but share the group ratio. kbpb==1
    # (every 1:1 layer) leaves the id lists and all downstream offsets identical.
    if per_block == 1:
        _lk = _kernel_blocks_per_group_block(geometry.num_blocks, local_num_blocks)
        _rk = _kernel_blocks_per_group_block(remote_num_blocks, remote_ref_blocks)
        # GLM53_INDEXER_KBPB: the peer advertises ONLY its GROUP-block scalar
        # (remote_num_blocks == remote_ref_blocks), so _rk cannot be derived
        # from it and comes back 1 even for the finely-paged DSA indexer.
        # In symmetric EP8/EP8 both legs run the identical model, so the
        # kernel-blocks-per-group ratio is a MODEL CONSTANT (kbpb=17), the
        # SAME on both sides regardless of each leg's group-block budget.
        # Mirror the side that DID resolve a >1 ratio onto the side that
        # didn't, so local (17N) and remote (17N) id lists stay aligned in
        # the zip. (Without this remote stays N -> misaligned past block 0.)
        if _lk > 1 and _rk <= 1:
            _rk = _lk
        elif _rk > 1 and _lk <= 1:
            _lk = _rk
        if _lk > 1 or _rk > 1:
            if os.environ.get('MORIIO_OFFSET_DBG') == '1':
                print('[GLM53_INDEXER_KBPB] L={} lkbpb={} rkbpb={} nb={} lref={} rnb={} rref={}'
                      .format(layer_name, _lk, _rk, geometry.num_blocks,
                              local_num_blocks, remote_num_blocks, remote_ref_blocks),
                      flush=True)
            local_block_ids = [lb * _lk + j for lb in local_block_ids for j in range(_lk)]
            remote_block_ids = [rb * _rk + j for rb in remote_block_ids for j in range(_rk)]

    total = len(local_block_ids) * per_block
    offset_local = [0] * total
    offset_remote = [0] * total
    sizes = [transfer_size_byte] * total

    w = 0
    for lb, rb in zip(local_block_ids, remote_block_ids):
        offset_local[w] = element_size * (lb * geometry.block_stride)
        offset_remote[w] = element_size * (rb * geometry.block_stride)
        w += 1
        if per_block == 2:
            assert geometry.local_kv_stride is not None
            assert geometry.remote_kv_stride is not None
            offset_local[w] = element_size * (
                geometry.local_kv_stride + lb * geometry.block_stride
            )
            offset_remote[w] = element_size * (
                geometry.remote_kv_stride + rb * geometry.block_stride
            )
            w += 1

    return merge_fn(offset_local, offset_remote, sizes)

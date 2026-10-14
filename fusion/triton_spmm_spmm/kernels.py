"""Triton kernels for consecutive sparse-matrix/dense-matrix products.

The fused kernel uses CSR for the producer multiplication and CSC for the
consumer scatter.  A program owns a small consecutive producer-row tile.  Its
sorted CSC columns are merged in destination order, which combines conflicts
inside the program before issuing a global atomic add.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


# A Triton ``num_warps`` value maps to 32 CUDA threads per warp.  Keeping every
# entry at four warps or fewer enforces the requested 128-thread upper bound.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_ROWS": 1, "BLOCK_FEATURES": 32}, num_warps=1),
    triton.Config({"BLOCK_ROWS": 1, "BLOCK_FEATURES": 64}, num_warps=2),
    triton.Config({"BLOCK_ROWS": 1, "BLOCK_FEATURES": 128}, num_warps=4),
    triton.Config({"BLOCK_ROWS": 2, "BLOCK_FEATURES": 32}, num_warps=2),
    triton.Config({"BLOCK_ROWS": 2, "BLOCK_FEATURES": 64}, num_warps=4),
    triton.Config({"BLOCK_ROWS": 2, "BLOCK_FEATURES": 128}, num_warps=4),
    triton.Config({"BLOCK_ROWS": 4, "BLOCK_FEATURES": 32}, num_warps=4),
    triton.Config({"BLOCK_ROWS": 4, "BLOCK_FEATURES": 64}, num_warps=4),
    triton.Config({"BLOCK_ROWS": 4, "BLOCK_FEATURES": 128}, num_warps=4),
    triton.Config({"BLOCK_ROWS": 8, "BLOCK_FEATURES": 32}, num_warps=4),
    triton.Config({"BLOCK_ROWS": 8, "BLOCK_FEATURES": 64}, num_warps=4),
    triton.Config({"BLOCK_ROWS": 8, "BLOCK_FEATURES": 128}, num_warps=4),
]

assert max(config.num_warps for config in AUTOTUNE_CONFIGS) * 32 <= 128


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["rows", "features", "nnz"],
    warmup=10,
    rep=30,
)
@triton.jit
def csr_spmm_kernel(
    values_ptr,
    col_indices_ptr,
    row_ptr,
    dense_input_ptr,
    output_ptr,
    rows,
    features,
    nnz,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
):
    """Compute one CSR SpMM; used twice by the unfused implementation."""
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    feature_offsets = (
        tl.program_id(1) * BLOCK_FEATURES + tl.arange(0, BLOCK_FEATURES)
    )
    valid_rows = row_offsets < rows
    valid_features = feature_offsets < features

    starts = tl.load(row_ptr + row_offsets, mask=valid_rows, other=0)
    ends = tl.load(row_ptr + row_offsets + 1, mask=valid_rows, other=0)
    cursors = starts
    accumulators = tl.zeros((BLOCK_ROWS, BLOCK_FEATURES), tl.float32)

    # Rows advance together.  BLOCK_ROWS is autotuned so the tuner can trade
    # row-level parallelism against divergence from unequal row lengths.
    while tl.sum((cursors < ends).to(tl.int32), axis=0) > 0:
        valid_edges = valid_rows & (cursors < ends)
        columns = tl.load(
            col_indices_ptr + cursors, mask=valid_edges, other=0
        )
        weights = tl.load(values_ptr + cursors, mask=valid_edges, other=0.0)
        dense = tl.load(
            dense_input_ptr
            + columns[:, None] * features
            + feature_offsets[None, :],
            mask=valid_edges[:, None] & valid_features[None, :],
            other=0.0,
        )
        accumulators += weights[:, None] * dense
        cursors += valid_edges.to(tl.int32)

    tl.store(
        output_ptr
        + row_offsets[:, None] * features
        + feature_offsets[None, :],
        accumulators,
        mask=valid_rows[:, None] & valid_features[None, :],
    )


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["rows", "features", "nnz"],
    reset_to_zero=["output_ptr"],
    warmup=10,
    rep=30,
)
@triton.jit
def csr_csc_fused_conflict_reduced_kernel(
    producer_values_ptr,
    producer_col_indices_ptr,
    producer_row_ptr,
    consumer_values_ptr,
    consumer_row_indices_ptr,
    consumer_col_ptr,
    dense_input_ptr,
    intermediate_ptr,
    output_ptr,
    rows,
    features,
    nnz,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
):
    """Fuse CSR SpMM into a conflict-reduced CSC scatter.

    CSC row indices are sorted.  The k-way merge below combines all consumer
    edges in this producer-row tile that share a destination.  Contributions
    from different programs still require an atomic operation.
    """
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    feature_offsets = (
        tl.program_id(1) * BLOCK_FEATURES + tl.arange(0, BLOCK_FEATURES)
    )
    valid_rows = row_offsets < rows
    valid_features = feature_offsets < features

    producer_starts = tl.load(
        producer_row_ptr + row_offsets, mask=valid_rows, other=0
    )
    producer_ends = tl.load(
        producer_row_ptr + row_offsets + 1, mask=valid_rows, other=0
    )
    producer_cursors = producer_starts
    intermediates = tl.zeros((BLOCK_ROWS, BLOCK_FEATURES), tl.float32)

    while tl.sum(
        (producer_cursors < producer_ends).to(tl.int32), axis=0
    ) > 0:
        valid_edges = valid_rows & (producer_cursors < producer_ends)
        columns = tl.load(
            producer_col_indices_ptr + producer_cursors,
            mask=valid_edges,
            other=0,
        )
        weights = tl.load(
            producer_values_ptr + producer_cursors,
            mask=valid_edges,
            other=0.0,
        )
        dense = tl.load(
            dense_input_ptr
            + columns[:, None] * features
            + feature_offsets[None, :],
            mask=valid_edges[:, None] & valid_features[None, :],
            other=0.0,
        )
        intermediates += weights[:, None] * dense
        producer_cursors += valid_edges.to(tl.int32)

    tl.store(
        intermediate_ptr
        + row_offsets[:, None] * features
        + feature_offsets[None, :],
        intermediates,
        mask=valid_rows[:, None] & valid_features[None, :],
    )

    consumer_cursors = tl.load(
        consumer_col_ptr + row_offsets, mask=valid_rows, other=0
    )
    consumer_ends = tl.load(
        consumer_col_ptr + row_offsets + 1, mask=valid_rows, other=0
    )

    # Each iteration consumes the smallest destination currently visible from
    # the BLOCK_ROWS sorted CSC columns.  Equal destinations are locally
    # reduced across rows, changing several global atomics into one.
    while tl.sum(
        (consumer_cursors < consumer_ends).to(tl.int32), axis=0
    ) > 0:
        valid_edges = valid_rows & (consumer_cursors < consumer_ends)
        destinations = tl.load(
            consumer_row_indices_ptr + consumer_cursors,
            mask=valid_edges,
            other=2147483647,
        )
        destination = tl.min(destinations, axis=0)
        matching_edges = valid_edges & (destinations == destination)
        weights = tl.load(
            consumer_values_ptr + consumer_cursors,
            mask=matching_edges,
            other=0.0,
        )
        contribution = tl.sum(
            tl.where(
                matching_edges[:, None],
                weights[:, None] * intermediates,
                0.0,
            ),
            axis=0,
        )
        tl.atomic_add(
            output_ptr + destination * features + feature_offsets,
            contribution,
            mask=valid_features,
        )
        consumer_cursors += matching_edges.to(tl.int32)


@dataclass(frozen=True)
class SelectedConfig:
    block_rows: int
    block_features: int
    num_warps: int

    @property
    def threads_per_block(self) -> int:
        return self.num_warps * 32


def _selected_config(kernel) -> SelectedConfig:
    config = kernel.best_config
    selected = SelectedConfig(
        block_rows=int(config.kwargs["BLOCK_ROWS"]),
        block_features=int(config.kwargs["BLOCK_FEATURES"]),
        num_warps=int(config.num_warps),
    )
    if selected.threads_per_block > 128:
        raise RuntimeError("autotuner selected more than 128 CUDA threads")
    return selected


def _check_inputs(values, indices, pointers, dense_input):
    if not all(tensor.is_cuda for tensor in (values, indices, pointers, dense_input)):
        raise ValueError("all Triton inputs must be CUDA tensors")
    if values.dtype != torch.float32 or dense_input.dtype != torch.float32:
        raise TypeError("matrix values and dense input must be float32")
    if indices.dtype != torch.int32 or pointers.dtype != torch.int32:
        raise TypeError("sparse indices and pointers must be int32")
    if not dense_input.is_contiguous():
        raise ValueError("dense input must be row-major contiguous")


def launch_unfused(
    csr_values,
    csr_indices,
    csr_ptr,
    dense_input,
    intermediate,
    output,
    return_config=True,
):
    """Launch two separately autotuned/cached CSR SpMM kernels."""
    _check_inputs(csr_values, csr_indices, csr_ptr, dense_input)
    rows, features = dense_input.shape
    nnz = csr_values.numel()
    grid = lambda meta: (
        triton.cdiv(rows, meta["BLOCK_ROWS"]),
        triton.cdiv(features, meta["BLOCK_FEATURES"]),
    )
    csr_spmm_kernel[grid](
        csr_values,
        csr_indices,
        csr_ptr,
        dense_input,
        intermediate,
        rows,
        features,
        nnz,
    )
    csr_spmm_kernel[grid](
        csr_values,
        csr_indices,
        csr_ptr,
        intermediate,
        output,
        rows,
        features,
        nnz,
    )
    if return_config:
        return _selected_config(csr_spmm_kernel)
    return None


def launch_fused(
    csr_values,
    csr_indices,
    csr_ptr,
    csc_values,
    csc_indices,
    csc_ptr,
    dense_input,
    intermediate,
    output,
    reset_output=True,
    return_config=True,
):
    """Launch the autotuned fused CSR-CSC conflict-reduced kernel."""
    _check_inputs(csr_values, csr_indices, csr_ptr, dense_input)
    _check_inputs(csc_values, csc_indices, csc_ptr, dense_input)
    rows, features = dense_input.shape
    nnz = csr_values.numel()
    if reset_output:
        output.zero_()
    grid = lambda meta: (
        triton.cdiv(rows, meta["BLOCK_ROWS"]),
        triton.cdiv(features, meta["BLOCK_FEATURES"]),
    )
    csr_csc_fused_conflict_reduced_kernel[grid](
        csr_values,
        csr_indices,
        csr_ptr,
        csc_values,
        csc_indices,
        csc_ptr,
        dense_input,
        intermediate,
        output,
        rows,
        features,
        nnz,
    )
    if return_config:
        return _selected_config(csr_csc_fused_conflict_reduced_kernel)
    return None

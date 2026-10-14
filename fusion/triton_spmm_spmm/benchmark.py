#!/usr/bin/env python3
"""Autotune and benchmark unfused and fused Triton SpMM-SpMM kernels."""

import argparse
import csv
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from datasets import (
    DEFAULT_DATASETS,
    load_csr,
    resolve_datasets,
    resolve_rcm_datasets,
)


RESULT_FIELDS = (
    "dataset",
    "ordering",
    "rows",
    "nnz",
    "features",
    "implementation",
    "median_ms",
    "normalized_to_unfused",
    "speedup_vs_unfused",
    "block_rows",
    "block_features",
    "num_warps",
    "threads_per_block",
    "correct",
    "max_abs_error",
    "matrix_path",
)


def _device_sparse(matrix, device):
    csc = matrix.tocsc()
    csc.sum_duplicates()
    csc.sort_indices()

    def tensor(array, dtype):
        contiguous = np.ascontiguousarray(array, dtype=dtype)
        return torch.from_numpy(contiguous).to(device=device, non_blocking=False)

    result = {
        "csr_values": tensor(matrix.data, np.float32),
        "csr_indices": tensor(matrix.indices, np.int32),
        "csr_ptr": tensor(matrix.indptr, np.int32),
        "csc_values": tensor(csc.data, np.float32),
        "csc_indices": tensor(csc.indices, np.int32),
        "csc_ptr": tensor(csc.indptr, np.int32),
    }
    result["csr_tensor"] = torch.sparse_csr_tensor(
        result["csr_ptr"],
        result["csr_indices"],
        result["csr_values"],
        size=matrix.shape,
        dtype=torch.float32,
        device=device,
        check_invariants=False,
    )
    return result


def _benchmark(launch, prepare=None, warmup=5, repetitions=30):
    prepare = prepare or (lambda: None)
    for _ in range(warmup):
        prepare()
        launch()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, end in zip(starts, ends):
        prepare()
        start.record()
        launch()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))


def _compare_chunked(reference, result, row_chunk=65536):
    if reference.shape != result.shape:
        return False, float("inf")
    correct = True
    maximum = 0.0
    for first in range(0, reference.shape[0], row_chunk):
        ref = reference[first : first + row_chunk]
        got = result[first : first + row_chunk]
        difference = (ref - got).abs().max().item()
        maximum = max(maximum, difference)
        correct = correct and torch.allclose(ref, got, rtol=2e-3, atol=2e-3)
    return correct, maximum


def _make_row(location, rows, nnz, features, implementation, elapsed, baseline,
              config, correct, maximum):
    return {
        "dataset": location.name,
        "ordering": location.ordering,
        "rows": rows,
        "nnz": nnz,
        "features": features,
        "implementation": implementation,
        "median_ms": f"{elapsed:.6f}",
        "normalized_to_unfused": f"{elapsed / baseline:.6f}",
        "speedup_vs_unfused": f"{baseline / elapsed:.6f}",
        "block_rows": "" if config is None else config.block_rows,
        "block_features": "" if config is None else config.block_features,
        "num_warps": "" if config is None else config.num_warps,
        "threads_per_block": "" if config is None else config.threads_per_block,
        "correct": int(correct),
        "max_abs_error": f"{maximum:.8g}",
        "matrix_path": str(location.path),
    }


def run_case(location, matrix, sparse, features, args, kernels):
    rows = matrix.shape[0]
    nnz = matrix.nnz
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    dense_input = torch.rand(
        (rows, features),
        dtype=torch.float32,
        device=args.device,
        generator=generator,
    )

    unfused_intermediate = torch.empty_like(dense_input)
    unfused_output = torch.empty_like(dense_input)

    def launch_unfused(return_config=True):
        return kernels.launch_unfused(
            sparse["csr_values"], sparse["csr_indices"], sparse["csr_ptr"],
            dense_input, unfused_intermediate, unfused_output,
            return_config=return_config,
        )

    # The first launch invokes Triton's autotuner; it is outside reported time.
    unfused_config = launch_unfused()
    torch.cuda.synchronize()
    unfused_ms = _benchmark(
        lambda: launch_unfused(return_config=False),
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    del unfused_intermediate

    fused_intermediate = torch.empty_like(dense_input)
    fused_output = torch.empty_like(dense_input)

    def launch_fused(reset_output=True, return_config=True):
        return kernels.launch_fused(
            sparse["csr_values"], sparse["csr_indices"], sparse["csr_ptr"],
            sparse["csc_values"], sparse["csc_indices"], sparse["csc_ptr"],
            dense_input, fused_intermediate, fused_output,
            reset_output=reset_output,
            return_config=return_config,
        )

    fused_config = launch_fused()
    torch.cuda.synchronize()
    fused_ms = _benchmark(
        lambda: launch_fused(reset_output=False, return_config=False),
        # Match the C++ benchmark by excluding output initialization from the
        # kernel interval while still clearing atomic output before every run.
        prepare=fused_output.zero_,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    fused_correct, fused_maximum = _compare_chunked(
        unfused_output, fused_output
    )
    if not fused_correct:
        print(
            f"WARNING: {location.name} F={features} fused failed, "
            f"max error={fused_maximum}",
            file=sys.stderr,
        )
    del fused_intermediate, fused_output

    # torch.addmm with a CUDA CSR left operand dispatches to cuSPARSE.  The
    # output buffers are preallocated and beta=0, matching the two-call C++
    # baseline without timing tensor allocation or initialization.
    cusparse_intermediate = torch.empty_like(dense_input)
    cusparse_output = torch.empty_like(dense_input)

    def launch_cusparse():
        torch.addmm(
            cusparse_intermediate,
            sparse["csr_tensor"],
            dense_input,
            beta=0,
            out=cusparse_intermediate,
        )
        torch.addmm(
            cusparse_output,
            sparse["csr_tensor"],
            cusparse_intermediate,
            beta=0,
            out=cusparse_output,
        )

    launch_cusparse()
    torch.cuda.synchronize()
    cusparse_ms = _benchmark(
        launch_cusparse, warmup=args.warmup, repetitions=args.repetitions
    )
    cusparse_correct, cusparse_maximum = _compare_chunked(
        unfused_output, cusparse_output
    )
    if not cusparse_correct:
        print(
            f"WARNING: {location.name} F={features} cuSPARSE failed, "
            f"max error={cusparse_maximum}",
            file=sys.stderr,
        )

    rows_out = [
        _make_row(
            location, rows, nnz, features, "Triton_Unfused_CSR_CSR",
            unfused_ms, unfused_ms, unfused_config, True, 0.0,
        ),
        _make_row(
            location, rows, nnz, features,
            "Triton_Fused_CSR_CSC_ConflictReduced", fused_ms, unfused_ms,
            fused_config, fused_correct, fused_maximum,
        ),
        _make_row(
            location, rows, nnz, features, "cuSPARSE_Unfused_CSR_CSR",
            cusparse_ms, unfused_ms, None, cusparse_correct,
            cusparse_maximum,
        ),
    ]
    print(
        f"{location.name:14s} F={features:3d} "
        f"unfused={unfused_ms:9.4f} ms "
        f"[{unfused_config.block_rows}x{unfused_config.block_features}, "
        f"{unfused_config.threads_per_block}t]  "
        f"fused={fused_ms:9.4f} ms "
        f"[{fused_config.block_rows}x{fused_config.block_features}, "
        f"{fused_config.threads_per_block}t]  "
        f"cuSPARSE={cusparse_ms:9.4f} ms  "
        f"fused/unfused={unfused_ms / fused_ms:.3f}x "
        f"fused/cuSPARSE={cusparse_ms / fused_ms:.3f}x "
        f"correct={fused_correct and cusparse_correct}",
        flush=True,
    )
    return rows_out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--features", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/triton_spmm_spmm_gnn.csv",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--ordering",
        choices=("original", "rcm"),
        default="original",
        help="benchmark the original graph or its cached/generated RCM ordering",
    )
    args = parser.parse_args()
    if any(feature <= 0 for feature in args.features):
        parser.error("feature sizes must be positive")
    if args.warmup < 1 or args.repetitions < 1:
        parser.error("warmup and repetitions must be positive")
    return args


def main():
    args = parse_args()
    resolver = resolve_rcm_datasets if args.ordering == "rcm" else resolve_datasets
    locations = resolver(args.datasets, allow_download=not args.no_download)
    for location in locations:
        print(f"{location.name:14s} {location.origin:24s} {location.path}")
    if args.prepare_only:
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA PyTorch is required. The repository venv currently contains "
            "a CPU-only PyTorch build; install a CUDA-enabled torch wheel first."
        )
    if not str(args.device).startswith("cuda"):
        raise ValueError("Triton benchmark device must be CUDA")

    # Importing after prepare-only lets data preparation work in a CPU-only
    # environment while retaining one self-contained benchmark entry point.
    import kernels

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        stream.flush()
        for location in locations:
            print(f"Loading {location.name} from {location.path}", flush=True)
            matrix = load_csr(location.path)
            sparse = _device_sparse(matrix, args.device)
            for features in args.features:
                case_rows = run_case(
                    location, matrix, sparse, features, args, kernels
                )
                output_rows.extend(case_rows)
                writer.writerows(case_rows)
                stream.flush()
            del sparse, matrix
            torch.cuda.empty_cache()
    if not all(bool(int(row["correct"])) for row in output_rows):
        print(f"Saved {args.output}, but at least one validation failed")
        return 1
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

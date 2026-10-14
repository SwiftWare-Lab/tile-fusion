#!/usr/bin/env python3
"""Cache an RCM-permuted Matrix Market file and run the GPU SpMM-SpMM demo.

The demo performs CSR-to-CSC conversion and tile coloring during its analysis
phase for both original and RCM-permuted input. This script runs the RCM case.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def cached_rcm_path(matrix_path: Path) -> Path:
    if matrix_path.suffix.lower() != ".mtx":
        raise ValueError("the input matrix must be a .mtx file")
    if matrix_path.stem.endswith("_rcm"):
        return matrix_path
    return matrix_path.with_name(matrix_path.stem + "_rcm.mtx")


def prepare_rcm(matrix_path: Path) -> Path:
    reordered_path = cached_rcm_path(matrix_path)
    if reordered_path.is_file():
        print(f"Using cached RCM matrix: {reordered_path}", file=sys.stderr)
        return reordered_path
    if reordered_path == matrix_path:
        raise FileNotFoundError(matrix_path)

    from scipy.io import mmread, mmwrite
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    matrix = csr_matrix(mmread(matrix_path))
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("RCM for this benchmark requires a square matrix")
    permutation = reverse_cuthill_mckee(matrix, symmetric_mode=False)
    reordered = matrix[permutation, :][:, permutation]

    fd, temporary_name = tempfile.mkstemp(
        prefix=matrix_path.stem + "_rcm.", suffix=".mtx", dir=matrix_path.parent
    )
    os.close(fd)
    try:
        mmwrite(temporary_name, reordered)
        try:
            # Avoid replacing a cache another process finished while we worked.
            os.link(temporary_name, reordered_path)
            print(f"Wrote RCM matrix: {reordered_path}", file=sys.stderr)
        except FileExistsError:
            print(f"Using cached RCM matrix: {reordered_path}", file=sys.stderr)
    finally:
        os.unlink(temporary_name)
    return reordered_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="original Matrix Market file")
    parser.add_argument("benchmark", type=Path, help="spmm_spmm_demo_gpu executable")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--header", action="store_true", help="print CSV header")
    args = parser.parse_args()

    matrix_path = args.matrix.resolve(strict=True)
    benchmark_path = args.benchmark.resolve(strict=True)
    reordered_path = prepare_rcm(matrix_path)
    command = [
        str(benchmark_path), "-sm", str(reordered_path), "-nt",
        str(args.threads), "-bc", str(args.features),
    ]
    if args.header:
        command.append("-ah")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())

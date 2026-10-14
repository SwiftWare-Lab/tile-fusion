#!/usr/bin/env python3
"""Sweep GPU hybrid tile sizes and row-nnz thresholds on end2end matrices.

Rebuild spmm_spmm_demo_gpu before running. The executable's
SPMM_SPMM_HYBRID_ONLY mode runs tile sizes 4, 8, 16, 32. With
--row-parallel-128, each block uses four warps and the tile sweep extends to
64, 128, 256. --cooperative also uses four warps and runs the full hybrid
computation in one cooperative kernel launch. All modes include baselines.
Cooperative execution clears the output inside its timed kernel; the other
variants clear their outputs before their GPU execution timers start.
"""

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1] / "data/end2end-test-data"
HYBRID_NAME = re.compile(
    r"GPU_CSRCSC_Hybrid(?:Colored(?:128)?|Cooperative)_T(\d+)_MaxNNZ_(\d+)"
)
DEFAULT_MATRICES = (
    "cora_full",
    "pubmed",
    "coauthor_cs",
    "deezer_europe",
    "facebook_page_page",
    "github",
    "flickr",
)
FIELDS = (
    "matrix", "ordering", "rows", "nnz", "implementation", "tile_size",
    "max_row_nnz", "mean_ms", "analysis_ms", "speedup_vs_unfused",
    "correct_trials", "trials", "max_abs_error", "colors", "fused_rows",
    "fused_nnz",
)


def matrix_paths(name):
    original = DATA_ROOT / name / (name + ".mtx")
    ordered = DATA_ROOT / (name + "_ordered") / (name + "_ordered.mtx")
    if not original.is_file() or not ordered.is_file():
        raise FileNotFoundError(f"missing matrix pair: {original}, {ordered}")
    return (("original", original), ("ordered", ordered))


def run_matrix(binary, matrix, threads, features, row_parallel_128,
               cooperative):
    env = os.environ.copy()
    env.pop("SPMM_SPMM_HYBRID_ONLY", None)
    env.pop("SPMM_SPMM_HYBRID_128", None)
    env.pop("SPMM_SPMM_HYBRID_COOPERATIVE", None)
    env["SPMM_SPMM_HYBRID_COOPERATIVE" if cooperative else
        ("SPMM_SPMM_HYBRID_128" if row_parallel_128
         else "SPMM_SPMM_HYBRID_ONLY")] = "1"
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    command = [str(binary), "-sm", str(matrix), "-nt", str(threads),
               "-bc", str(features), "-ah"]
    completed = subprocess.run(command, env=env, text=True,
                               capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"benchmark failed for {matrix} (exit {completed.returncode}):\n"
            + completed.stderr[-4000:]
        )
    rows = list(csv.DictReader(completed.stdout.splitlines()))
    if not rows:
        raise RuntimeError(f"no benchmark rows returned for {matrix}")
    return rows


def summarize(raw_rows, name, ordering):
    baseline = next(
        row for row in raw_rows
        if row["Implementation Name"] == "GPU_Unfused_SeqReduceRowBalance"
    )
    trial_count = int(baseline["Number of Trials"])

    def mean_ms(row):
        return 1000.0 * sum(
            float(row[f"Trial{i} Subregion0 Executor"])
            for i in range(trial_count)
        ) / trial_count

    baseline_ms = mean_ms(baseline)
    for row in raw_rows:
        impl = row["Implementation Name"]
        if impl == "CPU_Unfused_Seq":
            continue
        match = HYBRID_NAME.fullmatch(impl)
        trials = int(row["Number of Trials"])
        elapsed_ms = mean_ms(row)
        yield {
            "matrix": name,
            "ordering": ordering,
            "rows": int(float(row["nRows"])),
            "nnz": int(float(row["NNZ"])),
            "implementation": impl,
            "tile_size": int(match.group(1)) if match else "",
            "max_row_nnz": int(match.group(2)) if match else "",
            "mean_ms": f"{elapsed_ms:.6f}",
            "analysis_ms": f"{1000 * float(row['Trial0 Subregion0 Analysis']):.6f}",
            "speedup_vs_unfused": f"{baseline_ms / elapsed_ms:.6f}",
            "correct_trials": sum(row[f"Correct{i}"] == "1"
                                  for i in range(trials)),
            "trials": trials,
            "max_abs_error": max(float(row[f"Error{i}"])
                                 for i in range(trials)),
            "colors": int(float(row["Color Count0"])),
            "fused_rows": int(float(row["Number of Fused Rows0"])),
            "fused_nnz": int(float(row["Number of Fused Nnz0"])),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="rebuilt spmm_spmm_demo_gpu")
    parser.add_argument("output", type=Path, help="summary CSV output")
    parser.add_argument("--matrices", nargs="+", default=DEFAULT_MATRICES,
                        help="matrix pair names (default: seven moderate-sized pairs)")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--row-parallel-128", action="store_true",
                        help="use four warps per block and tiles through 256")
    parser.add_argument("--cooperative", action="store_true",
                        help="run the one-launch cooperative hybrid sweep")
    args = parser.parse_args()
    if args.row_parallel_128 and args.cooperative:
        parser.error("choose either --row-parallel-128 or --cooperative")
    binary = args.binary.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for name in args.matrices:
            for ordering, path in matrix_paths(name):
                print(f"Running {name} ({ordering})", flush=True)
                raw_rows = run_matrix(binary, path, args.threads, args.features,
                                      args.row_parallel_128, args.cooperative)
                summary = list(summarize(raw_rows, name, ordering))
                writer.writerows(summary)
                output.flush()
                hybrid = [row for row in summary if row["tile_size"] != ""]
                expected = 35 if args.row_parallel_128 or args.cooperative else 20
                if len(hybrid) != expected:
                    raise RuntimeError(
                        f"expected {expected} hybrid configurations for {path}, "
                        f"got {len(hybrid)}"
                    )
                print(f"  {len(hybrid)} configurations; "
                      f"{sum(row['correct_trials'] == row['trials'] for row in hybrid)} correct",
                      flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

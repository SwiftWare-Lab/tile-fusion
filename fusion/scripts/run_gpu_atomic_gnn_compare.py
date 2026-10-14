#!/usr/bin/env python3
"""Compare focused GPU SpMM-SpMM paths on original GNN graph matrices.

The atomic rows/block choices come from an earlier tuning sweep. This script
runs fresh benchmark repetitions for CSR-CSC atomic, row-wise unfused, and
cuSPARSE ALG3 only. Trial 0 is omitted from steady-state execution means.
"""

import argparse
import csv
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


FUSION_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = FUSION_ROOT / "data/end2end-test-data"
TUNING_FILE = FUSION_ROOT / "results/gpu_csrcsc_atomic_tuned_16t_32f.csv"
MATRICES = ("cora_full", "pubmed", "coauthor_cs", "github", "flickr",
            "ogbn_arxiv")
GPU_NAMES = (
    "GPU_Unfused_SeqReduceRowBalance",
    "GPU_Unfused_CuSparse_ALG3",
    "GPU_CSRCSC_Fused_Atomic",
)
RUN_FIELDS = (
    "matrix", "repeat", "rows", "nnz", "features", "threads",
    "atomic_rows_per_block", "implementation", "steady_ms", "all_trials_ms",
    "analysis_ms", "correct_trials", "trials", "max_abs_error",
)
SUMMARY_FIELDS = (
    "matrix", "rows", "nnz", "features", "threads", "repeats",
    "atomic_rows_per_block", "implementation", "steady_ms", "all_trials_ms",
    "analysis_ms", "speedup_vs_row_balance", "speedup_vs_cusparse_alg3",
    "correct_trials", "trials", "max_abs_error",
)


def tuned_rows_per_block():
    with TUNING_FILE.open(newline="") as stream:
        return {row["matrix"]: int(row["rows_per_block"])
                for row in csv.DictReader(stream)
                if row["cohort"] == "end2end" and
                row["ordering"] == "original"}


def run_one(binary, matrix, rows_per_block, threads, features, timeout):
    env = os.environ.copy()
    for key in ("SPMM_SPMM_HYBRID_ONLY", "SPMM_SPMM_HYBRID_128",
                "SPMM_SPMM_HYBRID_COOPERATIVE", "SPMM_SPMM_SKIP_2LN"):
        env.pop(key, None)
    env["SPMM_SPMM_ATOMIC_ONLY"] = "1"
    env["SPMM_SPMM_CSRCSC_ROWS_PER_BLOCK"] = str(rows_per_block)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    result = subprocess.run(
        [str(binary), "-sm", str(matrix), "-nt", str(threads),
         "-bc", str(features), "-ah"], env=env, text=True,
        capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{matrix} failed (exit {result.returncode}): {result.stderr[-700:]}")
    parsed = list(csv.DictReader(result.stdout.splitlines()))
    names = {row["Implementation Name"] for row in parsed}
    if len(parsed) != 4 or names != set(GPU_NAMES) | {"CPU_Unfused_Seq"}:
        raise RuntimeError(f"Unexpected benchmark implementations: {names}")
    return parsed


def parse_row(row, matrix, repeat, rows_per_block, features, threads):
    trials = int(row["Number of Trials"])
    times = [1000.0 * float(row[f"Trial{i} Subregion0 Executor"])
             for i in range(trials)]
    return {
        "matrix": matrix,
        "repeat": repeat,
        "rows": int(float(row["nRows"])),
        "nnz": int(float(row["NNZ"])),
        "features": features,
        "threads": threads,
        "atomic_rows_per_block": rows_per_block,
        "implementation": row["Implementation Name"],
        "steady_ms": statistics.mean(times[1:]),
        "all_trials_ms": statistics.mean(times),
        "analysis_ms": 1000.0 * float(row["Trial0 Subregion0 Analysis"]),
        "correct_trials": sum(row[f"Correct{i}"] == "1"
                              for i in range(trials)),
        "trials": trials,
        "max_abs_error": max(float(row[f"Error{i}"])
                             for i in range(trials)),
    }


def summarize(runs, output):
    groups = defaultdict(list)
    for row in runs:
        groups[(row["matrix"], row["implementation"])].append(row)
    averages = {key: statistics.mean(float(r["steady_ms"]) for r in group)
                for key, group in groups.items()}
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for (matrix, implementation), group in sorted(groups.items()):
            row = group[0]
            mean = lambda field: statistics.mean(float(r[field])
                                                 for r in group)
            steady = averages[(matrix, implementation)]
            writer.writerow({
                "matrix": matrix,
                "rows": row["rows"],
                "nnz": row["nnz"],
                "features": row["features"],
                "threads": row["threads"],
                "repeats": len(group),
                "atomic_rows_per_block": row["atomic_rows_per_block"],
                "implementation": implementation,
                "steady_ms": f"{steady:.6f}",
                "all_trials_ms": f"{mean('all_trials_ms'):.6f}",
                "analysis_ms": f"{mean('analysis_ms'):.6f}",
                "speedup_vs_row_balance":
                    f"{averages[(matrix, GPU_NAMES[0])] / steady:.6f}",
                "speedup_vs_cusparse_alg3":
                    f"{averages[(matrix, GPU_NAMES[1])] / steady:.6f}",
                "correct_trials": sum(int(r["correct_trials"]) for r in group),
                "trials": sum(int(r["trials"]) for r in group),
                "max_abs_error": max(float(r["max_abs_error"])
                                     for r in group),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path, help="summary CSV")
    parser.add_argument("--matrices", nargs="+", default=MATRICES)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or args.features < 1:
        parser.error("repeats, threads, and features must be positive")
    binary = args.binary.resolve(strict=True)
    selected = tuned_rows_per_block()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_path = args.output.with_name(args.output.stem + "_runs.csv")
    runs = []
    with run_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for repeat in range(1, args.repeats + 1):
            names = args.matrices if repeat % 2 else list(reversed(args.matrices))
            for name in names:
                matrix = DATA_ROOT / name / (name + ".mtx")
                if not matrix.is_file():
                    raise FileNotFoundError(matrix)
                rows_per_block = selected[name]
                print(f"{name}: repeat {repeat}/{args.repeats}, "
                      f"atomic rows/block={rows_per_block}", flush=True)
                parsed = run_one(binary, matrix, rows_per_block,
                                 args.threads, args.features, args.timeout)
                for row in parsed:
                    if row["Implementation Name"] == "CPU_Unfused_Seq":
                        continue
                    item = parse_row(row, name, repeat, rows_per_block,
                                     args.features, args.threads)
                    runs.append(item)
                    writer.writerow(item)
                stream.flush()
    summarize(runs, args.output)
    print(f"Saved {args.output} and {run_path}", flush=True)


if __name__ == "__main__":
    main()

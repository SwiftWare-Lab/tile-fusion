#!/usr/bin/env python3
"""Compare unfused and fused SpMM-SpMM on original and ordered GNN graphs.

Trial 0 is treated as GPU warm-up and omitted from steady-state means. Runs
alternate original/ordered order to reduce temperature and clock bias.
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
DEFAULT_MATRICES = (
    "cora_full", "pubmed", "coauthor_cs", "github", "flickr",
    "ogbn_arxiv",
)
IMPLEMENTATIONS = (
    "GPU_Unfused_SeqReduceRowBalance",
    "GPU_Unfused_CuSparse_ALG3",
    "GPU_CSRCSC_Fused_Atomic",
    "GPU_CSRCSC_Fused_ConflictReduce",
)
RUN_FIELDS = (
    "matrix", "ordering", "repeat", "rows", "nnz", "features", "threads",
    "atomic_rows_per_block", "conflict_rows_per_block",
    "conflict_min_savings_pct", "implementation", "steady_ms",
    "all_trials_ms", "analysis_ms", "grouped_rows", "eliminated_edges",
    "correct_trials", "trials", "max_abs_error",
)
SUMMARY_FIELDS = (
    "matrix", "ordering", "rows", "nnz", "features", "threads", "repeats",
    "atomic_rows_per_block", "conflict_rows_per_block",
    "conflict_min_savings_pct", "implementation", "steady_ms",
    "analysis_ms", "speedup_vs_rowwise", "speedup_vs_cusparse",
    "speedup_vs_atomic", "grouped_rows", "eliminated_edges",
    "correct_trials", "trials", "max_abs_error",
)


def matrix_paths(name):
    original = DATA_ROOT / name / f"{name}.mtx"
    ordered = DATA_ROOT / f"{name}_ordered" / f"{name}_ordered.mtx"
    if not original.is_file() or not ordered.is_file():
        raise FileNotFoundError(f"missing matrix pair: {original}, {ordered}")
    return {"original": original, "ordered": ordered}


def run_one(args, matrix):
    env = os.environ.copy()
    for name in (
        "SPMM_SPMM_HYBRID_ONLY", "SPMM_SPMM_HYBRID_128",
        "SPMM_SPMM_HYBRID_COOPERATIVE", "SPMM_SPMM_SKIP_2LN",
    ):
        env.pop(name, None)
    env["SPMM_SPMM_ATOMIC_ONLY"] = "1"
    env["SPMM_SPMM_CSRCSC_ROWS_PER_BLOCK"] = str(args.atomic_rows)
    env["SPMM_SPMM_CONFLICT_ROWS_PER_BLOCK"] = str(args.conflict_rows)
    env["SPMM_SPMM_CONFLICT_MIN_SAVINGS_PCT"] = str(args.min_savings)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    result = subprocess.run(
        [str(args.binary), "-sm", str(matrix), "-nt", str(args.threads),
         "-bc", str(args.features), "-ah"],
        env=env, text=True, capture_output=True, timeout=args.timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"{matrix} failed (exit {result.returncode}):\n"
            f"{result.stderr[-4000:]}"
        )
    rows = list(csv.DictReader(result.stdout.splitlines()))
    names = {row["Implementation Name"] for row in rows}
    required = set(IMPLEMENTATIONS) | {"CPU_Unfused_Seq"}
    if names != required:
        raise RuntimeError(f"unexpected implementations for {matrix}: {names}")
    return rows


def parse_run(args, row, matrix, ordering, repeat):
    trials = int(row["Number of Trials"])
    times = [1000.0 * float(row[f"Trial{i} Subregion0 Executor"])
             for i in range(trials)]
    return {
        "matrix": matrix,
        "ordering": ordering,
        "repeat": repeat,
        "rows": int(float(row["nRows"])),
        "nnz": int(float(row["NNZ"])),
        "features": args.features,
        "threads": args.threads,
        "atomic_rows_per_block": args.atomic_rows,
        "conflict_rows_per_block": args.conflict_rows,
        "conflict_min_savings_pct": args.min_savings,
        "implementation": row["Implementation Name"],
        "steady_ms": statistics.mean(times[1:]),
        "all_trials_ms": statistics.mean(times),
        "analysis_ms": 1000.0 * float(row["Trial0 Subregion0 Analysis"]),
        "grouped_rows": int(float(row["Number of Fused Rows0"])),
        "eliminated_edges": int(float(row["Number of Fused Nnz0"])),
        "correct_trials": sum(row[f"Correct{i}"] == "1"
                              for i in range(trials)),
        "trials": trials,
        "max_abs_error": max(float(row[f"Error{i}"]) for i in range(trials)),
    }


def write_summary(runs, path):
    groups = defaultdict(list)
    for row in runs:
        groups[(row["matrix"], row["ordering"],
                row["implementation"])].append(row)
    mean_ms = {
        key: statistics.mean(float(row["steady_ms"]) for row in group)
        for key, group in groups.items()
    }
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for key, group in sorted(groups.items()):
            matrix, ordering, implementation = key
            first = group[0]
            steady = mean_ms[key]
            baseline = lambda name: mean_ms[(matrix, ordering, name)]
            writer.writerow({
                "matrix": matrix,
                "ordering": ordering,
                "rows": first["rows"],
                "nnz": first["nnz"],
                "features": first["features"],
                "threads": first["threads"],
                "repeats": len(group),
                "atomic_rows_per_block": first["atomic_rows_per_block"],
                "conflict_rows_per_block": first["conflict_rows_per_block"],
                "conflict_min_savings_pct": first["conflict_min_savings_pct"],
                "implementation": implementation,
                "steady_ms": f"{steady:.6f}",
                "analysis_ms": f"{statistics.mean(float(row['analysis_ms']) for row in group):.6f}",
                "speedup_vs_rowwise": f"{baseline(IMPLEMENTATIONS[0]) / steady:.6f}",
                "speedup_vs_cusparse": f"{baseline(IMPLEMENTATIONS[1]) / steady:.6f}",
                "speedup_vs_atomic": f"{baseline(IMPLEMENTATIONS[2]) / steady:.6f}",
                "grouped_rows": round(statistics.mean(
                    int(row["grouped_rows"]) for row in group)),
                "eliminated_edges": round(statistics.mean(
                    int(row["eliminated_edges"]) for row in group)),
                "correct_trials": sum(int(row["correct_trials"])
                                      for row in group),
                "trials": sum(int(row["trials"]) for row in group),
                "max_abs_error": max(float(row["max_abs_error"])
                                     for row in group),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--matrices", nargs="+", default=DEFAULT_MATRICES)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atomic-rows", type=int, default=4)
    parser.add_argument("--conflict-rows", type=int, default=4)
    parser.add_argument("--min-savings", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runs_path = args.output.with_name(args.output.stem + "_runs.csv")
    runs = []
    with runs_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for matrix in args.matrices:
            paths = matrix_paths(matrix)
            for repeat in range(1, args.repeats + 1):
                orderings = ("original", "ordered") if repeat % 2 else \
                    ("ordered", "original")
                for ordering in orderings:
                    print(f"{matrix}: repeat {repeat}/{args.repeats}, "
                          f"{ordering}", flush=True)
                    for row in run_one(args, paths[ordering]):
                        if row["Implementation Name"] == "CPU_Unfused_Seq":
                            continue
                        parsed = parse_run(args, row, matrix, ordering, repeat)
                        runs.append(parsed)
                        writer.writerow(parsed)
                    stream.flush()
    write_summary(runs, args.output)
    print(f"Saved {args.output} and {runs_path}", flush=True)


if __name__ == "__main__":
    main()

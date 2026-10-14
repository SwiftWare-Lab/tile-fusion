#!/usr/bin/env python3
"""Compare GPU CSR-CSC atomic SpMM-SpMM on original and ordered matrices.

The demo must be rebuilt first. Runs alternate original/ordered order across
repetitions to reduce clock- and temperature-related ordering bias.
"""

import argparse
import csv
import os
import statistics
import subprocess
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1] / "data/end2end-test-data"
DEFAULT_MATRICES = (
    "cora_full", "pubmed", "coauthor_cs", "coauthor_physics",
    "deezer_europe", "facebook_page_page", "github", "flickr",
    "ogbn_arxiv",
)
RUN_FIELDS = (
    "matrix", "ordering", "repeat", "rows", "nnz", "implementation",
    "mean_ms", "analysis_ms", "correct_trials", "trials", "max_abs_error",
)
SUMMARY_FIELDS = (
    "matrix", "rows", "nnz", "repeats", "original_atomic_ms",
    "ordered_atomic_ms", "ordered_over_original", "ordering_change_pct",
    "original_unfused_ms", "ordered_unfused_ms",
    "original_atomic_analysis_ms", "ordered_atomic_analysis_ms",
    "original_atomic_correct", "ordered_atomic_correct",
)


def matrix_paths(name):
    original = DATA_ROOT / name / (name + ".mtx")
    ordered = DATA_ROOT / (name + "_ordered") / (name + "_ordered.mtx")
    if not original.is_file() or not ordered.is_file():
        raise FileNotFoundError(f"missing matrix pair: {original}, {ordered}")
    return {"original": original, "ordered": ordered}


def run_one(binary, matrix, threads, features):
    env = os.environ.copy()
    for name in ("SPMM_SPMM_HYBRID_ONLY", "SPMM_SPMM_HYBRID_128",
                 "SPMM_SPMM_HYBRID_COOPERATIVE"):
        env.pop(name, None)
    env["SPMM_SPMM_ATOMIC_ONLY"] = "1"
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    command = [str(binary), "-sm", str(matrix), "-nt", str(threads),
               "-bc", str(features), "-ah"]
    result = subprocess.run(command, env=env, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"benchmark failed for {matrix} (exit {result.returncode}):\n"
            + result.stderr[-4000:]
        )
    rows = list(csv.DictReader(result.stdout.splitlines()))
    names = {row["Implementation Name"] for row in rows}
    required = {"GPU_Unfused_SeqReduceRowBalance", "GPU_CSRCSC_Fused_Atomic"}
    if not required <= names:
        raise RuntimeError(f"missing expected benchmark rows for {matrix}: {names}")
    return rows


def parse_run(row, name, ordering, repeat):
    trials = int(row["Number of Trials"])
    return {
        "matrix": name,
        "ordering": ordering,
        "repeat": repeat,
        "rows": int(float(row["nRows"])),
        "nnz": int(float(row["NNZ"])),
        "implementation": row["Implementation Name"],
        "mean_ms": statistics.mean(
            1000.0 * float(row[f"Trial{i} Subregion0 Executor"])
            for i in range(trials)
        ),
        "analysis_ms": 1000.0 * float(row["Trial0 Subregion0 Analysis"]),
        "correct_trials": sum(row[f"Correct{i}"] == "1"
                              for i in range(trials)),
        "trials": trials,
        "max_abs_error": max(float(row[f"Error{i}"])
                             for i in range(trials)),
    }


def make_summary(name, runs, repeats):
    def selected(ordering, implementation):
        return [r for r in runs if r["ordering"] == ordering and
                r["implementation"] == implementation]

    original = selected("original", "GPU_CSRCSC_Fused_Atomic")
    ordered = selected("ordered", "GPU_CSRCSC_Fused_Atomic")
    if len(original) != repeats or len(ordered) != repeats:
        raise RuntimeError(f"incomplete atomic results for {name}")
    original_ms = statistics.mean(r["mean_ms"] for r in original)
    ordered_ms = statistics.mean(r["mean_ms"] for r in ordered)
    ratio = ordered_ms / original_ms
    return {
        "matrix": name,
        "rows": original[0]["rows"],
        "nnz": original[0]["nnz"],
        "repeats": repeats,
        "original_atomic_ms": f"{original_ms:.6f}",
        "ordered_atomic_ms": f"{ordered_ms:.6f}",
        "ordered_over_original": f"{ratio:.6f}",
        "ordering_change_pct": f"{100.0 * (ratio - 1.0):.3f}",
        "original_unfused_ms": f"{statistics.mean(r['mean_ms'] for r in selected('original', 'GPU_Unfused_SeqReduceRowBalance')):.6f}",
        "ordered_unfused_ms": f"{statistics.mean(r['mean_ms'] for r in selected('ordered', 'GPU_Unfused_SeqReduceRowBalance')):.6f}",
        "original_atomic_analysis_ms": f"{statistics.mean(r['analysis_ms'] for r in original):.6f}",
        "ordered_atomic_analysis_ms": f"{statistics.mean(r['analysis_ms'] for r in ordered):.6f}",
        "original_atomic_correct": sum(r["correct_trials"] for r in original),
        "ordered_atomic_correct": sum(r["correct_trials"] for r in ordered),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="rebuilt spmm_spmm_demo_gpu")
    parser.add_argument("output", type=Path, help="pair summary CSV")
    parser.add_argument("--matrices", nargs="+", default=DEFAULT_MATRICES)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    binary = args.binary.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runs_path = args.output.with_name(args.output.stem + "_runs.csv")
    with args.output.open("w", newline="") as summary_file, \
            runs_path.open("w", newline="") as runs_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        runs_writer = csv.DictWriter(runs_file, fieldnames=RUN_FIELDS)
        summary_writer.writeheader()
        runs_writer.writeheader()
        for name in args.matrices:
            paths = matrix_paths(name)
            matrix_runs = []
            for repeat in range(args.repeats):
                order = ("original", "ordered") if repeat % 2 == 0 else \
                    ("ordered", "original")
                for ordering in order:
                    print(f"{name}: repeat {repeat + 1}/{args.repeats}, "
                          f"{ordering}", flush=True)
                    rows = run_one(binary, paths[ordering], args.threads,
                                   args.features)
                    for row in rows:
                        if row["Implementation Name"] == "CPU_Unfused_Seq":
                            continue
                        parsed = parse_run(row, name, ordering, repeat + 1)
                        matrix_runs.append(parsed)
                        runs_writer.writerow(parsed)
                    runs_file.flush()
            summary = make_summary(name, matrix_runs, args.repeats)
            summary_writer.writerow(summary)
            summary_file.flush()
            print(f"  atomic original {summary['original_atomic_ms']} ms, "
                  f"ordered {summary['ordered_atomic_ms']} ms "
                  f"({summary['ordering_change_pct']}%)", flush=True)
    print(f"Saved {args.output} and {runs_path}", flush=True)


if __name__ == "__main__":
    main()

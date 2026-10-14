#!/usr/bin/env python3
"""Run every default GPU SpMM-SpMM benchmark variant on matrix pairs.

The demo must be rebuilt first. This script deliberately unsets the opt-in
atomic-only and hybrid modes, so colored variants are not benchmarked. SPD
matrices use cached RCM files alongside the originals.
"""

import argparse
import csv
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
END2END_MATRICES = (
    "cora_full", "pubmed", "coauthor_cs", "coauthor_physics",
    "deezer_europe", "facebook_page_page", "github", "flickr",
    "ogbn_arxiv",
)
SPD_MATRICES = ("pwtk", "pdb1HYS", "thermal1", "ship_003")
FIELDS = (
    "matrix", "ordering", "repeat", "rows", "nnz", "implementation",
    "steady_ms", "all_trials_ms", "analysis_ms", "correct_trials", "trials",
    "max_error",
)
SUMMARY_FIELDS = (
    "matrix", "ordering", "rows", "nnz", "implementation", "repeats",
    "steady_ms", "all_trials_ms", "analysis_ms", "speedup_vs_row_balance",
    "correct_trials", "trials", "max_error",
)


def matrix_path(name, ordering, dataset):
    if dataset == "spd":
        path = DATA_ROOT / "SPD" / name / (name + ".mtx")
        if not path.is_file():
            raise FileNotFoundError(path)
        if ordering == "ordered":
            from run_gpu_rcm_exp import prepare_rcm
            return prepare_rcm(path)
        return path
    folder = name if ordering == "original" else name + "_ordered"
    path = DATA_ROOT / "end2end-test-data" / folder / (folder + ".mtx")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_row(row, name, ordering, repeat):
    trials = int(row["Number of Trials"])
    times = [1000 * float(row[f"Trial{i} Subregion0 Executor"])
             for i in range(trials)]
    return {
        "matrix": name,
        "ordering": ordering,
        "repeat": repeat,
        "rows": int(float(row["nRows"])),
        "nnz": int(float(row["NNZ"])),
        "implementation": row["Implementation Name"],
        "steady_ms": statistics.mean(times[1:]) if trials > 1 else times[0],
        "all_trials_ms": statistics.mean(times),
        "analysis_ms": 1000 * float(row["Trial0 Subregion0 Analysis"]),
        "correct_trials": sum(row[f"Correct{i}"] == "1"
                              for i in range(trials)),
        "trials": trials,
        "max_error": max(float(row[f"Error{i}"]) for i in range(trials)),
    }


def write_summary(runs_path):
    with runs_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["matrix"], row["ordering"],
                row["implementation"])].append(row)
    baseline = {}
    for (name, ordering, impl), group in groups.items():
        if impl == "GPU_Unfused_SeqReduceRowBalance":
            baseline[(name, ordering)] = statistics.mean(
                float(row["steady_ms"]) for row in group)
    summary_path = runs_path.with_name(
        runs_path.stem.removesuffix("_runs") + "_summary.csv")
    with summary_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for (name, ordering, impl), group in sorted(groups.items()):
            mean = lambda field: statistics.mean(float(row[field])
                                                 for row in group)
            steady = mean("steady_ms")
            writer.writerow({
                "matrix": name,
                "ordering": ordering,
                "rows": group[0]["rows"],
                "nnz": group[0]["nnz"],
                "implementation": impl,
                "repeats": len(group),
                "steady_ms": f"{steady:.6f}",
                "all_trials_ms": f"{mean('all_trials_ms'):.6f}",
                "analysis_ms": f"{mean('analysis_ms'):.6f}",
                "speedup_vs_row_balance":
                    f"{baseline[(name, ordering)] / steady:.6f}",
                "correct_trials": sum(int(row["correct_trials"])
                                      for row in group),
                "trials": sum(int(row["trials"]) for row in group),
                "max_error": max(float(row["max_error"]) for row in group),
            })
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path, help="per-run CSV output")
    parser.add_argument("--dataset", choices=("end2end", "spd"),
                        default="end2end")
    parser.add_argument("--matrices", nargs="+", help="matrix folder names")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.matrices is None:
        args.matrices = SPD_MATRICES if args.dataset == "spd" else \
            END2END_MATRICES
    binary = args.binary.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failure_path = args.output.with_name(args.output.stem + "_failures.csv")
    env = os.environ.copy()
    for key in ("SPMM_SPMM_ATOMIC_ONLY", "SPMM_SPMM_HYBRID_ONLY",
                "SPMM_SPMM_HYBRID_128", "SPMM_SPMM_HYBRID_COOPERATIVE",
                "SPMM_SPMM_SKIP_2LN"):
        env.pop(key, None)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)

    with args.output.open("w", newline="") as output, \
            failure_path.open("w", newline="") as failures:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        error_writer = csv.DictWriter(
            failures, fieldnames=("matrix", "ordering", "repeat", "error"))
        writer.writeheader()
        error_writer.writeheader()
        for name in args.matrices:
            paths = {ordering: matrix_path(name, ordering, args.dataset)
                     for ordering in ("original", "ordered")}
            for repeat in range(1, args.repeats + 1):
                order = ("original", "ordered") if repeat % 2 else \
                    ("ordered", "original")
                for ordering in order:
                    matrix = paths[ordering]
                    print(f"{name} {ordering} {repeat}/{args.repeats}", flush=True)
                    command = [str(binary), "-sm", str(matrix), "-nt",
                               str(args.threads), "-bc", str(args.features),
                               "-ah"]
                    try:
                        result = subprocess.run(
                            command, env=env, text=True, capture_output=True,
                            timeout=args.timeout, check=False)
                        expected_rows = 25
                        if result.returncode:
                            error = (f"full benchmark exit {result.returncode}; "
                                     f"retrying without 2LN: {result.stderr[-500:]}")
                            error_writer.writerow({"matrix": name,
                                                   "ordering": ordering,
                                                   "repeat": repeat,
                                                   "error": error})
                            failures.flush()
                            print(f"  {error}", flush=True)
                            retry_env = env.copy()
                            retry_env["SPMM_SPMM_SKIP_2LN"] = "1"
                            result = subprocess.run(
                                command, env=retry_env, text=True,
                                capture_output=True, timeout=args.timeout,
                                check=False)
                            expected_rows = 21
                            if result.returncode:
                                raise RuntimeError(
                                    f"retry exit {result.returncode}: "
                                    f"{result.stderr[-500:]}")
                        rows = list(csv.DictReader(result.stdout.splitlines()))
                        if len(rows) != expected_rows:
                            raise RuntimeError(
                                f"expected {expected_rows} rows, got {len(rows)}")
                        if any("Colored" in r["Implementation Name"] for r in rows):
                            raise RuntimeError("colored variant found in default run")
                        for row in rows:
                            writer.writerow(parse_row(row, name, ordering, repeat))
                        output.flush()
                    except (subprocess.TimeoutExpired, RuntimeError) as exc:
                        error = str(exc).replace("\n", " ")
                        error_writer.writerow({"matrix": name,
                                               "ordering": ordering,
                                               "repeat": repeat,
                                               "error": error})
                        failures.flush()
                        print(f"  FAILED: {error}", flush=True)
    summary_path = write_summary(args.output)
    print(f"Saved {args.output}, {summary_path}, and {failure_path}",
          flush=True)


if __name__ == "__main__":
    main()

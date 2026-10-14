#!/usr/bin/env python3
"""Tune only GPU CSR-CSC atomic rows per block on saved benchmark suites.

Runs the benchmark's atomic-only mode at 32 features. Every matrix/order case
is evaluated at 1, 2, 4, and 8 producer rows per CUDA block, with repeated
seven-trial measurements. The reported speedup uses the existing fixed
row-balance baseline from the full benchmark results.
"""

import argparse
import csv
import math
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


FUSION_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = FUSION_ROOT / "data"
RESULTS_ROOT = FUSION_ROOT / "results"
END2END = (
    "cora_full", "pubmed", "coauthor_cs", "coauthor_physics",
    "deezer_europe", "facebook_page_page", "github", "flickr",
    "ogbn_arxiv",
)
SPD = ("pwtk", "pdb1HYS", "thermal1", "ship_003")
TILES = (1, 2, 4, 8)
RUN_FIELDS = (
    "cohort", "matrix", "ordering", "rows_per_block", "repeat",
    "steady_ms", "all_trials_ms", "analysis_ms", "correct_trials",
    "trials", "max_error",
)
SUMMARY_FIELDS = (
    "cohort", "matrix", "ordering", "rows_per_block", "repeats",
    "steady_ms", "baseline_ms", "speedup_vs_row_balance",
    "correct_trials", "trials", "max_error", "fully_correct",
)


def matrix_path(cohort, name, ordering):
    if cohort == "end2end":
        folder = name if ordering == "original" else name + "_ordered"
        path = DATA_ROOT / "end2end-test-data" / folder / (folder + ".mtx")
    else:
        suffix = "" if ordering == "original" else "_rcm"
        path = DATA_ROOT / "SPD" / name / (name + suffix + ".mtx")
    return path.resolve(strict=True)


def baselines(cohort):
    path = RESULTS_ROOT / f"gpu_all_impl_{cohort}_16t_32f_summary.csv"
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        return {(row["matrix"], row["ordering"]): float(row["steady_ms"])
                for row in rows if row["implementation"] ==
                "GPU_Unfused_SeqReduceRowBalance"}


def run_one(binary, path, threads, rows_per_block, timeout):
    env = os.environ.copy()
    for key in ("SPMM_SPMM_HYBRID_ONLY", "SPMM_SPMM_HYBRID_128",
                "SPMM_SPMM_HYBRID_COOPERATIVE"):
        env.pop(key, None)
    env["SPMM_SPMM_ATOMIC_ONLY"] = "1"
    env["SPMM_SPMM_CSRCSC_ROWS_PER_BLOCK"] = str(rows_per_block)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    result = subprocess.run(
        [str(binary), "-sm", str(path), "-nt", str(threads),
         "-bc", "32", "-ah"], env=env, text=True, capture_output=True,
        timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{path} rows/block={rows_per_block}: exit {result.returncode}; "
            f"{result.stderr[-600:]}")
    rows = list(csv.DictReader(result.stdout.splitlines()))
    atomic = [r for r in rows if r["Implementation Name"] ==
              "GPU_CSRCSC_Fused_Atomic"]
    if len(rows) != 3 or len(atomic) != 1:
        raise RuntimeError(f"Unexpected output from {path}: {len(rows)} rows")
    row = atomic[0]
    trials = int(row["Number of Trials"])
    times = [1000 * float(row[f"Trial{i} Subregion0 Executor"])
             for i in range(trials)]
    return {
        "steady_ms": statistics.mean(times[1:]),
        "all_trials_ms": statistics.mean(times),
        "analysis_ms": 1000 * float(row["Trial0 Subregion0 Analysis"]),
        "correct_trials": sum(row[f"Correct{i}"] == "1"
                              for i in range(trials)),
        "trials": trials,
        "max_error": max(float(row[f"Error{i}"]) for i in range(trials)),
    }


def summarize(runs, summary_path):
    by_config = defaultdict(list)
    for run in runs:
        by_config[(run["cohort"], run["matrix"], run["ordering"],
                   run["rows_per_block"])].append(run)
    baseline = {cohort: baselines(cohort) for cohort in ("end2end", "spd")}
    configs = []
    for (cohort, matrix, ordering, rows_per_block), group in sorted(
            by_config.items()):
        steady = statistics.mean(float(r["steady_ms"]) for r in group)
        correct = sum(int(r["correct_trials"]) for r in group)
        trials = sum(int(r["trials"]) for r in group)
        base = baseline[cohort][(matrix, ordering)]
        configs.append({
            "cohort": cohort,
            "matrix": matrix,
            "ordering": ordering,
            "rows_per_block": rows_per_block,
            "repeats": len(group),
            "steady_ms": steady,
            "baseline_ms": base,
            "speedup_vs_row_balance": base / steady,
            "correct_trials": correct,
            "trials": trials,
            "max_error": max(float(r["max_error"]) for r in group),
            "fully_correct": int(correct == trials),
        })
    config_path = summary_path.with_name(summary_path.stem + "_all_configs.csv")
    with config_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(configs)
    cases = defaultdict(list)
    for item in configs:
        cases[(item["cohort"], item["matrix"], item["ordering"])].append(item)
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for _, choices in sorted(cases.items()):
            valid = [item for item in choices if item["fully_correct"]]
            writer.writerow(min(valid or choices,
                                key=lambda item: item["steady_ms"]))
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path, help="selected-config CSV")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cohort", choices=("all", "end2end", "spd"),
                        default="all")
    parser.add_argument("--matrices", nargs="+")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    binary = args.binary.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_path = args.output.with_name(args.output.stem + "_runs.csv")
    cohorts = ("end2end", "spd") if args.cohort == "all" else (args.cohort,)
    runs = []
    with run_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for cohort in cohorts:
            names = args.matrices or (END2END if cohort == "end2end" else SPD)
            for name in names:
                for ordering in ("original", "ordered"):
                    path = matrix_path(cohort, name, ordering)
                    for repeat in range(1, args.repeats + 1):
                        configs = TILES if repeat % 2 else tuple(reversed(TILES))
                        for rows_per_block in configs:
                            print(f"{cohort} {name} {ordering} "
                                  f"rows/block={rows_per_block} "
                                  f"repeat={repeat}/{args.repeats}", flush=True)
                            run = run_one(binary, path, args.threads,
                                          rows_per_block, args.timeout)
                            run.update({"cohort": cohort, "matrix": name,
                                        "ordering": ordering,
                                        "rows_per_block": rows_per_block,
                                        "repeat": repeat})
                            runs.append(run)
                            writer.writerow(run)
                            stream.flush()
    config_path = summarize(runs, args.output)
    print(f"Saved {args.output}, {config_path}, and {run_path}")


if __name__ == "__main__":
    main()

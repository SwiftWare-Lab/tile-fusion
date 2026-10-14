#!/usr/bin/env python3
"""Select the best conflict-reduced ordering for each feature-sweep case."""

import argparse
import csv
from pathlib import Path


ROWWISE = "GPU_Unfused_SeqReduceRowBalance"
CUSPARSE = "GPU_Unfused_CuSparse_ALG3"
CONFLICT = "GPU_CSRCSC_Fused_ConflictReduce"
FIELDS = (
    "matrix", "features", "rows", "nnz", "best_conflict_ordering",
    "unordered_rowwise_ms", "unordered_cusparse_ms", "best_conflict_ms",
    "conflict_over_rowwise", "conflict_over_cusparse",
    "speedup_vs_rowwise", "speedup_vs_cusparse", "conflict_analysis_ms",
    "grouped_rows", "eliminated_edges", "correct_trials", "trials",
    "max_abs_error",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    selected = []
    for source in args.inputs:
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_key = {
            (row["matrix"], row["ordering"], row["implementation"]): row
            for row in rows
        }
        matrices = sorted({row["matrix"] for row in rows})
        for matrix in matrices:
            rowwise = by_key[(matrix, "original", ROWWISE)]
            cusparse = by_key[(matrix, "original", CUSPARSE)]
            candidates = [
                by_key[(matrix, "original", CONFLICT)],
                by_key[(matrix, "ordered", CONFLICT)],
            ]
            conflict = min(candidates, key=lambda row: float(row["steady_ms"]))
            rowwise_ms = float(rowwise["steady_ms"])
            cusparse_ms = float(cusparse["steady_ms"])
            conflict_ms = float(conflict["steady_ms"])
            selected.append({
                "matrix": matrix,
                "features": int(conflict["features"]),
                "rows": int(conflict["rows"]),
                "nnz": int(conflict["nnz"]),
                "best_conflict_ordering": conflict["ordering"],
                "unordered_rowwise_ms": f"{rowwise_ms:.6f}",
                "unordered_cusparse_ms": f"{cusparse_ms:.6f}",
                "best_conflict_ms": f"{conflict_ms:.6f}",
                "conflict_over_rowwise": f"{conflict_ms / rowwise_ms:.6f}",
                "conflict_over_cusparse": f"{conflict_ms / cusparse_ms:.6f}",
                "speedup_vs_rowwise": f"{rowwise_ms / conflict_ms:.6f}",
                "speedup_vs_cusparse": f"{cusparse_ms / conflict_ms:.6f}",
                "conflict_analysis_ms": conflict["analysis_ms"],
                "grouped_rows": conflict["grouped_rows"],
                "eliminated_edges": conflict["eliminated_edges"],
                "correct_trials": conflict["correct_trials"],
                "trials": conflict["trials"],
                "max_abs_error": conflict["max_abs_error"],
            })
    selected.sort(key=lambda row: (int(row["features"]), row["matrix"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

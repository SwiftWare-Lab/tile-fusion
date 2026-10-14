from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class DatasetSpec:
    family: str
    name: str


DATASETS = {
    "cora": DatasetSpec("Planetoid", "Cora"),
    "citeseer": DatasetSpec("Planetoid", "CiteSeer"),
    "pubmed": DatasetSpec("Planetoid", "PubMed"),
    "coauthor-cs": DatasetSpec("Coauthor", "CS"),
    "coauthor-physics": DatasetSpec("Coauthor", "Physics"),
    "amazon-computers": DatasetSpec("Amazon", "Computers"),
    "amazon-photo": DatasetSpec("Amazon", "Photo"),
    "flickr": DatasetSpec("Flickr", "Flickr"),
    "reddit": DatasetSpec("Reddit", "Reddit"),
}


def load_pyg_dataset(dataset_key: str, root: Path):
    from torch_geometric.datasets import Amazon, Coauthor, Flickr, Planetoid, Reddit

    key = dataset_key.lower()
    if key not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset '{dataset_key}'. Known datasets: {known}")

    spec = DATASETS[key]
    dataset_root = root / key
    if spec.family == "Planetoid":
        dataset = Planetoid(root=str(dataset_root), name=spec.name)
    elif spec.family == "Coauthor":
        dataset = Coauthor(root=str(dataset_root), name=spec.name)
    elif spec.family == "Amazon":
        dataset = Amazon(root=str(dataset_root), name=spec.name)
    elif spec.family == "Flickr":
        dataset = Flickr(root=str(dataset_root))
    elif spec.family == "Reddit":
        dataset = Reddit(root=str(dataset_root))
    else:
        raise ValueError(f"Unsupported dataset family: {spec.family}")

    return dataset[0]


def pyg_to_csr(data):
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.cpu()
    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    coo = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
    csr = coo.coalesce().to_sparse_csr()

    indptr = csr.crow_indices().to(torch.int32).cpu().numpy()
    indices = csr.col_indices().to(torch.int32).cpu().numpy()
    values = csr.values().to(torch.float32).cpu().numpy()
    return csr, values, indices, indptr


def make_features(data, in_dim: int | None, seed: int) -> np.ndarray:
    if data.x is not None and in_dim is None:
        return data.x.to(torch.float32).cpu().contiguous().numpy()

    num_nodes = int(data.num_nodes)
    feature_dim = in_dim if in_dim is not None else 32
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_nodes, feature_dim)).astype(np.float32)


def median_time_ms(fn, warmup: int, repeats: int) -> tuple[float, object]:
    result = None
    for _ in range(warmup):
        result = fn()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1e3)

    return float(np.median(samples)), result


def torch_unfused_baseline(adj_csr: torch.Tensor, feature: np.ndarray, weight_t: np.ndarray):
    feature_torch = torch.from_numpy(feature)
    weight_torch = torch.from_numpy(weight_t)
    gemm_out = torch.mm(feature_torch, weight_torch.T)
    return torch.sparse.mm(adj_csr, gemm_out)


def run_dataset(dataset_key: str, data, args, writer: csv.DictWriter | None) -> None:
    import tf_lib

    torch.set_num_threads(args.threads)

    adj_csr_torch, values, indices, indptr = pyg_to_csr(data)
    feature = make_features(data, args.in_dim, args.seed)
    out_dim = args.out_dim

    rng = np.random.default_rng(args.seed + 1)
    weight_t = rng.normal(size=(out_dim, feature.shape[1])).astype(np.float32)

    adj = tf_lib.CSRMatrix(
        np.ascontiguousarray(values, dtype=np.float32),
        np.ascontiguousarray(indices, dtype=np.int32),
        np.ascontiguousarray(indptr, dtype=np.int32),
        int(data.num_nodes),
        int(data.num_nodes),
    )

    schedule_start = time.perf_counter()
    level_ptr, mix_ptr, partition = tf_lib.inspect_vt(
        adj, feature.shape[1], out_dim, args.cache_size, args.threads
    )
    schedule_ms = (time.perf_counter() - schedule_start) * 1e3

    vt_ms, vt_out = median_time_ms(
        lambda: tf_lib.fusedGeMMSpMMVT(
            adj,
            weight_t,
            feature,
            tf_lib.Op.Transposed,
            level_ptr,
            mix_ptr,
            partition,
            args.threads,
        ),
        args.warmup,
        args.repeats,
    )

    torch_ms, torch_out = median_time_ms(
        lambda: torch_unfused_baseline(adj_csr_torch, feature, weight_t),
        args.warmup,
        args.repeats,
    )

    vt_np = np.asarray(vt_out)
    torch_np = torch_out.cpu().numpy()
    max_abs_diff = float(np.max(np.abs(vt_np - torch_np)))
    passed = bool(np.allclose(vt_np, torch_np, rtol=args.rtol, atol=args.atol))

    row = {
        "dataset": dataset_key,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "nnz": int(values.size),
        "in_dim": int(feature.shape[1]),
        "out_dim": int(out_dim),
        "threads": int(args.threads),
        "cache_size": int(args.cache_size),
        "schedule_ms": schedule_ms,
        "vt_ms": vt_ms,
        "torch_unfused_ms": torch_ms,
        "speedup_vs_torch": torch_ms / vt_ms if vt_ms > 0 else float("inf"),
        "max_abs_diff": max_abs_diff,
        "passed": passed,
    }

    print(
        f"{dataset_key}: nodes={row['num_nodes']} edges={row['num_edges']} "
        f"vt={vt_ms:.3f} ms torch={torch_ms:.3f} ms "
        f"speedup={row['speedup_vs_torch']:.3f} check={passed}"
    )

    if writer is not None:
        writer.writerow(row)


def print_dataset_summary(dataset_key: str, data) -> None:
    x_shape = tuple(data.x.shape) if data.x is not None else None
    print(
        f"{dataset_key}: nodes={int(data.num_nodes)} "
        f"edges={int(data.edge_index.size(1))} x_shape={x_shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load PyG GNN datasets and benchmark tf_lib variable-tile fused GeMM-SpMM."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cora"],
        help=f"Dataset keys. Known: {', '.join(sorted(DATASETS))}",
    )
    parser.add_argument("--root", type=Path, default=Path("data/pyg"))
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--in-dim", type=int, default=None)
    parser.add_argument("--out-dim", type=int, default=32)
    parser.add_argument("--cache-size", type=int, default=256 * 1024)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--csv", type=Path, default=Path("pyg_vt_results.csv"))
    args = parser.parse_args()

    rows = [
        "dataset",
        "num_nodes",
        "num_edges",
        "nnz",
        "in_dim",
        "out_dim",
        "threads",
        "cache_size",
        "schedule_ms",
        "vt_ms",
        "torch_unfused_ms",
        "speedup_vs_torch",
        "max_abs_diff",
        "passed",
    ]

    writer = None
    csv_file = None
    if not args.load_only:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.csv.open("w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=rows)
        writer.writeheader()

    try:
        for dataset_key in args.datasets:
            data = load_pyg_dataset(dataset_key, args.root)
            print_dataset_summary(dataset_key, data)
            if not args.load_only:
                run_dataset(dataset_key, data, args, writer)
    finally:
        if csv_file is not None:
            csv_file.close()

    if not args.load_only:
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()

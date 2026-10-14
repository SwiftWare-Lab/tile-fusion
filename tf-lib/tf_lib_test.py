from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import tf_lib


@dataclass(frozen=True)
class GraphInput:
    name: str
    adj: tf_lib.CSRMatrix
    torch_adj: torch.Tensor
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    rows: int
    cols: int
    feature: np.ndarray


@dataclass(frozen=True)
class TestCase:
    name: str
    schedule_name: str
    op: object
    weight: np.ndarray
    expected: np.ndarray
    run: Callable[[], np.ndarray]


@dataclass(frozen=True)
class SpMMSpMMTestCase:
    name: str
    schedule_name: str
    expected: np.ndarray
    run: Callable[[], np.ndarray]


PYG_DATASETS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


def tribanded_csr_matrix(
    rows: int,
    cols: int,
) -> tuple[tf_lib.CSRMatrix, np.ndarray, np.ndarray, np.ndarray]:
    data = []
    indices = []
    indptr = [0]

    for row in range(rows):
        for col in (row - 1, row, row + 1):
            if 0 <= col < cols:
                indices.append(col)
                data.append(1.0)
        indptr.append(len(indices))

    data_arr = np.asarray(data, dtype=np.float32)
    indices_arr = np.asarray(indices, dtype=np.int32)
    indptr_arr = np.asarray(indptr, dtype=np.int32)

    csr = tf_lib.CSRMatrix(data_arr, indices_arr, indptr_arr, rows, cols)
    return csr, data_arr, indices_arr, indptr_arr


def csr_to_torch(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.from_numpy(indptr.astype(np.int64, copy=False)),
        torch.from_numpy(indices.astype(np.int64, copy=False)),
        torch.from_numpy(data.astype(np.float32, copy=False)),
        size=shape,
    )


def load_tribanded_graph(args) -> GraphInput:
    adj, data, indices, indptr = tribanded_csr_matrix(
        rows=args.rows,
        cols=args.cols,
    )
    torch_adj = csr_to_torch(data, indices, indptr, (args.rows, args.cols))
    rng = np.random.default_rng(args.seed + 1)
    feature = rng.normal(size=(args.cols, args.in_dim)).astype(np.float32)

    return GraphInput(
        name="tribanded",
        adj=adj,
        torch_adj=torch_adj,
        data=data,
        indices=indices,
        indptr=indptr,
        rows=args.rows,
        cols=args.cols,
        feature=np.ascontiguousarray(feature, dtype=np.float32),
    )


def load_pyg_graph(dataset_key: str, root: Path, in_dim: int | None, seed: int) -> GraphInput:
    from torch_geometric.data import Data
    from torch_geometric.datasets import Planetoid

    key = dataset_key.lower()
    if key not in PYG_DATASETS:
        known = ", ".join(sorted(PYG_DATASETS))
        raise ValueError(f"Unknown PyG dataset '{dataset_key}'. Known datasets: {known}")

    try:
        dataset = Planetoid(root=str(root / key), name=PYG_DATASETS[key])
        graph = dataset[0]
    except ValueError as exc:
        processed_path = root / key / PYG_DATASETS[key] / "processed" / "data.pt"
        if "too many values to unpack" not in str(exc) or not processed_path.exists():
            raise
        data_dict, _, _ = torch.load(processed_path)
        graph = Data(**data_dict)
    rows = int(graph.num_nodes)
    cols = rows

    values = torch.ones(graph.edge_index.size(1), dtype=torch.float32)
    torch_adj = torch.sparse_coo_tensor(
        graph.edge_index.cpu(),
        values,
        (rows, cols),
    ).coalesce().to_sparse_csr()

    data = torch_adj.values().to(torch.float32).cpu().numpy()
    indices = torch_adj.col_indices().to(torch.int32).cpu().numpy()
    indptr = torch_adj.crow_indices().to(torch.int32).cpu().numpy()

    if graph.x is not None and in_dim is None:
        feature = graph.x.to(torch.float32).cpu().contiguous().numpy()
    else:
        feature_dim = in_dim if in_dim is not None else 32
        rng = np.random.default_rng(seed + 1)
        feature = rng.normal(size=(cols, feature_dim)).astype(np.float32)

    adj = tf_lib.CSRMatrix(
        np.ascontiguousarray(data, dtype=np.float32),
        np.ascontiguousarray(indices, dtype=np.int32),
        np.ascontiguousarray(indptr, dtype=np.int32),
        rows,
        cols,
    )

    return GraphInput(
        name=key,
        adj=adj,
        torch_adj=torch_adj,
        data=data,
        indices=indices,
        indptr=indptr,
        rows=rows,
        cols=cols,
        feature=np.ascontiguousarray(feature, dtype=np.float32),
    )


def load_graph(args) -> GraphInput:
    if args.pyg_dataset is not None:
        return load_pyg_graph(args.pyg_dataset, args.pyg_root, args.in_dim, args.seed)
    return load_tribanded_graph(args)


def torch_reference(
    torch_adj: torch.Tensor,
    feature: np.ndarray,
    weight: np.ndarray,
    transposed: bool,
) -> np.ndarray:
    feature_torch = torch.from_numpy(feature)
    weight_torch = torch.from_numpy(weight)
    if transposed:
        gemm = torch.mm(feature_torch, weight_torch.T)
    else:
        gemm = torch.mm(feature_torch, weight_torch)
    return torch.sparse.mm(torch_adj, gemm).cpu().numpy()


def torch_spmm_spmm_reference(
    torch_adj: torch.Tensor,
    feature: np.ndarray,
) -> np.ndarray:
    feature_torch = torch.from_numpy(feature)
    intermediate = torch.sparse.mm(torch_adj, feature_torch)
    return torch.sparse.mm(torch_adj, intermediate).cpu().numpy()


def run_and_check_case(
    case: TestCase,
    rtol: float,
    atol: float,
) -> None:
    output = case.run()
    np.testing.assert_allclose(output, case.expected, rtol=rtol, atol=atol)
    print(
        f"PASS {case.name}: schedule={case.schedule_name} "
        f"weight_shape={case.weight.shape} output_shape={output.shape}"
    )


def run_and_check_spmm_spmm_case(
    case: SpMMSpMMTestCase,
    rtol: float,
    atol: float,
) -> None:
    output = case.run()
    np.testing.assert_allclose(output, case.expected, rtol=rtol, atol=atol)
    print(
        f"PASS {case.name}: schedule={case.schedule_name} "
        f"output_shape={output.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tf_lib correctness tests for fixed/VT schedules and transposed/non-transposed weights."
    )
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--in-dim", type=int, default=32)
    parser.add_argument("--out-dim", type=int, default=32)
    parser.add_argument(
        "--spmm-spmm-dim",
        type=int,
        default=32,
        help="Feature dimension for SpMM-SpMM tests. Must be a multiple of 32.",
    )
    parser.add_argument("--pyg-dataset", choices=sorted(PYG_DATASETS), default=None)
    parser.add_argument("--pyg-root", type=Path, default=Path("data/pyg"))
    parser.add_argument("--m-tile-size", type=int, default=8)
    parser.add_argument("--cache-size", type=int, default=256 * 1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    if args.spmm_spmm_dim % 32 != 0:
        raise ValueError("--spmm-spmm-dim must be a multiple of 32 for the AVX256 SpMM-SpMM kernels")

    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed + 1)

    graph = load_graph(args)
    adj = graph.adj
    feature = graph.feature
    in_dim = feature.shape[1]
    weight = rng.normal(size=(in_dim, args.out_dim)).astype(np.float32)
    weight_t = rng.normal(size=(args.out_dim, in_dim)).astype(np.float32)

    fixed_level_ptr, fixed_mix_ptr, fixed_partition = tf_lib.inspect(
        adj,
        args.m_tile_size,
    )
    vt_level_ptr, vt_mix_ptr, vt_partition = tf_lib.inspect_vt(
        adj,
        in_dim,
        args.out_dim,
        args.cache_size,
        args.threads,
    )

    expected_non_transposed = torch_reference(
        graph.torch_adj, feature, weight, transposed=False
    )
    expected_transposed = torch_reference(
        graph.torch_adj, feature, weight_t, transposed=True
    )

    spmm_spmm_feature = rng.normal(
        size=(graph.cols, args.spmm_spmm_dim)
    ).astype(np.float32)
    spmm_spmm_expected = torch_spmm_spmm_reference(
        graph.torch_adj,
        spmm_spmm_feature,
    )
    (
        spmm_spmm_fixed_level_ptr,
        spmm_spmm_fixed_par_ptr,
        spmm_spmm_fixed_partition,
        spmm_spmm_fixed_par_type,
    ) = tf_lib.inspect_spmm_spmm_fixed(adj, args.m_tile_size)
    (
        spmm_spmm_vt_level_ptr,
        spmm_spmm_vt_par_ptr,
        spmm_spmm_vt_partition,
        spmm_spmm_vt_par_type,
    ) = tf_lib.inspect_spmm_spmm_vt(
        adj,
        args.spmm_spmm_dim,
        args.cache_size,
        args.threads,
    )

    cases = [
        TestCase(
            name="fixed_non_transposed",
            schedule_name="fixed",
            op=tf_lib.Op.NonTransposed,
            weight=weight,
            expected=expected_non_transposed,
            run=lambda: np.asarray(
                tf_lib.fusedGeMMSpMM(
                    adj,
                    weight,
                    feature,
                    tf_lib.Op.NonTransposed,
                    fixed_level_ptr,
                    fixed_mix_ptr,
                    fixed_partition,
                    args.threads,
                )
            ),
        ),
        TestCase(
            name="fixed_transposed",
            schedule_name="fixed",
            op=tf_lib.Op.Transposed,
            weight=weight_t,
            expected=expected_transposed,
            run=lambda: np.asarray(
                tf_lib.fusedGeMMSpMM(
                    adj,
                    weight_t,
                    feature,
                    tf_lib.Op.Transposed,
                    fixed_level_ptr,
                    fixed_mix_ptr,
                    fixed_partition,
                    args.threads,
                )
            ),
        ),
        TestCase(
            name="vt_non_transposed",
            schedule_name="variable_tile",
            op=tf_lib.Op.NonTransposed,
            weight=weight,
            expected=expected_non_transposed,
            run=lambda: np.asarray(
                tf_lib.fusedGeMMSpMMVT(
                    adj,
                    weight,
                    feature,
                    tf_lib.Op.NonTransposed,
                    vt_level_ptr,
                    vt_mix_ptr,
                    vt_partition,
                    args.threads,
                )
            ),
        ),
        TestCase(
            name="vt_transposed",
            schedule_name="variable_tile",
            op=tf_lib.Op.Transposed,
            weight=weight_t,
            expected=expected_transposed,
            run=lambda: np.asarray(
                tf_lib.fusedGeMMSpMMVT(
                    adj,
                    weight_t,
                    feature,
                    tf_lib.Op.Transposed,
                    vt_level_ptr,
                    vt_mix_ptr,
                    vt_partition,
                    args.threads,
                )
            ),
        ),
    ]

    spmm_spmm_cases = [
        SpMMSpMMTestCase(
            name="spmm_spmm_fixed_avx256",
            schedule_name="fixed",
            expected=spmm_spmm_expected,
            run=lambda: np.asarray(
                tf_lib.SpMMSpMMFusedInterLayerVectorizedAvx256SP(
                    adj,
                    adj,
                    spmm_spmm_feature,
                    spmm_spmm_fixed_level_ptr,
                    spmm_spmm_fixed_par_ptr,
                    spmm_spmm_fixed_partition,
                    spmm_spmm_fixed_par_type,
                    args.threads,
                )
            ),
        ),
        SpMMSpMMTestCase(
            name="spmm_spmm_vt_avx256",
            schedule_name="variable_tile",
            expected=spmm_spmm_expected,
            run=lambda: np.asarray(
                tf_lib.FusedSpMMSpMM_VT(
                    adj,
                    adj,
                    spmm_spmm_feature,
                    spmm_spmm_vt_level_ptr,
                    spmm_spmm_vt_par_ptr,
                    spmm_spmm_vt_partition,
                    spmm_spmm_vt_par_type,
                    args.threads,
                )
            ),
        ),
    ]

    print(
        f"graph={graph.name} rows={graph.rows} cols={graph.cols} "
        f"nnz={graph.data.size} feature_shape={feature.shape}"
    )

    for case in cases:
        run_and_check_case(case, rtol=args.rtol, atol=args.atol)

    for case in spmm_spmm_cases:
        run_and_check_spmm_spmm_case(case, rtol=args.rtol, atol=args.atol)

    print("All tf_lib cases passed")


if __name__ == "__main__":
    main()

# Tile-fusion

This repository is a research artifact for **"Loop Fusion in Matrix
Multiplications with Sparse Dependence"**, Mohammad Mehdi Salehi and Kazem
Cheshmi, ICS 2025.

The current supported interface is the Python/PyTorch extension under `tf-lib/`.
It exposes fused kernels for testing GNN-style sparse/dense operations from
Python.

## Repository Layout

- `tf-lib/`: Python extension exposing fused kernels and PyTorch GNN helpers.
- `fusion/`: research prototype sources, benchmarks, and examples.
- `binding-project/`: earlier Torch operator binding prototype used by GNN
  experiments.
- `modeling/` and `codegen/`: supporting research/prototype code.

## Build `tf-lib`

Build the Python extension from `tf-lib/`:

```bash
cd tf-lib
../venv/bin/python setup.py build_ext --inplace
```

The build expects MKL, OpenMP, pybind11, and PyTorch to be available. The setup
script adds the active virtual environment and PyTorch CMake paths
automatically.

`tf-lib` builds a Python extension named `tf_lib`. It exposes two classes of
functionality:

1. Low-level fused operations using NumPy arrays and `tf_lib.CSRMatrix`.
2. PyTorch GNN functions/layers using `torch.sparse_csr` tensors.

## Low-Level Fused Operations

The low-level API uses:

```python
tf_lib.CSRMatrix(data, indices, indptr, rows, cols)
```

where `data` is `float32`, and `indices`/`indptr` are `int32`.

Fused GeMM-SpMM computes:

```python
out = Adj @ (Feature @ Weight)
```

or, for transposed weight:

```python
out = Adj @ (Feature @ Weight.T)
```

Example:

```python
import numpy as np
import tf_lib


def tribanded_csr(n):
    data, indices, indptr = [], [], [0]
    for row in range(n):
        for col in (row - 1, row, row + 1):
            if 0 <= col < n:
                data.append(1.0)
                indices.append(col)
        indptr.append(len(indices))

    data = np.asarray(data, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int32)
    indptr = np.asarray(indptr, dtype=np.int32)
    return tf_lib.CSRMatrix(data, indices, indptr, n, n)


n = 64
in_dim = 32
out_dim = 32
threads = 4

adj = tribanded_csr(n)
feature = np.random.randn(n, in_dim).astype(np.float32)
weight = np.random.randn(in_dim, out_dim).astype(np.float32)

level_ptr, mix_ptr, partition = tf_lib.inspect(adj, 8)
out = tf_lib.fusedGeMMSpMM(
    adj,
    weight,
    feature,
    tf_lib.Op.NonTransposed,
    level_ptr,
    mix_ptr,
    partition,
    threads,
)
```

Variable-tile GeMM-SpMM:

```python
schedule = tf_lib.inspect_vt(adj, in_dim, out_dim, 256 * 1024, threads)
out = tf_lib.fusedGeMMSpMMVT(
    adj,
    weight,
    feature,
    tf_lib.Op.NonTransposed,
    *schedule,
    threads,
)
```

Fused SpMM-SpMM computes:

```python
out = B @ (A @ Feature)
```

For the common one-graph case, pass the same adjacency for `A` and `B`.

```python
feature = np.random.randn(n, 32).astype(np.float32)
schedule = tf_lib.inspect_spmm_spmm_vt(adj, feature.shape[1], 256 * 1024, threads)
out = tf_lib.FusedSpMMSpMM_VT(adj, adj, feature, *schedule, threads)
```

The AVX256 SpMM-SpMM kernels require the feature dimension to be a multiple of
32.

## PyTorch GNN Functions

The Torch-facing API accepts CPU `torch.sparse_csr` adjacency tensors and CPU
`float32` dense tensors. Weight tensors use PyTorch `Linear` layout:

```python
weight.shape == (out_dim, in_dim)
```

Example fused function:

```python
import torch
import tf_lib
from tf_lib_layers import fused_gemm_spmm

threads = 4
in_dim = 32
out_dim = 16

# adj: CPU torch.sparse_csr_tensor, x: [num_nodes, in_dim]
weight = torch.randn(out_dim, in_dim, requires_grad=True)
x = torch.randn(adj.size(0), in_dim, requires_grad=True)

schedule = tf_lib.torch_inspect_vt(adj, in_dim, out_dim, 500_000, threads)
y = fused_gemm_spmm(adj, x, weight, schedule, threads)
y.square().sum().backward()
```

Example GCN layer:

```python
import tf_lib
from tf_lib_layers import FusedGCNLayer, FusedGCN

schedule = tf_lib.torch_inspect_vt(adj, feat_dim, hidden_dim, 500_000, threads)
layer = FusedGCNLayer(feat_dim, hidden_dim, adj, schedule, threads)
hidden = layer(x).relu()

model = FusedGCN(feat_dim, hidden_dim, num_classes, adj, threads)
logits = model(x)
```

Available helpers:

- `tf_lib.torch_inspect`
- `tf_lib.torch_inspect_vt`
- `tf_lib.torch_fused_gemm_spmm`
- `tf_lib.torch_fused_gemm_spmm_vt`
- `tf_lib.torch_cached_spmm_gemm`
- `tf_lib.torch_fused_spmm_spmm`
- `tf_lib_layers.fused_gemm_spmm`
- `tf_lib_layers.cached_spmm_gemm`
- `tf_lib_layers.unfused_gemm_spmm`
- `tf_lib_layers.FusedGCNLayer`
- `tf_lib_layers.FirstLayerGCNCached`
- `tf_lib_layers.UnFusedGCNLayer`
- `tf_lib_layers.FusedGCN`

The fused forward path calls the C++ kernels. The PyTorch layer backward path is
implemented with PyTorch sparse/dense operations, so the layers can be used in
normal PyTorch training loops.

Run `tf-lib` correctness tests:

```bash
cd tf-lib
../venv/bin/python tf_lib_test.py
```

With a PyG dataset:

```bash
../venv/bin/python tf_lib_test.py --pyg-dataset pubmed --pyg-root data/pyg
```

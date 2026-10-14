# tf-lib

`tf-lib` exposes fused sparse/dense kernels for GNN-style workloads through a
Python extension named `tf_lib`.

The library currently has two groups of APIs:

1. Low-level fused operations that operate on NumPy arrays and `tf_lib.CSRMatrix`.
2. PyTorch GNN functions/layers that operate on `torch.sparse_csr` tensors.

## Build

From this directory:

```bash
../venv/bin/python setup.py build_ext --inplace
```

The build expects MKL, OpenMP, pybind11, and PyTorch to be available. The setup
script adds the active virtual environment and PyTorch CMake paths automatically.

## Low-Level Fused Operations

The low-level API uses a custom CSR wrapper:

```python
tf_lib.CSRMatrix(data, indices, indptr, rows, cols)
```

where `data` is `float32`, and `indices`/`indptr` are `int32`.

### Fused GeMM-SpMM

This computes:

```python
out = Adj @ (Feature @ Weight)
```

for non-transposed weight, or:

```python
out = Adj @ (Feature @ Weight.T)
```

for transposed weight.

```python
import numpy as np
import tf_lib


def tribanded_csr(n: int):
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

Variable-tile scheduling uses `inspect_vt` and `fusedGeMMSpMMVT`:

```python
level_ptr, mix_ptr, partition = tf_lib.inspect_vt(
    adj,
    in_dim,
    out_dim,
    256 * 1024,
    threads,
)
out = tf_lib.fusedGeMMSpMMVT(
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

### Fused SpMM-SpMM

This computes:

```python
out = B @ (A @ Feature)
```

For the common same-adjacency case, pass the same matrix for `A` and `B`.

The AVX256 SpMM-SpMM kernels require `Feature.shape[1]` to be a multiple of 32.

```python
feature = np.random.randn(n, 32).astype(np.float32)

schedule = tf_lib.inspect_spmm_spmm_vt(
    adj,
    feature.shape[1],
    256 * 1024,
    threads,
)

out = tf_lib.FusedSpMMSpMM_VT(
    adj,
    adj,
    feature,
    *schedule,
    threads,
)
```

Fixed schedule:

```python
schedule = tf_lib.inspect_spmm_spmm_fixed(adj, 8)
out = tf_lib.SpMMSpMMFusedInterLayerVectorizedAvx256SP(
    adj,
    adj,
    feature,
    *schedule,
    threads,
)
```

## PyTorch GNN Functions

The Torch-facing functions accept CPU `torch.sparse_csr` adjacency tensors and
CPU `float32` dense tensors.

The weight layout matches the PyTorch `Linear` convention:

```python
weight.shape == (out_dim, in_dim)
```

### Fused Function Example

```python
import torch
import tf_lib
from tf_lib_layers import fused_gemm_spmm

n = 64
in_dim = 32
out_dim = 16
threads = 4

rows, cols = [], []
for row in range(n):
    for col in (row - 1, row, row + 1):
        if 0 <= col < n:
            rows.append(row)
            cols.append(col)

crow = torch.zeros(n + 1, dtype=torch.int32)
for row in rows:
    crow[row + 1] += 1
crow = torch.cumsum(crow, dim=0).to(torch.int32)
col = torch.tensor(cols, dtype=torch.int32)
val = torch.ones(col.numel(), dtype=torch.float32)
adj = torch.sparse_csr_tensor(crow, col, val, size=(n, n))

x = torch.randn(n, in_dim, requires_grad=True)
weight = torch.randn(out_dim, in_dim, requires_grad=True)

schedule = tf_lib.torch_inspect_vt(
    adj,
    in_dim,
    out_dim,
    500_000,
    threads,
)

y = fused_gemm_spmm(adj, x, weight, schedule, threads)
loss = y.square().sum()
loss.backward()
```

For a fixed schedule:

```python
schedule = tf_lib.torch_inspect(adj, 8)
y = fused_gemm_spmm(
    adj,
    x,
    weight,
    schedule,
    threads,
    variable_tile=False,
)
```

### GCN Layer Example

```python
import torch
import tf_lib
from tf_lib_layers import FusedGCNLayer, FusedGCN

num_nodes = 2708
feat_dim = 1433
hidden_dim = 64
num_classes = 7
threads = 4

# adj must be a CPU torch.sparse_csr_tensor with float32 values.
# x must be CPU float32 with shape [num_nodes, feat_dim].

schedule = tf_lib.torch_inspect_vt(
    adj,
    feat_dim,
    hidden_dim,
    500_000,
    threads,
)

layer = FusedGCNLayer(
    feat_dim,
    hidden_dim,
    adj,
    schedule,
    threads,
)

hidden = layer(x).relu()

model = FusedGCN(
    feat_dim,
    hidden_dim,
    num_classes,
    adj,
    threads,
)
logits = model(x)
```

Available PyTorch helpers in `tf_lib_layers.py`:

- `fused_gemm_spmm`
- `cached_spmm_gemm`
- `unfused_gemm_spmm`
- `FusedGCNLayer`
- `FirstLayerGCNCached`
- `UnFusedGCNLayer`
- `FusedGCN`

The fused forward path calls the C++ kernels. Backward is implemented with
PyTorch sparse/dense operations, so these layers can participate in ordinary
PyTorch training loops.

## Testing

Run the correctness tests:

```bash
../venv/bin/python tf_lib_test.py
```

With a PyG dataset:

```bash
../venv/bin/python tf_lib_test.py --pyg-dataset pubmed --pyg-root data/pyg
```

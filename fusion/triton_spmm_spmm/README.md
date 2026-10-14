# Triton SpMM-SpMM benchmark

This benchmark contains two single-precision implementations:

- `Triton_Unfused_CSR_CSR`: two separately launched CSR SpMM kernels.
- `Triton_Fused_CSR_CSC_ConflictReduced`: computes producer rows from CSR and
  immediately scatters through CSC. A program merges equal consumer
  destinations across its producer-row tile before issuing an atomic update.

It also reports a two-call unfused cuSPARSE baseline through CUDA PyTorch's
preallocated CSR `torch.addmm` path.

Both kernels use Triton's autotuner. It selects `BLOCK_ROWS` from 1, 2, 4, or
8 and `BLOCK_FEATURES` from 32, 64, or 128 independently for each matrix shape
and feature size. Every candidate uses at most four warps, so no candidate has
more than 128 CUDA threads per program/thread block.

The dataset resolver prefers existing Matrix Market files in this repository.
Only missing PyG/OGB datasets are downloaded and materialized beneath
`fusion/data/triton-gnn`.

Prepare or locate all datasets without requiring a GPU:

```bash
venv/bin/python fusion/triton_spmm_spmm/benchmark.py --prepare-only
```

Prepare RCM-ordered matrices (existing full-graph caches are reused):

```bash
venv/bin/python fusion/triton_spmm_spmm/benchmark.py \
  --ordering rcm --prepare-only --no-download
```

Run the default 13-dataset, 32/64/128-feature sweep:

```bash
python fusion/triton_spmm_spmm/benchmark.py
```

Run the same sweep on RCM-ordered matrices:

```bash
python fusion/triton_spmm_spmm/benchmark.py --ordering rcm \
  --no-download --output fusion/results/triton_spmm_spmm_gnn_rcm.csv
```

The Python environment used for the second command must have CUDA-enabled
PyTorch and Triton. Results default to
`fusion/results/triton_spmm_spmm_gnn.csv`.

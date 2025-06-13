
import triton
import torch
import triton.language as tl
from numba import cuda
from utils import get_matrix_list
import sys

DEVICE = torch.device("cuda:0")

@cuda.jit()
def spmm_numba_seqreduce_kernel(
        data, indices, ind_ptr,
        b, c,
        M, N, K
):
    row = cuda.threadIdx.y + cuda.blockIdx.x * cuda.blockDim.y
    bcol = cuda.threadIdx.x + cuda.blockIdx.y * cuda.blockDim.x
    if bcol < N:
        start = ind_ptr[row]
        end = ind_ptr[row+1]
        res = 0
        for p in range(start,end):
            col = indices[p]
            val = data[p]
            res += val * b[col][bcol]
        c[row][bcol] = res



@triton.jit
def spmm_kernel(
        # Pointers to matrices
        data_ptr, indices_ptr, indptr_ptr, # A
        b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Shared memory to store the tiles of A and B
    #row_block, col_block = tl.program_id(0), tl.program_id(1)
    row_block = pid_m
    col_block = pid_n
    row_start = row_block * BLOCK_SIZE_M #+ tl.arange(0, BLOCK_SIZE_M)
    col_start = col_block * BLOCK_SIZE_N #+ tl.arange(0, BLOCK_SIZE_N)
    thread_idxx = row_start + tl.arange(0, BLOCK_SIZE_M)
    thread_idxy = col_start + tl.arange(0, BLOCK_SIZE_N)
    for row in range(row_start, row_start+BLOCK_SIZE_M):
        is_valid_row = row < M
        start = tl.load(indptr_ptr + row, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + row + 1, mask=is_valid_row, other=0)
        for i in range(start, end):
            col_idx = tl.load(indices_ptr + i)
            val = tl.load(data_ptr + i)
            b_val = tl.load(b_ptr + col_idx * N + thread_idxy, mask=thread_idxy < N,)
            c_val = val * b_val
            c_ptrs = c_ptr + row * N + thread_idxy
            #tl.atomic_add(c_ptrs, c_val)
            tl.store(c_ptrs, tl.load(c_ptrs) + c_val)


@triton.jit
def spmm_kernel_naive(
        # Pointers to matrices
        data_ptr, indices_ptr, indptr_ptr, # A
        b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Shared memory to store the tiles of A and B
    row_block, col_block = tl.program_id(0), tl.program_id(1)
    row_start = row_block * BLOCK_SIZE_M #+ tl.arange(0, BLOCK_SIZE_M)
    col_start = col_block * BLOCK_SIZE_N #+ tl.arange(0, BLOCK_SIZE_N)
    print("row", row_start)
    print("col", col_start)
    thread_idxx = row_start + tl.arange(0, BLOCK_SIZE_M)
    thread_idxy = col_start + tl.arange(0, BLOCK_SIZE_N)
    #print("dd", row_block)
    #print(row)
    #row, col = BLOCK_SIZE_M, BLOCK_SIZE_M
    for row in range(row_start, row_start+BLOCK_SIZE_M):
        is_valid_row = row < M
        start = tl.load(indptr_ptr + row, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + row + 1, mask=is_valid_row, other=0)
        for kk in range(col_start, col_start+BLOCK_SIZE_N):
            for i in range(start, end):
                col_idx = tl.load(indices_ptr + i)
                val = tl.load(data_ptr + i)
                b_val = tl.load(b_ptr + col_idx * N + kk)
                c_val = val * b_val
                c_ptrs = c_ptr + row * N + kk
                tl.atomic_add(c_ptrs, c_val)
            #tile_c += val * b_val

    #tl.store(c_ptr, 10)


def spmm_numba_seqreduce(data, indices, indptr, b, M, N, K):
    c = torch.zeros(M, N, dtype=torch.float32, device=DEVICE)
    tpb = 128
    n_block_size = min(N, tpb)
    m_block_size = (tpb + n_block_size - 1) // n_block_size
    grid_dim_x = (M + m_block_size - 1) // m_block_size
    grid_dim_y = (N + n_block_size - 1) // n_block_size
    block_dim_x = n_block_size
    block_dim_y = m_block_size

    spmm_numba_seqreduce_kernel[(grid_dim_x, grid_dim_y), (block_dim_x, block_dim_y)](data, indices, indptr, b, c, M, N, K)
    return c

def spmm_triton(data, indices, indptr, b, M, N, K):
    assert b.dtype == torch.float32, "b must be float32"
    assert data.dtype == torch.float32, "data must be float32"
    assert indices.dtype == torch.int32, "indices must be int32"
    assert indptr.dtype == torch.int32, "indptr must be int32"

    # Allocate output matrix
    c = torch.zeros(M, N, dtype=torch.float32, device=DEVICE)

    # Launch kernel
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 32
    GROUP_SIZE_M = 8
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    spmm_kernel[grid](
        data, indices, indptr, b, c, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )
    # grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    # spmm_kernel_naive[grid](
    #     data, indices, indptr, b, c, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N
    #)


    return c


def spmm_torch(data, indices, indptr, b, M, N, K):
    a = torch.sparse_csr_tensor(indptr, indices, data, dtype=torch.float32)
    c = torch.sparse.mm(a, b)
    return c


def spmm_csr_cpu(m, data, indices, indptr, b, c):
    for i in range(m):
        start = indptr[i]
        end = indptr[i + 1]
        for j in range(start, end):
            for k in range(b.shape[1]):
                c[i, k] += data[j] * b[indices[j], k]
    return c


def spmm_csr_blocked_cpu(m, data, indices, indptr, b, c, block_size_m=32, block_size_n=32):
    for i in range(0, m, block_size_m):
        for j in range(0, b.shape[1], block_size_n):
            c_tile = c[i:i + block_size_m, j:j + block_size_n]
            for ii in range(i, min(i + block_size_m, m)):
                start = indptr[ii]
                end = indptr[ii + 1]
                for jj in range(j, min(j + block_size_n, b.shape[1])):
                    for k in range(start, end):
                        c_tile[ii - i, jj - j] += data[k] * b[indices[k], jj]
    return c


def spmm_csr_blocked1_cpu(m, data, indices, indptr, b, c, block_size_m=32, block_size_n=32):
    for i in range(0, m, block_size_m):
        for j in range(0, b.shape[1], block_size_n):
            c_tile = c[i:i + block_size_m, j:j + block_size_n]
            for ii in range(i, min(i + block_size_m, m)):
                start = indptr[ii]
                end = indptr[ii + 1]
                for k in range(start, end):
                    jj = range(j, min(j + block_size_n, b.shape[1]))
                    b_row_jj = b[indices[k], jj]
                    c_tile[ii - i, jj] += data[k] * b_row_jj
    return c


# test spmm_csr_cpu and spmm_csr_blocked_cpu

import numpy as np
import cupy as cp
import scipy.sparse as sp
import scipy.io as scio


if len(sys.argv) != 3:
    print("Usage: python sddmm_spmm.py <file_path> <data_path>")
    sys.exit(1)
file_path = sys.argv[1]
data_path = sys.argv[2]
mtx_list = get_matrix_list(file_path, data_path)
method_list = ["triton", "torch", 'numba']

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["matrix"],  # Argument names to use as an x-axis for the plot
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=method_list,  # Label name for the lines
        line_names=["triton", "torch", 'numba'],  # Name of the lines
        styles=[("green", "-"), ("blue", "-"), ("red", "-"), ("gold", "-"), ("purple", "-")],  # Visual styles for the lines
        ylabel="GFLOP/S",  # Label name for the y-axis
        plot_name="gemm-spmm-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={}
    ))


@triton.testing.perf_report(configs)
def benchmark(matrix, provider):
    matrix = scio.mmread(matrix)
    A = matrix.tocsr()
    M, K = A.shape
    N = 32
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    quantiles = [0.5, 0.2, 0.8]
    rep, warmpup = 100, 25
    if provider=='torch':
        A_data = torch.tensor(A.data, dtype=torch.float32, device=DEVICE)
        A_indices = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        A_indptr = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        d_B = torch.tensor(B, dtype=torch.float32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: spmm_torch(A_data, A_indices, A_indptr, d_B, M, N, K), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider=='triton':
        A_data = torch.tensor(A.data, dtype=torch.float32, device=DEVICE)
        A_indices = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        A_indptr = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        d_B = torch.tensor(B, dtype=torch.float32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: spmm_triton(A_data, A_indices, A_indptr, d_B, M, N, K), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider=='numba':
        data_d = cuda.to_device(A.data)
        indices_d = cuda.to_device(A.indices)
        indptr_d = cuda.to_device(A.indptr)
        B_d = cuda.to_device(B)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: spmm_numba_seqreduce(data_d, indices_d, indptr_d, B_d, M, N, K), quantiles=quantiles, warmup=warmpup, rep=rep)
    return ms, min_ms, max_ms

def correctness_test():
    M, N, K = 16, 32, 64
    density = 0.001
    # generate sparse matrix
    A = sp.random(M, K, density=density, format='csr', dtype=np.float32)
    if len(sys.argv) != 3:
        print("Usage: python sddmm_spmm.py <file_path> <data_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    data_path = sys.argv[2]
    mtx_list = get_matrix_list(file_path, data_path)

    for mat_path in mtx_list:
        print(f"------------ matrix: {mat_path}")
        matrix = scio.mmread(mat_path)
        A = matrix.tocsr()
        M, K = A.shape
        print(M, K)

        B = np.random.rand(K, N).astype(np.float32)


        # convert to cupy
        A_data = torch.tensor(A.data, dtype=torch.float32, device=DEVICE)
        A_indices = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        A_indptr = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        d_B = torch.tensor(B, dtype=torch.float32, device=DEVICE)
        # run torch.sparse.mm
        c1 = spmm_torch(A_data, A_indices, A_indptr, d_B, M, N, K)
        c1 = c1.cpu().numpy()

        print(c1)

        # run triton kernel
        c3 = spmm_triton(A_data, A_indices, A_indptr, d_B, M, N, K)
        c3 = c3.cpu().numpy()
        # print(c3[:1, :1])
        # print(c1[:1, :1])
        if cp.allclose(c1, c3, atol=1e-5):
            print("Passed Triton")
        else:
            print("FailedTriton")
            # print where it fails
            print(c3)


        data_d = cuda.to_device(A.data)
        indices_d = cuda.to_device(A.indices)
        indptr_d = cuda.to_device(A.indptr)
        B_d = cuda.to_device(B)

        c4 = spmm_numba_seqreduce(data_d, indices_d, indptr_d, B_d, M, N, K)
        if cp.allclose(c1, c4, atol=1e-5):
            print("Passed Numba")
        else:
            print("Failed Numba")
            # print where it fails
            print(c4)


# main entry
if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=".", show_plots=False)
    # correctness_test()
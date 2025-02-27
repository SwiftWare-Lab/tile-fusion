import cupy as cp
import cupyx.scipy.sparse as sp
from cupy_backends.cuda.libs import cusparse
from cupyx.cusparse import spmm

import numpy as np
import cupy as cp  # For GPU operations
# import csr_matrix as cp_csr_matrix  # GPU sparse matrix
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix  # GPU sparse matrix
import scipy.io as scio
import torch
import triton
import triton.language as tl

from numba import cuda, int32, float64, void
from numba import config

from utils import get_matrix_list
from gemm import matmul_triton, matmul_triton_a_stationary
from spmm import spmm_triton
from utils import convert_csr_matrix_to_power2_dim

DEVICE = torch.device("cuda:0")
REAL_TYPE = torch.float32
config.CUDA_ENABLE_PYNVJITLINK = 1




@triton.jit
def spmm_gemm_kernel(M,
        # Pointers to matrices
        data_ptr, indices_ptr, indptr_ptr, # A
        b_ptr, c_ptr, d1_ptr, d_ptr,
        # Matrix dimensions
        N, K,
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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

    d1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
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
            d1_row = d1 + row - row_start
            tl.add(d1_row + thread_idxy, c_val)
            #c_ptrs = d1_ptr + row * N + thread_idxy
            #tl.atomic_add(c_ptrs, c_val)
            #tl.store(c_ptrs, tl.load(c_ptrs) + c_val)

    # GEMM
    pid_k = pid_n # HACK
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_ak = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_bk = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    #offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    #&A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);

    # create a copy of the block of A
    # a[i:i + block_size_m, k:k + block_size_k]
    #a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_ak [None, :]*stride_ak)
    #a_block = tl.load(a_ptrs, mask=offs_ak[None, :] < K, other=0.0)
    a_block = d1
    #print("a block", a_block)

    c_ptrs = c_ptr + (offs_bk[:, None]*stride_bk + offs_bn[None, :]*stride_bn)
    d_ptrs = d_ptr + (offs_am[:, None]*stride_cm + offs_bn[None, :]*stride_cn)
    for j in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # load the block of B
        b_block = tl.load(c_ptrs)
        #print("b block", b_block)
        pmul = tl.dot(a_block, b_block)
        # atomic add to the output C
        tl.atomic_add(d_ptrs, pmul)
        #tl.store(c_ptrs, tl.load(c_ptrs) + pmul)
        d_ptrs += BLOCK_SIZE_N * stride_cn
        c_ptrs += BLOCK_SIZE_N * stride_bn



def spmm_gemm_tile_fused_triton(M, d_Adata, d_Aindices, d_Aindptr, d_B, d_C):

    # Allocate output matrix
    D = torch.zeros(M, d_C.shape[1], dtype=torch.float32, device=DEVICE)
    D1 = torch.zeros(M, d_C.shape[1], dtype=torch.float32, device=DEVICE)

    # Launch kernel
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
    GROUP_SIZE_M = 8
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(d_C.shape[1], BLOCK_SIZE_N), )
    spmm_gemm_kernel[grid](M,
        d_Adata, d_Aindices, d_Aindptr,
        d_B, d_C, D1, D,
        d_C.shape[0], d_C.shape[1], # C
        M, d_B.stride(1), # equal to D1 is M x K
        d_C.shape[0], d_C.shape[1], # C K x N
        D.stride(0), D.stride(1), # D
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    return D



def spmm_gemm_unfused_cusparse(A, B, C):
    """
    Perform fused SpMM (D1 = A * B) and GeMM (D = D1 * C) on GPU.

    Parameters:
        A (cupyx.scipy.sparse.csr_matrix): Sparse matrix A.
        B (cupy.ndarray): Dense matrix B.
        C (cupy.ndarray): Dense matrix C.

    Returns:
        cupy.ndarray: Result of the fused operation D = (A * B) * C.
    """
    # Ensure A is in CSR format
    if not sp.isspmatrix_csr(A):
        raise ValueError("A must be a CSR sparse matrix")

    # Ensure B and C are dense matrices
    if not isinstance(B, cp.ndarray) or not isinstance(C, cp.ndarray):
        raise ValueError("B and C must be dense matrices")
    # make sure B is in fortran order


    # Perform SpMM: D1 = A * B
    D1 = spmm(A, B)
    #print(D1[0:2])
    # Perform GeMM: D = D1 * C
    D = cp.dot(D1, C)
    return D


def spmm_gemm_unfused_triton(M, d_Adata, d_Aindices, d_Aindptr, d_B, d_C):


    # Perform SpMM: D1 = A * B
    D1 = spmm_triton(d_Adata, d_Aindices, d_Aindptr, d_B, M, d_B.shape[1], d_B.shape[0])
    #print(D1[0:2])
    # Perform GeMM: D = D1 * C
    D = matmul_triton_a_stationary(D1, d_C)
    return D, D1


def spmm_gemm_unfused_cpu(A, B, C):
    # Ensure B and C are dense matrices
    if not isinstance(B, np.ndarray) or not isinstance(C, np.ndarray):
        # conver cupy array to np array
        B = cp.asnumpy(B)
        C = cp.asnumpy(C)

    # Perform SpMM: D1 = A * B
    D1 = A.dot(B)
    #print(D1[0:2])
    # Perform GeMM: D = D1 * C
    D = D1.dot(C)
    return D, D1




device = cuda.get_current_device()
print(f"Using GPU: {device.name}")
print(f"Compute Capability: {device.compute_capability}")



mtx_list = [
    #"/home/kazem/Downloads/LFAT5/LFAT5.mtx", "/home/kazem/Downloads/ex5/ex5.mtx", "/home/kazem/Downloads/mesh1e1/mesh1e1.mtx","/home/kazem/Downloads/nos4/nos4.mtx",
    "/home/kazem/Downloads/bcsstk03/bcsstk03.mtx",
    "/home/kazem/UFDB/Trefethen_20000b/Trefethen_20000b.mtx",
    "/home/kazem/Downloads/msc23052/msc23052.mtx",
    "/home/kazem/Downloads/olafu/olafu.mtx",
    "/home/kazem/Downloads/gyro_k/gyro_k.mtx",
    "/home/kazem/Downloads/cant/cant.mtx"
            ]
method_list = [ "cusparse",  "tile fused", "unfused triton"]
#mtx_list = get_matrix_list("/home/kazem/UFDB/spd_list.txt", "/home/kazem/UFDB/")


for matrices in mtx_list:
    print(f"-----------------> Matrix: {matrices}")
    matrix = scio.mmread(matrices)
    # hack
    A_sp = matrix.tocsr()
    A_sp = convert_csr_matrix_to_power2_dim(A_sp)
    A = cp_csr_matrix(A_sp.astype(cp.float32))
    #A = sp.random(16, 64, density=0.01, format='csr', dtype=np.float32)
    n = 64
    B = cp.random.rand(A.shape[1], n).astype(np.float32)
    C = cp.random.rand(n, 32).astype(cp.float32)

    # # cpu solution
    # # conver cupy A to scipy A
    D_cpu, D1_cpu = spmm_gemm_unfused_cpu(A_sp.tocsr(), B, C)

    # cusparse spmm-gemm
    B_f = cp.asfortranarray(B)
    D = spmm_gemm_unfused_cusparse(A, B_f, C)

    # convert to cupy
    A_data = torch.tensor(A.data, dtype=torch.float32, device=DEVICE)
    A_indices = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
    A_indptr = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
    d_B = torch.tensor(B, dtype=torch.float32, device=DEVICE)
    d_C = torch.tensor(C, dtype=torch.float32, device=DEVICE)

    # triton unfuded spmm-gemm
    D_u_triton, D1_u_triton = spmm_gemm_unfused_triton(A.shape[0], A_data, A_indices, A_indptr, d_B, d_C)

    D_fused_triton = spmm_gemm_tile_fused_triton(A.shape[0], A_data, A_indices, A_indptr, d_B, d_C)

    # compare results
    if not cp.allclose(D_cpu, D_u_triton, atol=1e-6):
        print("Results unfused triton do not match")
        print("diffs", np.sum(cp.asnumpy(D_cpu)-cp.asnumpy(D_u_triton.cpu())))
        print("diff D1", np.sum(cp.asnumpy(D1_cpu)-cp.asnumpy(D1_u_triton.cpu())))
    else:
        print(f"U Triton Results match for {matrices}")

    if not cp.allclose(D_cpu, D_fused_triton, atol=1e-6):
        print("Results fused triton do not match")
        print("diffs", np.sum(cp.asnumpy(D_cpu)-cp.asnumpy(D_fused_triton.cpu())))
    else:
        print(f"Fused Triton Results match for {matrices}")








configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["matrices"],  # Argument names to use as an x-axis for the plot
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=method_list,  # Label name for the lines
        line_names=["cuSparse", "Tiled Fused", "Unfused Triton"],  # Name of the lines
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],  # Visual styles for the lines
        ylabel="GFLOP/S",  # Label name for the y-axis
        plot_name="gemm-spmm-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={}
    ))


# use triton benchmark
@triton.testing.perf_report(configs)
def benchmark(matrices, provider):
#    for mtx_matrix_path in mtx_list:
    # read matrix from file
    matrix = scio.mmread(matrices)
    A = matrix.tocsr()
    A = cp_csr_matrix(A.astype(cp.float32))
    n = 32
    B = cp.random.rand(A.shape[1], n).astype(cp.float32)

    C = cp.random.rand(n, n).astype(cp.float32)
    quantiles = [0.5, 0.2, 0.8]
    rep, warmpup = 100, 25
    if provider == 'tile fused':
        A_data = torch.tensor(A.data, dtype=torch.float32, device=DEVICE)
        A_indices = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        A_indptr = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        d_B = torch.tensor(B, dtype=torch.float32, device=DEVICE)
        d_C = torch.tensor(C, dtype=torch.float32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: spmm_gemm_tile_fused_triton(A.shape[0], A_data, A_indices, A_indptr, d_B, d_C), quantiles=quantiles, warmup=warmpup, rep=rep)

    if provider == 'cusparse':
        B_f = cp.asfortranarray(B)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: spmm_gemm_unfused_cusparse(A, B_f, C), quantiles=quantiles, warmup=warmpup, rep=rep)

    if provider == 'unfused triton':
        A_data = torch.tensor(A.data, dtype=torch.float32, device=DEVICE)
        A_indices = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        A_indptr = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        d_B = torch.tensor(B, dtype=torch.float32, device=DEVICE)
        d_C = torch.tensor(C, dtype=torch.float32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: spmm_gemm_unfused_triton(A.shape[0], A_data, A_indices, A_indptr, d_B, d_C), quantiles=quantiles, warmup=warmpup, rep=rep)
    perf = lambda ms: (2 * A.nnz * B.shape[1] + 2 * A.shape[0] * B.shape[1] * C.shape[1]) / (ms * 1e-3) / 1e9
    #print(f"seconds: {ms}")
    #print(f"max: {perf(max_ms)}, min: {perf(min_ms)}, median: {perf(ms)}")
    #print(f"{provider} -- {matrices} -- seconds: {ms}")
    return perf(ms), perf(min_ms), perf(max_ms)

# Call the benchmark function
benchmark.run(print_data=True, save_path=".", show_plots=True)
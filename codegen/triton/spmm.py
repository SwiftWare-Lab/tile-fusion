
import triton
import torch
import triton.language as tl

DEVICE = torch.device("cuda:0")


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

# main entry
if __name__ == "__main__":
    # Generate random sparse matrix
    M, N, K = 16, 64, 64
    density = 0.001
    block_size_m, block_size_n = 32, 32
    # generate sparse matrix
    A = sp.random(M, K, density=density, format='csr', dtype=np.float32)
    matrix = scio.mmread("/home/kazem/Downloads/LFAT5/LFAT5.mtx")
    # hack
    A = matrix.tocsr()
    M, K = A.shape

    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    c1 = spmm_csr_cpu(M, A.data, A.indices, A.indptr, B, C.copy())
    c2 = spmm_csr_blocked_cpu(M, A.data, A.indices, A.indptr, B, C.copy(), block_size_m, block_size_n)

    if np.allclose(c1, c2, atol=1e-6):
        print("Passed blocked")
    else:
        print("Failed blocked")


    # convert to cupy
    A_data = torch.tensor(A.data, dtype=torch.float32, device=DEVICE)
    A_indices = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
    A_indptr = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
    d_B = torch.tensor(B, dtype=torch.float32, device=DEVICE)

    # run triton kernel
    c3 = spmm_triton(A_data, A_indices, A_indptr, d_B, M, N, K)
    c3 = c3.cpu().numpy()
    # print(c3[:1, :1])
    # print(c1[:1, :1])
    if cp.allclose(c1, c3, atol=1e-6):
        print("Passed Triton")
    else:
        print("FailedTriton")
        # print where it fails
        print(np.where(c1 != c3))


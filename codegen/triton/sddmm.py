import triton
import torch
import triton.language as tl
from numba import njit
import numba
import numpy as np
import scipy.io as scio
import time
from numba import cuda


DEVICE = torch.device("cuda:0")
NUM_RUNS=1

S_STATIONARY_BLOCK_SIZE = 128
MBLOCK_SIZE = 32
KBLOCK_SIZE = 32


#TODO: Later add shared memory too see if it has any effect

## needs atomic since each thread corresponds to one column of dense and on row of sparse matrix
@cuda.jit()
def sddmm_kernel_CSR_u_stationary_atomic(
        row_ptr, col_ind, data_ptr,
        u_data, v_data,
        M, N, K,
        res_data_ptr
):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    cuda.syncthreads()
    nnz_start = row_ptr[row]
    nnz_end = row_ptr[row+1]
    u_el = u_data[row][col]
    if row < M and col < K:
        for i in range(nnz_start, nnz_end):
            v_row = col_ind[i]
            res = data_ptr[i] * u_el * v_data[v_row][col]
            cuda.atomic.add(res_data_ptr, i, res)

@cuda.jit()
def sddmm_kernel_CSR_u_stationary_par_reduce(
        row_ptr, col_ind, data_ptr,
        u_data, v_data,
        M, N, K,
        res_data_ptr
):
    SA = cuda.shared.array(shape=(MBLOCK_SIZE, KBLOCK_SIZE), dtype=numba.float32)
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    cuda.syncthreads()
    nnz_start = row_ptr[row]
    nnz_end = row_ptr[row+1]
    u_el = u_data[row][col]
    sa_row = cuda.threadIdx.y
    sa_col = cuda.threadIdx.x
    if row < M and col < K:
        for i in range(nnz_start, nnz_end):
            v_row = col_ind[i]
            res = data_ptr[i] * u_el * v_data[v_row][col]
            SA[sa_row][sa_col] = res
            cuda.syncthreads()
            stride = KBLOCK_SIZE // 2
            while stride > 0:
                cuda.syncthreads()
                if sa_col < stride:
                    SA[sa_row][sa_col] += SA[sa_row][sa_col + stride]
                stride = stride // 2
            if sa_col == 0:
                res_data_ptr[i] = SA[sa_row][0]

@cuda.jit()
def sddmm_kernel_COO_packed_s_stationary(
        row_ptr, col_ind, row_ind, data_ptr,
        u_data, v_data,
        M, N, K,
        res_data_ptr
):
    SA = cuda.shared.array(shape=(S_STATIONARY_BLOCK_SIZE, KBLOCK_SIZE), dtype=numba.float32)
    rows_per_block = S_STATIONARY_BLOCK_SIZE
    step = MBLOCK_SIZE
    start_row = cuda.blockIdx.y * rows_per_block + cuda.threadIdx.y
    end_row = min(M, start_row + rows_per_block)

    for i in range(start_row, end_row, step):
        u_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        SA_x = cuda.threadIdx.x
        SA_y = i - start_row
        SA[SA_y][SA_x] = u_data[i][u_x]
    cuda.syncthreads()
    thread_linear_index = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    nnz_start = row_ptr[start_row] + thread_linear_index
    nnz_end = row_ptr[end_row + 1]
    step = cuda.blockDim.x * cuda.blockDim.y
    if nnz_start < nnz_end:
        for i in range(nnz_start, nnz_end, step):
            row = row_ind[i]
            col = col_ind[i]
            for j in range(K):
                res_data_ptr[i] += data_ptr[i] * SA[row - start_row][j] * v_data[col][j]


@cuda.jit()
def sddmm_kernel_s_stationary(
        row_ptr, col_ind, data_ptr,
        u_data, v_data,
        M, N, K,
        res_data_ptr
):
    SA = cuda.shared.array(shape=(S_STATIONARY_BLOCK_SIZE, KBLOCK_SIZE), dtype=numba.float32)
    rows_per_block = S_STATIONARY_BLOCK_SIZE
    start_row = cuda.blockIdx.y * rows_per_block
    end_row = min(M, start_row + rows_per_block)
    for i in range(start_row, end_row):
        u_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        SA_x = cuda.threadIdx.x
        SA_y = i - start_row
        SA[SA_y][SA_x] = u_data[i][u_x]
    cuda.syncthreads()
    step = cuda.blockDim.x
    for row in range(start_row, end_row):
        nnz_start = row_ptr[row] + cuda.threadIdx.x
        nnz_end = row_ptr[row + 1]
        if nnz_start < nnz_end:
            for i in range(nnz_start, nnz_end, step):
                col = col_ind[i]
                for j in range(K):
                    res_data_ptr[i] += data_ptr[i] * SA[row - start_row][j] * v_data[col][j]

@cuda.jit()
def sddmm_kernel_COO_packed_s_stationary(
        row_ptr, col_ind, row_ind, data_ptr,
        u_data, v_data,
        M, N, K,
        res_data_ptr
):
    SA = cuda.shared.array(shape=(S_STATIONARY_BLOCK_SIZE, KBLOCK_SIZE), dtype=numba.float32)
    rows_per_block = S_STATIONARY_BLOCK_SIZE
    step = MBLOCK_SIZE
    start_row = cuda.blockIdx.y * rows_per_block + cuda.threadIdx.y
    end_row = min(M, start_row + rows_per_block)

    for i in range(start_row, end_row, step):
        u_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        SA_x = cuda.threadIdx.x
        SA_y = i - start_row
        SA[SA_y][SA_x] = u_data[i][u_x]
    cuda.syncthreads()
    thread_linear_index = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    nnz_start = row_ptr[start_row] + thread_linear_index
    nnz_end = row_ptr[end_row + 1]
    step = cuda.blockDim.x * cuda.blockDim.y
    if nnz_start < nnz_end:
        for i in range(nnz_start, nnz_end, step):
            row = row_ind[i]
            col = col_ind[i]
            for j in range(K):
                res_data_ptr[i] += data_ptr[i] * SA[row - start_row][j] * v_data[col][j]

## each thread corresponds to one nonzero
# @cuda.jit()
# def sddmm_kernel_CSR(
#         row_ptr, col_ind, data_ptr,
#         u_data, v_data,
#         M, N, K,
#         res_data_ptr
# ):
#     SA = cuda.shared.array(shape=(MBLOCK_SIZE, K_BLOCK_SIZE), dtype=numba.float32)
#     k_block_num = cuda.gridDim.x
#     for k in range(k_block_num):
#         u_x = cuda.blockIDx.x * cuda.blockIdx.y
#         SA_x = cuda.threadIdx.x
#         SA_y = cuda.threadIdx.y
#         SA[SA_y][SA_x] = SA_x
#     row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
#     u_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#     nnz_start = row_ptr[row]
#     nnz_end = row_ptr[row+1]
#     if row < M and u_col < K:
#         for i in range(nnz_start, nnz_end):
#             v_col = col_ind[i]
#             res_data_ptr[i] += data_ptr[i] * u_data[u_col] * v_data[v_col]

# @cuda.reduce(DEVICE=True)
# def sum_reduce(a,b):
#     return a+b

@triton.jit
def sddmm_kernel_CSR(
        row_ptr, col_ind, data_ptr,
        u_data, v_data,
        M, N, K,
        res_data_ptr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    m_block_start = pid * BLOCK_SIZE_M
    block_size_m = min(M - m_block_start, BLOCK_SIZE_M)
    for row in range(m_block_start, m_block_start + block_size_m):
        offs_u = row * K + tl.arange(0, BLOCK_SIZE_K)
        u_vec = tl.load(u_data + offs_u)
        start = tl.load(row_ptr + row)
        end = tl.load(row_ptr + row + 1)
        for i in range(start, end):
            col_idx = tl.load(col_ind + i)
            offs_v = col_idx * K + tl.arange(0, BLOCK_SIZE_K)
            v_vec = tl.load(v_data + offs_v)
            val = tl.load(data_ptr + i)
            uv = tl.sum(u_vec * v_vec)
            tl.store(res_data_ptr + i, uv * val)

def sddmm_triton(indptr, indices, data, u, v, M, N, K):
    assert indptr.dtype == torch.int32, "Matrix indptr should be int32"
    assert indices.dtype == torch.int32, "Matrix indices should be int32"
    assert data.dtype == torch.float32, "Matrix data should be float32"

    res_data = torch.zeros(data.size(0), dtype=torch.float32, device=DEVICE)
    BLOCK_SIZE_M = 4
    META = {'BLOCK_SIZE_M': BLOCK_SIZE_M, 'BLOCK_SIZE_K': K}
    grid = (triton.cdiv(M, BLOCK_SIZE_M), )
    sddmm_kernel_CSR[grid](
        indptr, indices, data, u, v, M, N, K, res_data, META['BLOCK_SIZE_M'], META['BLOCK_SIZE_K'])

    return res_data


@njit(numba.float32[:](numba.int32[:], numba.int32[:], numba.float32[:], numba.float32[:,:], numba.float32[:,:], numba.int64, numba.int64, numba.int64))
def sddmm_cpu(row_ptr, col_ind, data_ptr,
              u_data, v_data,
              M, N, K):
    n = len(data_ptr)
    res_data_ptr = np.zeros(n, dtype=np.float32)
    for row in range(M):
        for i in range(row_ptr[row], row_ptr[row + 1]):
            col_idx = col_ind[i]
            val = data_ptr[i]
            uv = np.dot(u_data[row],v_data[col_idx])
            res_data_ptr[i] = uv * val
    return res_data_ptr

def get_matrix_list(file_path, base_path):
    with open(file_path, 'r') as f:
        matrix_list = f.readlines()
    matrix_list = [x.strip() for x in matrix_list]
    matrix_list = [base_path + x for x in matrix_list]
    # remove bundle1.mtx from the list
    matrix_list = [x for x in matrix_list if "bundle1.mtx" not in x]
    # sort matrix_list by nnz
    nnz_list, new_matrixlist = [], []
    for matrix in matrix_list:
        print(f"------------ matrix: {matrix}")
        A = scio.mmread(matrix)
        if A.data.dtype != np.float64:
            print(f"Matrix {matrix} is not double precision")
            A.data = A.data.astype(np.float64)
        nnz_list.append(A.nnz)
    matrix_list = [x for _, x in sorted(zip(nnz_list, matrix_list))]
    return matrix_list



#TODO: move data extraction part to the benchmark function
def gpu_sddmm_correctness(matrix, u, v):
    print("Running SDDMM on GPU ...")
    indptr = torch.tensor(matrix.indptr, dtype=torch.int32, device=DEVICE)
    indices = torch.tensor(matrix.indices, dtype=torch.int32, device=DEVICE)
    data = torch.tensor(matrix.data, dtype=torch.float32, device=DEVICE)
    u = torch.tensor(u, dtype=torch.float32, device=DEVICE)
    v = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    x_gpu = sddmm_triton(indptr, indices, data, u, v, matrix.shape[0], matrix.shape[1], u.shape[1])
    res_data = x_gpu.cpu().numpy()
    return res_data

def torch_sddmm_correctness(matrix,u,v):
    print("Running torch SDDMM ...")
    indptr = torch.tensor(matrix.indptr, dtype=torch.int32, device=DEVICE)
    indices = torch.tensor(matrix.indices, dtype=torch.int32, device=DEVICE)
    data = torch.tensor(matrix.data, dtype=torch.float32, device=DEVICE)
    u = torch.tensor(u, dtype=torch.float32, device=DEVICE)
    v = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    csr_mat = torch.sparse_csr_tensor(indptr, indices, data)
    x_torch = torch.sparse.sampled_addmm(csr_mat, u, v.T)
    res_data = x_torch.values().cpu().numpy()
    return res_data

def sddmm_numba_correctness(matrix, u, v):
    indptr_d = cuda.to_device(matrix.indptr.astype(np.int32))
    indices_d = cuda.to_device(matrix.indices.astype(np.int32))
    data_d = cuda.to_device(matrix.data.astype(np.float32))
    u_d = cuda.to_device(u)
    v_d = cuda.to_device(v)
    res_data = np.zeros(matrix.data.size, dtype=np.float32)
    res_data_d = cuda.to_device(res_data)
    M = matrix.shape[0]
    N = matrix.shape[1]
    K = u.shape[1]
    gridDimY = (M + MBLOCK_SIZE - 1) // MBLOCK_SIZE
    gridDimX = (K + KBLOCK_SIZE - 1) // KBLOCK_SIZE
    blockDimY = MBLOCK_SIZE
    blockDimX = KBLOCK_SIZE
    # print((gridDimX, gridDimY), (blockDimX, blockDimY))
    sddmm_kernel_CSR_u_stationary_par_reduce[(gridDimX, gridDimY), (blockDimX, blockDimY)](indptr_d, indices_d, data_d, u_d, v_d, M, N, K, res_data_d)
    res_data = res_data_d.copy_to_host()
    return res_data

def sddmm_numba_s_stationary_correctness(matrix, u, v):
    indptr_d = cuda.to_device(matrix.indptr.astype(np.int32))
    indices_d = cuda.to_device(matrix.indices.astype(np.int32))
    data_d = cuda.to_device(matrix.data.astype(np.float32))
    u_d = cuda.to_device(u)
    v_d = cuda.to_device(v)
    res_data = np.zeros(matrix.data.size, dtype=np.float32)
    res_data_d = cuda.to_device(res_data)
    M = matrix.shape[0]
    N = matrix.shape[1]
    K = u.shape[1]
    gridDimY = (M + S_STATIONARY_BLOCK_SIZE - 1) // S_STATIONARY_BLOCK_SIZE
    gridDimX = (K + KBLOCK_SIZE - 1) // KBLOCK_SIZE
    blockDimX = KBLOCK_SIZE
    # print((gridDimX, gridDimY), (blockDimX, blockDimY))
    sddmm_kernel_s_stationary[(gridDimX, gridDimY), blockDimX](indptr_d, indices_d, data_d, u_d, v_d, M, N, K, res_data_d)
    res_data = res_data_d.copy_to_host()
    return res_data

def sddmm_numba_a_stationary(indptr, indices, data, res_data, u, v, M, N, K):
    gridDimY = (M + MBLOCK_SIZE - 1) // MBLOCK_SIZE
    gridDimX = (K + KBLOCK_SIZE - 1) // KBLOCK_SIZE
    blockDimY = MBLOCK_SIZE
    blockDimX = KBLOCK_SIZE
    sddmm_kernel_CSR_u_stationary_par_reduce[(gridDimX, gridDimY), (blockDimX, blockDimY)](indptr, indices, data, u, v, M, N, K, res_data)

def sddmm_numba_s_stationary_coo_packed(indptr, indices, row_ind, data, res_data, u, v, M, N, K):
    gridDimY = (M + S_STATIONARY_BLOCK_SIZE - 1) // S_STATIONARY_BLOCK_SIZE
    gridDimX = (K + KBLOCK_SIZE - 1) // KBLOCK_SIZE
    blockDimY = MBLOCK_SIZE
    blockDimX = KBLOCK_SIZE
    sddmm_kernel_COO_packed_s_stationary[(gridDimX, gridDimY), (blockDimX, blockDimY)](indptr, indices, row_ind, data, u, v, M, N, K, res_data)


def sddmm_numba_s_stationary(indptr, indices, data, res_data, u, v, M, N, K):
    gridDimY = (M + S_STATIONARY_BLOCK_SIZE - 1) // S_STATIONARY_BLOCK_SIZE
    gridDimX = (K + KBLOCK_SIZE - 1) // KBLOCK_SIZE
    blockDimX = KBLOCK_SIZE
    sddmm_kernel_s_stationary[(gridDimX, gridDimY), blockDimX](indptr, indices, data, u, v, M, N, K, res_data)


def correctness_test():
    matrix_signtures = []
    method_correctness = {}
    mtx_list = get_matrix_list("/home/salehm32/projects/fused-gnn/fusion/data/SPD/spd_list.txt", "/home/salehm32/projects/fused-gnn/fusion/data/SPD/")

    for mtx_matrix_path in mtx_list:
        print(f"------------ matrix: {mtx_matrix_path}")
        matrix = scio.mmread(mtx_matrix_path)
        n = matrix.shape[0]
        A = matrix.tocsr()
        u = np.random.rand(n, 32).astype(np.float32)
        v = np.random.rand(n, 32).astype(np.float32)
        data = A.data.astype(np.float32)
        res_data_cpu = sddmm_cpu(A.indptr, A.indices, data, u, v, matrix.shape[0], matrix.shape[1], u.shape[1])
        res_data_torch = torch_sddmm_correctness(A, u, v)
        res_data_gpu = gpu_sddmm_correctness(A, u, v)
        res_data_numba = sddmm_numba_correctness(A, u, v)
        res_data_numba_s_stationary = sddmm_numba_s_stationary_correctness(A, u, v)


        print("Triton correctness test:")
        if np.allclose(res_data_cpu, res_data_gpu, atol=1e-6):
            print("Passed")
        else:
            print("Failed")
            print(res_data_cpu)
            print(res_data_gpu)
        print("-----------------")
        print("Torch correctness test:")
        if np.allclose(res_data_cpu, res_data_torch, atol=1e-6):
            print("Passed")
        else:
            print("Failed")
            print(res_data_cpu)
            print(res_data_torch)
        print("-----------------")
        print("Numba correctness test:")
        if np.allclose(res_data_cpu, res_data_numba, atol=1e-6):
            print("Passed")
        else:
            print("Failed")
            print(res_data_cpu)
            print(res_data_numba)
        print("-----------------")
        print("Numba s-stationary correctness test:")
        if np.allclose(res_data_cpu, res_data_numba_s_stationary, atol=1e-6):
            print("Passed")
        else:
            print("Failed")
            print(res_data_cpu)
            print(res_data_numba_s_stationary)



mtx_list = get_matrix_list("/home/salehm32/projects/fused-gnn/fusion/data/SPD/spd_list.txt", "/home/salehm32/projects/fused-gnn/fusion/data/SPD/")
method_list = ["triton", "torch", 'numba-u-stationary', 'numba-s-stationary-coo-packed', 'numba-s-stationary'] #TODO: Add numba gpu version and dgl implementation

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["matrices"],  # Argument names to use as an x-axis for the plot
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=method_list,  # Label name for the lines
        line_names=["triton", "torch", 'numba-u-stationary', 'numba-s-stationary-coo-packed', 'numba-s-stationary'],  # Name of the lines
        styles=[("green", "-"), ("blue", "-"), ("red", "-"), ("gold", "-"), ("purple", "-")],  # Visual styles for the lines
        ylabel="mili-seconds",  # Label name for the y-axis
        plot_name="gemm-spmm-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={}
    ))

@triton.testing.perf_report(configs)
def benchmark(matrices, provider):
    matrix = scio.mmread(matrices)
    A = matrix.tocsr()
    n = A.shape[0]
    m = A.shape[1]
    sp_data = A.data.astype(np.float32)
    quantiles = [0.5, 0.2, 0.8]
    rep, warmpup = 100, 25
    feat_dim = 32 #TODO: make this a parameter in configs
    u = np.random.rand(n, feat_dim).astype(np.float32)
    v = np.random.rand(m, feat_dim).astype(np.float32)
    if provider == 'triton':
        indptr = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        indices = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        data = torch.tensor(sp_data, dtype=torch.float32, device=DEVICE)
        u = torch.tensor(u, dtype=torch.float32, device=DEVICE)
        v = torch.tensor(v, dtype=torch.float32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sddmm_triton(indptr, indices, data, u, v, matrix.shape[0], matrix.shape[1], u.shape[1]), quantiles=quantiles, warmup=warmpup, rep=rep)
    # if provider == 'cpu':
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: sddmm_cpu(A.indptr, A.indices, sp_data, u, v, matrix.shape[0], matrix.shape[1], u.shape[1]), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'torch':
        indptr = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        indices = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        data = torch.tensor(A.data, dtype=torch.float32, device=DEVICE)
        u_torch = torch.tensor(u, dtype=torch.float32, device=DEVICE)
        v_torch = torch.tensor(v, dtype=torch.float32, device=DEVICE)
        csr_mat = torch.sparse_csr_tensor(indptr, indices, data)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.sparse.sampled_addmm(csr_mat, u_torch, v_torch.T), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'numba-u-stationary':
        indptr_d = cuda.to_device(A.indptr.astype(np.int32))
        indices_d = cuda.to_device(A.indices.astype(np.int32))
        data_d = cuda.to_device(A.data.astype(np.float32))
        u_d = cuda.to_device(u)
        v_d = cuda.to_device(v)
        res_data = np.zeros(A.data.size, dtype=np.float32)
        res_data_d = cuda.to_device(res_data)
        M = A.shape[0]
        N = A.shape[1]
        K = u.shape[1]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sddmm_numba_a_stationary(indptr_d, indices_d, data_d, res_data_d, u_d, v_d, M, N, K), quantiles=quantiles, warmup=warmpup, rep=rep)
        #TODO: calculate GFLOPS as perf
    if provider == 'numba-s-stationary-coo-packed':
        indptr_d = cuda.to_device(A.indptr.astype(np.int32))
        indices_d = cuda.to_device(A.indices.astype(np.int32))
        data_d = cuda.to_device(A.data.astype(np.float32))
        u_d = cuda.to_device(u)
        v_d = cuda.to_device(v)
        res_data = np.zeros(A.data.size, dtype=np.float32)
        res_data_d = cuda.to_device(res_data)
        row_ind = extract_row_indices(A)
        row_ind_d = cuda.to_device(row_ind)
        M = A.shape[0]
        N = A.shape[1]
        K = u.shape[1]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sddmm_numba_s_stationary_coo_packed(indptr_d, indices_d, row_ind_d, data_d, res_data_d, u_d, v_d, M, N, K), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'numba-s-stationary':
        indptr_d = cuda.to_device(A.indptr.astype(np.int32))
        indices_d = cuda.to_device(A.indices.astype(np.int32))
        data_d = cuda.to_device(A.data.astype(np.float32))
        u_d = cuda.to_device(u)
        v_d = cuda.to_device(v)
        res_data = np.zeros(A.data.size, dtype=np.float32)
        res_data_d = cuda.to_device(res_data)
        M = A.shape[0]
        N = A.shape[1]
        K = u.shape[1]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sddmm_numba_s_stationary(indptr_d, indices_d, data_d, res_data_d, u_d, v_d, M, N, K), quantiles=quantiles, warmup=warmpup, rep=rep)
    return ms, min_ms, max_ms

def extract_row_indices(matrix):
    nnz = matrix.data.size
    row_indices = np.zeros(nnz, dtype=np.int32)
    row_ptr = matrix.indptr
    for i in range(matrix.shape[0]):
        start = row_ptr[i]
        end = row_ptr[i+1]
        row_indices[start:end] = i
    return row_indices


if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=".", show_plots=True)

    # correctness_test()




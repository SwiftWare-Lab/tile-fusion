from numba import cuda
import numba
from utils import get_matrix_list
import scipy.io as scio
import numpy as np
from sddmm import sddmm_kernel_s_stationary
from spmm import spmm_numba_seqreduce_kernel
import triton


S_STATIONARY_BLOCK_SIZE = 128
KBLOCK_SIZE = 32
BCOL=32


##TODO: Add column dimension for b(separated than N which is column dimension of u and v)

## sample fused kernel using sddmm s stationary. Assumed that bcols in u_data, v_data and b_data are same.
@cuda.jit()
def sddmm_spmm_fused_kernel_atomic(
        row_ptr, col_ind, data_ptr,
        u_data, v_data,
        b_data,
        M, N, K,
        c_data
):
    SA = cuda.shared.array(shape=(S_STATIONARY_BLOCK_SIZE, KBLOCK_SIZE), dtype=numba.float32)
    rows_per_block = S_STATIONARY_BLOCK_SIZE
    start_row = cuda.blockIdx.y * rows_per_block
    end_row = min(M, start_row + rows_per_block)
    u_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    for i in range(start_row, end_row):
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
                res = 0
                col = col_ind[i]
                for j in range(K):
                    res += data_ptr[i] * SA[row - start_row][j] * v_data[col][j]

                for j in range(K):
                    cuda.atomic.add(c_data, (row, j), res * b_data[col][j])



##It doesn't work due to shared memory limitations(Dynamic was tested and it didn't work probably due to the size limitation)
@cuda.jit()
def sddmm_spmm_fused_kernel_intermediate_SA(
        row_ptr, col_ind, data_ptr,
        u_data, v_data,
        b_data,
        M, N, K,
        c_data
):
    SA = cuda.shared.array(shape=(S_STATIONARY_BLOCK_SIZE, KBLOCK_SIZE), dtype=numba.float32)
    SA_intermediate = cuda.shared.array(shape=N, dtype=numba.float32)
    rows_per_block = S_STATIONARY_BLOCK_SIZE
    start_row = cuda.blockIdx.y * rows_per_block
    end_row = min(M, start_row + rows_per_block)
    u_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    for i in range(start_row, end_row):
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
                SA_intermediate[col] = 0
                for j in range(K):
                    SA_intermediate[col] += data_ptr[i] * SA[row - start_row][j] * v_data[col][j]
        cuda.syncthreads()
        res = 0
        nnz_start = row_ptr[row]
        print(nnz_start)
        for i in range(nnz_start, nnz_end):
            col = col_ind[i]
            val = SA_intermediate[col]
            res += val * b_data[col][u_x]
        c_data[row][u_x] = res


@cuda.jit()
def sddmm_spmm_fused_kernel_intermediate_res(
        row_ptr, col_ind, data_ptr,
        u_data, v_data,
        b_data,
        M, N, K,
        res_data,
        c_data
):
    SA = cuda.shared.array(shape=(S_STATIONARY_BLOCK_SIZE, KBLOCK_SIZE), dtype=numba.float32)
    rows_per_block = S_STATIONARY_BLOCK_SIZE
    start_row = cuda.blockIdx.y * rows_per_block
    end_row = min(M, start_row + rows_per_block)
    u_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    for i in range(start_row, end_row):
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
                    res_data[i] += data_ptr[i] * SA[row - start_row][j] * v_data[col][j]
        cuda.syncthreads()
        res = 0
        nnz_start = row_ptr[row]
        for i in range(nnz_start, nnz_end):
            col = col_ind[i]
            val = res_data[i]
            res += val * b_data[col][u_x]
        c_data[row][u_x] = res

def sddmm_spmm_unfused(indptr, indices, data, u, v, res_data, b, c, M, N, K):
    sddmm_grid_dim_y = (M + S_STATIONARY_BLOCK_SIZE - 1) // S_STATIONARY_BLOCK_SIZE
    sddmm_grid_dim_x = (K + KBLOCK_SIZE - 1) // KBLOCK_SIZE
    sddmm_block_dim_x = KBLOCK_SIZE
    spmm_tpb = 128
    spmm_block_dim_x = min(K, spmm_tpb)
    spmm_block_dim_y = (spmm_tpb + spmm_block_dim_x - 1) // spmm_block_dim_x
    spmm_grid_dim_x = (M + spmm_block_dim_y - 1) // spmm_block_dim_y
    spmm_grid_dim_y = (K + spmm_block_dim_x - 1) // spmm_block_dim_x
    sddmm_kernel_s_stationary[(sddmm_grid_dim_x, sddmm_grid_dim_y), sddmm_block_dim_x](indptr, indices, data, u, v, M, N, K, res_data)
    spmm_numba_seqreduce_kernel[(spmm_grid_dim_x, spmm_grid_dim_y), (spmm_block_dim_x, spmm_block_dim_y)](res_data, indices, indptr, b, c, M, K, N)
    return c.copy_to_host()


def sddmm_spmm_fused_atomic(indptr, indices, data, u, v, b, c, M, N, K):
    gridDimY = (M + S_STATIONARY_BLOCK_SIZE - 1) // S_STATIONARY_BLOCK_SIZE
    gridDimX = (K + KBLOCK_SIZE - 1) // KBLOCK_SIZE
    blockDimX = KBLOCK_SIZE
    sddmm_spmm_fused_kernel_atomic[(gridDimX, gridDimY), blockDimX](indptr, indices, data, u, v, b, M, N, K, c)
    return c.copy_to_host()

def sddmm_spmm_fused_intermediate_SA(indptr, indices, data, u, v, b, c, M, N, K):
    gridDimY = (M + S_STATIONARY_BLOCK_SIZE - 1) // S_STATIONARY_BLOCK_SIZE
    gridDimX = (K + KBLOCK_SIZE - 1) // KBLOCK_SIZE
    blockDimX = KBLOCK_SIZE
    sddmm_spmm_fused_kernel_intermediate_SA[(gridDimX, gridDimY), blockDimX](indptr, indices, data, u, v, b, M, N, K, c)
    return c.copy_to_host()


def sddmm_spmm_fused_intermediate_res(indptr, indices, data, u, v, res_data, b, c, M, N, K):
    gridDimY = (M + S_STATIONARY_BLOCK_SIZE - 1) // S_STATIONARY_BLOCK_SIZE
    gridDimX = (K + KBLOCK_SIZE - 1) // KBLOCK_SIZE
    blockDimX = KBLOCK_SIZE
    sddmm_spmm_fused_kernel_intermediate_res[(gridDimX, gridDimY), blockDimX](indptr, indices, data, u, v, b, M, N, K, res_data, c)
    return c.copy_to_host()


def correctness_test():
    matrix_signtures = []
    method_correctness = {}
    mtx_list = get_matrix_list("/home/salehm32/projects/fused-gnn/fusion/data/SPD/spd_list.txt", "/home/salehm32/projects/fused-gnn/fusion/data/SPD/")

    for mtx_matrix_path in mtx_list:
        print(f"------------ matrix: {mtx_matrix_path}")
        matrix = scio.mmread(mtx_matrix_path)
        M = matrix.shape[0]
        N = matrix.shape[1]
        A = matrix.tocsr()
        K = 32
        u = np.random.rand(M, K).astype(np.float32)
        v = np.random.rand(M, K).astype(np.float32)
        b = np.random.rand(N, K).astype(np.float32)

        data = A.data.astype(np.float32)
        indices = A.indices.astype(np.int32)
        indptr = A.indptr.astype(np.int32)
        data_d = cuda.to_device(data)
        indices_d = cuda.to_device(indices)
        indptr_d = cuda.to_device(indptr)
        u_d = cuda.to_device(u)
        v_d = cuda.to_device(v)
        b_d = cuda.to_device(b)

        res_data = np.zeros(data.size, dtype=np.float32)
        c = np.zeros((M, K), dtype=np.float32)
        c_d = cuda.to_device(c)
        res_data_d = cuda.to_device(res_data)

        c1 = sddmm_spmm_unfused(indptr_d, indices_d, data_d, u_d, v_d, res_data_d, b_d, c_d, M, N, K)

        c = np.zeros((M, K), dtype=np.float32)
        c_d = cuda.to_device(c)
        c2 = sddmm_spmm_fused_atomic(indptr_d, indices_d, data_d, u_d, v_d, b_d, c_d, M, N, K)

        res_data = np.zeros(data.size, dtype=np.float32)
        c = np.zeros((M, K), dtype=np.float32)
        c_d = cuda.to_device(c)
        res_data_d = cuda.to_device(res_data)

        c3 = sddmm_spmm_fused_intermediate_res(indptr_d, indices_d, data_d, u_d, v_d, res_data_d, b_d, c_d, M, N, K)

        # c = np.zeros((M, K), dtype=np.float32)
        # c_d = cuda.to_device(c)
        # c4 = sddmm_spmm_fused_intermediate_SA(indptr_d, indices_d, data_d, u_d, v_d, b_d, c_d, M, N, K)

        print("Fused Atomic vs Unfused Correctness Test")
        if np.allclose(c1, c2, atol=1e-6):
            print("PASS")
        else:
            print("FAIL")
        print("----------------------------------")
        print("Fused Intermediate Res vs Unfused Correctness Test")
        if np.allclose(c1, c3, atol=1e-6):
            print("PASS")
        else:
            print("FAIL")
        print("----------------------------------")
        # print("Fused Intermediate SA vs Unfused Correctness Test")
        # if np.allclose(c1, c4, atol=1e-6):
        #     print("PASS")
        # else:
        #     print("FAIL")


mtx_list = get_matrix_list("/home/salehm32/projects/fused-gnn/fusion/data/SPD/spd_list.txt", "/home/salehm32/projects/fused-gnn/fusion/data/SPD/")
method_list = ["unfused-sddmm-spmm", "fused-sddmm-spmm-atomic", "fused-sddmm-spmm-intermediate-res"] #TODO: Add numba gpu version and dgl implementation
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["matrices"],  # Argument names to use as an x-axis for the plot
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=method_list,  # Label name for the lines
        line_names=["unfused-sddmm-spmm", "fused-sddmm-spmm-atomic", "fused-sddmm-spmm-intermediate-res"],  # Name of the lines
        styles=[("green", "-"), ("blue", "-"), ("red", "-"), ("gold", "-"), ("purple", "-")],  # Visual styles for the lines
        ylabel="GFLOP/S",  # Label name for the y-axis
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
    b = np.random.rand(n, feat_dim).astype(np.float32)
        #TODO: calculate GFLOPS as perf
    if provider == 'unfused-sddmm-spmm':
        indptr_d = cuda.to_device(A.indptr.astype(np.int32))
        indices_d = cuda.to_device(A.indices.astype(np.int32))
        data_d = cuda.to_device(A.data.astype(np.float32))
        u_d = cuda.to_device(u)
        v_d = cuda.to_device(v)
        b_d = cuda.to_device(b)
        res_data = np.zeros(A.data.size, dtype=np.float32)
        res_data_d = cuda.to_device(res_data)
        c = np.zeros((m, feat_dim), dtype=np.float32)
        c_d = cuda.to_device(c)
        M = A.shape[0]
        N = A.shape[1]
        K = u.shape[1]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sddmm_spmm_unfused(indptr_d, indices_d, data_d, u_d, v_d, res_data_d, b_d, c_d, M, N, K), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'fused-sddmm-spmm-atomic':
        indptr_d = cuda.to_device(A.indptr.astype(np.int32))
        indices_d = cuda.to_device(A.indices.astype(np.int32))
        data_d = cuda.to_device(A.data.astype(np.float32))
        u_d = cuda.to_device(u)
        v_d = cuda.to_device(v)
        b_d = cuda.to_device(b)
        c = np.zeros((m, feat_dim), dtype=np.float32)
        c_d = cuda.to_device(c)
        M = A.shape[0]
        N = A.shape[1]
        K = u.shape[1]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sddmm_spmm_fused_atomic(indptr_d, indices_d, data_d, u_d, v_d, b_d, c_d, M, N, K), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'fused-sddmm-spmm-intermediate-res':
        indptr_d = cuda.to_device(A.indptr.astype(np.int32))
        indices_d = cuda.to_device(A.indices.astype(np.int32))
        data_d = cuda.to_device(A.data.astype(np.float32))
        u_d = cuda.to_device(u)
        v_d = cuda.to_device(v)
        b_d = cuda.to_device(b)
        c = np.zeros((m, feat_dim), dtype=np.float32)
        c_d = cuda.to_device(c)
        res_data = np.zeros(A.data.size, dtype=np.float32)
        res_data_d = cuda.to_device(res_data)
        M = A.shape[0]
        N = A.shape[1]
        K = u.shape[1]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sddmm_spmm_fused_intermediate_res(indptr_d, indices_d, data_d, u_d, v_d, res_data_d, b_d, c_d, M, N, K), quantiles=quantiles, warmup=warmpup, rep=rep)
    return ms, min_ms, max_ms


if __name__ == "__main__":
    # correctness_test()
    benchmark.run(print_data=True, save_path=".", show_plots=False)
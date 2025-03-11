//
// Created by salehm32 on 30/01/25.
//

#ifndef SPARSE_FUSION_GEMM_KERNELS_CUH
#define SPARSE_FUSION_GEMM_KERNELS_CUH
#include <cuda.h>


template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_X = 0U,
          size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_from_global_memory_to_shared_memory(
    T const* A, size_t lda, T const* B, size_t ldb,
    T A_thread_block_tile[BLOCK_TILE_SIZE_Y]
                         [BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
    T B_thread_block_tile[BLOCK_TILE_SIZE_K]
                         [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n,
    size_t k)
{
  // Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
  for (size_t load_idx{0U};
       load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
                      NUM_THREADS;
       ++load_idx)
  {
    size_t const A_thread_block_tile_row_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
    size_t const A_thread_block_tile_col_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
    size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                           A_thread_block_tile_row_idx};
    size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                           A_thread_block_tile_col_idx};

    // These boundary checks might slow down the kernel to some extent.
    // But they guarantee the correctness of the kernel for all
    // different GEMM configurations.
    T val{static_cast<T>(0)};
    if (A_row_idx < m && A_col_idx < k)
    {
      val = A[A_row_idx * lda + A_col_idx];
    }
    // This if will slow down the kernel.
    // Add static asserts from the host code to guarantee this if is
    // always true.
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS ==
                  0U);
    // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
    //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
    // {
    //     A_thread_block_tile[A_thread_block_tile_row_idx]
    //                        [A_thread_block_tile_col_idx] = val;
    // }
    A_thread_block_tile[A_thread_block_tile_row_idx]
                       [A_thread_block_tile_col_idx] = val;
  }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
  for (size_t load_idx{0U};
       load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
                      NUM_THREADS;
       ++load_idx)
  {
    size_t const B_thread_block_tile_row_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X};
    size_t const B_thread_block_tile_col_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X};
    size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                           B_thread_block_tile_row_idx};
    size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                           B_thread_block_tile_col_idx};

    // These boundary checks might slow down the kernel to some extent.
    // But they guarantee the correctness of the kernel for all
    // different GEMM configurations.
    T val{static_cast<T>(0)};
    if (B_row_idx < k && B_col_idx < n)
    {
      val = B[B_row_idx * ldb + B_col_idx];
    }
    // This if will slow down the kernel.
    // Add static asserts from the host code to guarantee this if is
    // always true.
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS ==
                  0U);
    // if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
    //     B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
    // {
    //     B_thread_block_tile[B_thread_block_tile_row_idx]
    //                        [B_thread_block_tile_col_idx] = val;
    // }
    B_thread_block_tile[B_thread_block_tile_row_idx]
                       [B_thread_block_tile_col_idx] = val;
  }
}


//TODO: implement this.
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_X = 0U,
          size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_single_tile_data_from_global_memory_to_shared_memory(
    T const* B, size_t ldb,
    T B_thread_block_tile[BLOCK_TILE_SIZE_K]
                         [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n,
    size_t k)
{
  // Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
  for (size_t load_idx{0U};
       load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
                      NUM_THREADS;
       ++load_idx)
  {
    size_t const B_thread_block_tile_row_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X};
    size_t const B_thread_block_tile_col_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X};
    size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                           B_thread_block_tile_row_idx};
    size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                           B_thread_block_tile_col_idx};

    // These boundary checks might slow down the kernel to some extent.
    // But they guarantee the correctness of the kernel for all
    // different GEMM configurations.
    T val{static_cast<T>(0)};
    if (B_row_idx < k && B_col_idx < n)
    {
      val = B[B_row_idx * ldb + B_col_idx];
    }
    // This if will slow down the kernel.
    // Add static asserts from the host code to guarantee this if is
    // always true.
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS ==
                  0U);
    // if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
    //     B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
    // {
    //     B_thread_block_tile[B_thread_block_tile_row_idx]
    //                        [B_thread_block_tile_col_idx] = val;
    // }
    B_thread_block_tile[B_thread_block_tile_row_idx]
                       [B_thread_block_tile_col_idx] = val;
  }
}

template <typename T, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_columnwise_single_tile_data_from_global_memory_to_shared_memory(
    T const* A, size_t lda,
    T A_thread_block_tile[BLOCK_TILE_SIZE_Y]
                         [BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m,
    size_t k)
{
  // Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
  for (size_t load_idx{0U};
       load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
                      NUM_THREADS;
       ++load_idx)
  {
    size_t const A_thread_block_tile_row_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
    size_t const A_thread_block_tile_col_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
    size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                           A_thread_block_tile_row_idx};
    size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                           A_thread_block_tile_col_idx};

    // These boundary checks might slow down the kernel to some extent.
    // But they guarantee the correctness of the kernel for all
    // different GEMM configurations.
    T val{static_cast<T>(0)};
    if (A_row_idx < m && A_col_idx < k)
    {
      val = A[A_row_idx * lda + A_col_idx];
    }
    // This if will slow down the kernel.
    // Add static asserts from the host code to guarantee this if is
    // always true.
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS ==
                  0U);
    // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
    //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
    // {
    //     A_thread_block_tile[A_thread_block_tile_row_idx]
    //                        [A_thread_block_tile_col_idx] = val;
    // }
    A_thread_block_tile[A_thread_block_tile_row_idx]
                       [A_thread_block_tile_col_idx] = val;
  }
}

template <typename T, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_thread_block_tile_from_global_memory_to_shared_memory(
    T const* A, size_t lda,
    T A_thread_block_tile[BLOCK_TILE_SIZE_Y]
                         [BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
    size_t m, size_t k)
{
  // Load data from A on DRAM to A_thread_block_tile on shared memory.
    size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                           threadIdx.y};
    size_t const A_col_idx{blockIdx.x * BLOCK_TILE_SIZE_K +
                           threadIdx.x};

    // These boundary checks might slow down the kernel to some extent.
    // But they guarantee the correctness of the kernel for all
    // different GEMM configurations.
    T val{static_cast<T>(0)};
    if (A_row_idx < m && A_col_idx < k)
    {
      val = A[A_row_idx * lda + A_col_idx];
    }
    // This if will slow down the kernel.
    // Add static asserts from the host code to guarantee this if is
    // always true.
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS ==
                  0U);
    // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
    //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
    // {
    //     A_thread_block_tile[A_thread_block_tile_row_idx]
    //                        [A_thread_block_tile_col_idx] = val;
    // }
    A_thread_block_tile[threadIdx.y]
                       [threadIdx.x] = val;
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm2DBlocking(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
  // Avoid using blockDim.x * blockDim.y as the number of threads per block.
  // Because it is a runtime constant and the compiler cannot optimize the
  // loop unrolling based on that.
  // Use a compile time constant instead.
  constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
  size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

  // Compute the row and column of C that this thread is responsible for.
  size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
  size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};
  // Cache a tile of A and B in shared memory for data reuse.
  __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                      BLOCK_TILE_SIZE_K};

  T sum{static_cast<T>(0)};
  for (size_t thread_block_tile_idx{0U};
       thread_block_tile_idx < num_thread_block_tiles;
       ++thread_block_tile_idx)
  {
    load_data_from_global_memory_to_shared_memory<
        T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
        NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile,
                     B_thread_block_tile, thread_block_tile_idx,
                     thread_linear_idx, m, n, k);
#pragma unroll
    for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    {
      // Doing this results in 2 TOPS.
      // Suppose blockDim.x = blockDim.y = 32.
      // Effectively, for a warp, in one iteration, we read the value from
      // A_thread_block_tile at the same location on the shared memory
      // resulting in a broadcast, we also read 32 values that have no
      // bank conflicts from B_thread_block_tile. Even with that, all the
      // values have to be read from the shared memory and consequence is
      // the shared memory instruction runs very intensively just to
      // compute a small number of values using simple arithmetic
      // instructions, which is not efficient.
      sum += A_thread_block_tile[threadIdx.y][k_i] *
             B_thread_block_tile[k_i][threadIdx.x];
    }
    __syncthreads();
  }
  if (C_row_idx < m && C_col_idx < n)
  {
    C[C_row_idx * ldc + C_col_idx] =
        alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
  }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemmAStationary2DBlocking(size_t m, size_t n, size_t k, T alpha, T const* A,
                               size_t lda, T const* B, size_t ldb, T beta, T* C,
                               size_t ldc)
{
  // Avoid using blockDim.x * blockDim.y as the number of threads per block.
  // Because it is a runtime constant and the compiler cannot optimize the
  // loop unrolling based on that.
  // Use a compile time constant instead.
  constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
  size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

  // Cache a tile of A and B in shared memory for data reuse.
  __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  size_t const num_thread_block_tiles{(n + BLOCK_TILE_SIZE_X - 1) /
                                      BLOCK_TILE_SIZE_X};

  load_thread_block_tile_from_global_memory_to_shared_memory<float,
                                                             BLOCK_TILE_SIZE_Y,
                                                             BLOCK_TILE_SIZE_K,
                                                             NUM_THREADS>(
      A, lda, A_thread_block_tile, m, k);


  for (size_t thread_block_tile_idx{0U};
       thread_block_tile_idx < num_thread_block_tiles;
       ++thread_block_tile_idx)
  {
    load_columnwise_single_tile_data_from_global_memory_to_shared_memory<
        T, BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_X,
        NUM_THREADS>( B, ldc,
                     B_thread_block_tile, thread_block_tile_idx,
                     thread_linear_idx, k, n);
    T sum{static_cast<T>(0)};
#pragma unroll
    for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    {
      // Doing this results in 2 TOPS.
      // Suppose blockDim.x = blockDim.y = 32.
      // Effectively, for a warp, in one iteration, we read the value from
      // A_thread_block_tile at the same location on the shared memory
      // resulting in a broadcast, we also read 32 values that have no
      // bank conflicts from B_thread_block_tile. Even with that, all the
      // values have to be read from the shared memory and consequence is
      // the shared memory instruction runs very intensively just to
      // compute a small number of values using simple arithmetic
      // instructions, which is not efficient.
      sum += A_thread_block_tile[threadIdx.y][k_i] *
             B_thread_block_tile[k_i][threadIdx.x];
    }
    int C_row_idx = blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.y;
    int C_col_idx = thread_block_tile_idx * BLOCK_TILE_SIZE_X + threadIdx.x;

    if (C_row_idx < m && C_col_idx < n)
    {
      atomicAdd(&C[C_row_idx * ldc + C_col_idx], alpha * sum);
    }
    __syncthreads();
  }

}


template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm2DBlockingSpMMSeqRedFused(size_t m, size_t n, size_t k,
                                              const int *Ap, const int *Ai, const T* Ax,
                                              const int FPtr[], const int FId[],
                                              T alpha, T const*Bx, size_t ldb,
                                              T const* Cx, size_t ldc, T beta,
                                              T* BCx,T* Xx)
{
  constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
  size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

  // Compute the row and column of Cx that this thread is responsible for.
  size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
  size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};
  // Cache a tile of A and Bx in shared memory for data reuse.
  __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  __shared__ T C_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                      BLOCK_TILE_SIZE_K};

  T sum{static_cast<T>(0)};
  for (size_t thread_block_tile_idx{0U};
       thread_block_tile_idx < num_thread_block_tiles;
       ++thread_block_tile_idx)
  {
    load_data_from_global_memory_to_shared_memory<
        T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
        NUM_THREADS>(Bx, ldb, Cx, ldc, B_thread_block_tile,
                     C_thread_block_tile, thread_block_tile_idx,
                     thread_linear_idx, m, n, k);
#pragma unroll
    for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    {
      sum += B_thread_block_tile[threadIdx.y][k_i] *
             C_thread_block_tile[k_i][threadIdx.x];
    }
    __syncthreads();
  }
  if (C_row_idx < m && C_col_idx < n)
  {
    BCx[C_row_idx * ldc + C_col_idx] =
        alpha * sum + beta * BCx[C_row_idx * ldc + C_col_idx];
  }
  __syncthreads();
  if (C_col_idx < n){
    T res = 0, val;
    int col;
    int rowTileId = blockIdx.y;
    int firstInd = __ldg(&FPtr[rowTileId]);
    int lastInd = __ldg(&FPtr[rowTileId + 1]);
    int rowInd = firstInd + threadIdx.y;
    if (rowInd < lastInd){
      int row = __ldg(&FId[rowInd]);
      res = 0;
      int start = __ldg(&Ap[rowInd]);
      int end = __ldg(&Ap[rowInd + 1]);
      for (int i = start; i < end; ++i){
        col = __ldg(&Ai[i]);
        val = Ax[i];
        res += val * BCx[col * ldc + C_col_idx];
      }
      if (C_col_idx == 0){
      }
      Xx[row * ldc + C_col_idx] = res;
    }
  }
}



template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void fusedSpMM1DGeMM2DBlocking(size_t m, size_t n, size_t k,
                                   const int *Ap, const int *Ai, const T* Ax,
                                   T alpha, T const*Bx, size_t ldb,
                                   T const* Cx, size_t ldc, T beta,T* ABx,
                                   T* Xx)
{

  constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
  size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

  // Compute the row and column of C that this thread is responsible for.
  size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
  size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};
  // Cache a tile of A and B in shared memory for data reuse.
//  __shared__ T ABx_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  __shared__ T C_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                      BLOCK_TILE_SIZE_K};

  /// SPMM
  int stride = NUM_THREADS / ldb;
  int threadId = threadIdx.y * blockDim.x + threadIdx.x;
  int subwarpId = threadId / ldb;
  int rowStart = blockIdx.y * BLOCK_TILE_SIZE_Y + subwarpId;
  int rowEnd = min(rowStart + BLOCK_TILE_SIZE_Y, m);
  int aBxCol = threadId % ldb;
  if (aBxCol < n){
    for (int row=rowStart; row < rowEnd; row+=stride) {
      int start = Ap[row];
      int end = Ap[row + 1];
      T res = 0;
      for (int i = start; i < end; ++i) {
        int col = Ai[i];
        T val = Ax[i];
        res += val * Bx[col * ldb + aBxCol];
      }
      ABx[row * ldb + aBxCol] = res;
    }
  }

  /// SPMM

  T sum{static_cast<T>(0)};
  for (size_t thread_block_tile_idx{0U};
       thread_block_tile_idx < num_thread_block_tiles;
       ++thread_block_tile_idx)
  {
    //TODO: change this to only load one tile.
    load_single_tile_data_from_global_memory_to_shared_memory<
        T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
        NUM_THREADS>( Cx, ldc,
                     C_thread_block_tile, thread_block_tile_idx,
                     thread_linear_idx, m, n, k);
#pragma unroll
    for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    {
      aBxCol = thread_block_tile_idx * BLOCK_TILE_SIZE_K + k_i;
      int aBxRow = blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.y;
      sum += ABx[aBxRow * ldb + aBxCol] *
             C_thread_block_tile[k_i][threadIdx.x];
    }
    __syncthreads();
  }
  if (C_row_idx < m && C_col_idx < n)
  {
    Xx[C_row_idx * ldc + C_col_idx] =
        alpha * sum + beta * Xx[C_row_idx * ldc + C_col_idx];
  }
}


template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void fusedSpMM2DGeMM2DBlocking(size_t m, size_t n, size_t k,
                                        const int *Ap, const int *Ai, const T* Ax,
                                        T alpha, T const*Bx, size_t ldb,
                                        T const* Cx, size_t ldc, T beta,T* ABx,
                                        T* Xx)
{

  constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
  size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

  // Compute the row and column of C that this thread is responsible for.
  size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
  size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};
  // Cache a tile of A and B in shared memory for data reuse.
  __shared__ T AB_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  __shared__ T C_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                      BLOCK_TILE_SIZE_K};

  T sum{static_cast<T>(0)};
  for (size_t thread_block_tile_idx{0U};
       thread_block_tile_idx < num_thread_block_tiles;
       ++thread_block_tile_idx)
  {
    /// SPMM
    int row = blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.y;
    int bCol = thread_block_tile_idx*BLOCK_TILE_SIZE_K + threadIdx.x;
    if (bCol < n && row < m){
      int start = Ap[row];
      int end = Ap[row + 1];
      T res = 0;
      for (int i = start; i < end; ++i) {
        int col = Ai[i];
        T val = Ax[i];
        res += val * Bx[col * ldb + bCol];
      }
      //TODO: use unrolling here on K_i instead of using threadIdx.x(doesn't work on cases that TILE_X !=TILE_K)
      AB_thread_block_tile[threadIdx.y][threadIdx.x] = res;
    }

    /// SPMM

    load_single_tile_data_from_global_memory_to_shared_memory<
        T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
        NUM_THREADS>( Cx, ldc,
                     C_thread_block_tile, thread_block_tile_idx,
                     thread_linear_idx, m, n, k);
#pragma unroll
    for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    {

      sum += AB_thread_block_tile[threadIdx.y][k_i] *
             C_thread_block_tile[k_i][threadIdx.x];
    }
    __syncthreads();
  }
  if (C_row_idx < m && C_col_idx < n)
  {
    Xx[C_row_idx * ldc + C_col_idx] =
        alpha * sum + beta * Xx[C_row_idx * ldc + C_col_idx];
  }
}


template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void fusedSpMM1DSMGeMM2DBlocking(size_t m, size_t n, size_t k,
                                          const int *Ap, const int *Ai, const T* Ax,
                                          T alpha, T const*Bx, size_t ldb,
                                          T const* Cx, size_t ldc, T beta,T* ABx,
                                          T* Xx)
{

  // Avoid using blockDim.x * blockDim.y as the number of threads per block.
  // Because it is a runtime constant and the compiler cannot optimize the
  // loop unrolling based on that.
  // Use a compile time constant instead.
  constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
  constexpr size_t AB_BLOCK_SIZE{BLOCK_TILE_SIZE_Y*128};
  size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

  // Compute the row and column of C that this thread is responsible for.
  size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
  size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};
  // Cache a tile of A and B in shared memory for data reuse.
  //  __shared__ T ABx_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];

  __shared__ T AB_thread_block_tile[AB_BLOCK_SIZE];
  __shared__ T C_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                      BLOCK_TILE_SIZE_K};

  /// SPMM
  int stride = NUM_THREADS / ldb;
  int threadId = threadIdx.y * blockDim.x + threadIdx.x;
  int subwarpId = threadId / ldb;
  int rowStart = blockIdx.y * BLOCK_TILE_SIZE_Y + subwarpId;
  int rowEnd = min(rowStart + BLOCK_TILE_SIZE_Y, m);
  int aBxCol = threadId % ldb;
  if (aBxCol < n){
    for (int row=rowStart; row < rowEnd; row+=stride) {
      int start = Ap[row];
      int end = Ap[row + 1];
      T res = 0;
      for (int i = start; i < end; ++i) {
        int col = Ai[i];
        T val = Ax[i];
        res += val * Bx[col * ldb + aBxCol];
      }
      int blockRow = row % BLOCK_TILE_SIZE_Y;
      AB_thread_block_tile[blockRow * ldb + aBxCol] = res;
    }
  }

  /// SPMM

  T sum{static_cast<T>(0)};
  for (size_t thread_block_tile_idx{0U};
       thread_block_tile_idx < num_thread_block_tiles;
       ++thread_block_tile_idx)
  {
    //TODO: change this to only load one tile.
    load_single_tile_data_from_global_memory_to_shared_memory<
        T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
        NUM_THREADS>( Cx, ldc,
                     C_thread_block_tile, thread_block_tile_idx,
                     thread_linear_idx, m, n, k);
#pragma unroll
    for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    {
      aBxCol = thread_block_tile_idx * BLOCK_TILE_SIZE_K + k_i;
      sum += AB_thread_block_tile[threadIdx.y * ldb + aBxCol] *
             C_thread_block_tile[k_i][threadIdx.x];
    }
    __syncthreads();
  }
  if (C_row_idx < m && C_col_idx < n)
  {
    Xx[C_row_idx * ldc + C_col_idx] =
        alpha * sum + beta * Xx[C_row_idx * ldc + C_col_idx];
  }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void fusedSpMM2DGeMMAStationary(size_t m, size_t n, size_t k,
                                           const int *Ap, const int *Ai, const T* Ax,
                                           T alpha, T const*Bx, size_t ldb,
                                           T const* Cx, size_t ldc, T beta,T* ABx,
                                           T* Xx){
  constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
  size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

  // Compute the row and column of C that this thread is responsible for.
  size_t const AB_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
  size_t const AB_row_idx{blockIdx.y * blockDim.y + threadIdx.y};
  // Cache a tile of A and B in shared memory for data reuse.
  __shared__ T AB_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  __shared__ T C_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

  size_t const num_thread_block_tiles{(n + BLOCK_TILE_SIZE_X - 1) /
                                      BLOCK_TILE_SIZE_X};

  //TODO: SpMM outside of the for loop.
  /// SPMM

  if (AB_col_idx < n && AB_row_idx < m){
    int start = Ap[AB_row_idx];
    int end = Ap[AB_row_idx + 1];
    T res = 0;
    for (int i = start; i < end; ++i) {
      int col = Ai[i];
      T val = Ax[i];
      res += val * Bx[col * ldb + AB_col_idx];
    }
    AB_thread_block_tile[threadIdx.y][threadIdx.x] = res;
  }

  /// SPMM


  for (size_t thread_block_tile_idx{0U};
       thread_block_tile_idx < num_thread_block_tiles;
       ++thread_block_tile_idx)
  {

    T sum{static_cast<T>(0)};
    //TODO: This part needs change to iterate over columns of C and Xx
    load_columnwise_single_tile_data_from_global_memory_to_shared_memory<
        T, BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_X,
        NUM_THREADS>( Cx, ldc,
                     C_thread_block_tile, thread_block_tile_idx,
                     thread_linear_idx, k, n);
#pragma unroll
    for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    {

      sum += AB_thread_block_tile[threadIdx.y][k_i] *
             C_thread_block_tile[k_i][threadIdx.x];
    }
    __syncthreads();
    int C_row_idx = blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.y;
    int C_col_idx = thread_block_tile_idx * BLOCK_TILE_SIZE_X + threadIdx.x;
    if (C_row_idx < m && C_col_idx < n){
      atomicAdd(&Xx[C_row_idx * ldc + C_col_idx], alpha * sum);
    }
  }
}

#endif // SPARSE_FUSION_GEMM_KERNELS_CUH

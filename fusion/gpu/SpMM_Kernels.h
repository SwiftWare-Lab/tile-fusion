//
// Created by salehm32 on 18/06/24.
//
#include <aggregation/def.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cusparse.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

__device__ __forceinline__ float sum_init() { return 0.0; }

// Ge-SpMM

__global__ void topoSimpleSPMMKernel(int m, int k, const int *A_indptr,
                                     const int *A_indices, const float *B,
                                     float *C) {
  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid + 1)];
    float acc1 = sum_init();
    int offset;
    for (int ptr = lb; ptr < hb; ptr++) {
      offset = A_indices[ptr] * k + threadIdx.x;
      acc1 = sum_reduce(acc1, B[offset]);
    }
    C[(rid * k + threadIdx.x)] = acc1;
  }
}

__global__ void topoCacheSPMMKernel(int m, int k, const int *A_indptr,
                                    const int *A_indices, const float *A_value,
                                    const float *B, float *C) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y << 5); // << 5 is equal to *32 ->> col
  int thread_idx = sm_offset + threadIdx.x;

  int cid = (blockIdx.y << 5) + threadIdx.x;
  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  int value_off = blockDim.y * blockDim.x;

  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid + 1)];
    int offset;
    int ptr = lb + threadIdx.x;
    float acc1 = sum_init();
    if (blockIdx.y != gridDim.y - 1) {
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          reinterpret_cast<float *>(sh)[thread_idx + value_off] = A_value[ptr];
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[sm_offset + kk] + cid;
          acc1 = sum_reduce(acc1, reinterpret_cast<float *>(
                                      sh)[(sm_offset + kk + value_off)] *
                                      B[offset]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
        }
        __syncwarp();
      }
      offset = rid * k + cid;
      C[offset] = acc1;
    } else { // threadIdx.y==blockDim.y-1
      int nout = (k - cid + 31) / 32;
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          reinterpret_cast<float *>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[(sm_offset + kk)] + cid;
          if (nout > 0) {
            acc1 = sum_reduce(acc1, reinterpret_cast<float *>(
                                        sh)[(sm_offset + kk + value_off)] *
                                        B[offset]);
          }
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
        }
        __syncwarp();
      }
      offset = rid * k + cid;
      if (nout > 0) {
        C[offset] = acc1;
      }
    }
  }
}

__global__ void topoCacheCoarsenSPMMKernel(int m, int k, const int *A_indptr,
                                           const int *A_indices,
                                           const float *A_value, const float *B,
                                           float *C) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y << 5);
  int thread_idx = sm_offset + threadIdx.x;
  int value_off = blockDim.y * blockDim.x;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < m) {

    int cid = (blockIdx.y << 6) + threadIdx.x;
    int lb = A_indptr[rid];
    int hb = A_indptr[rid + 1];
    int ptr = lb + threadIdx.x;
    int offset;
    float acc1 = sum_init();
    float acc2 = sum_init();
    if (blockIdx.y != gridDim.y - 1) {
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          reinterpret_cast<float *>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[(sm_offset + kk)] + cid;
          acc1 = sum_reduce(acc1, reinterpret_cast<float *>(
                                      sh)[(sm_offset + kk + value_off)] *
                                      B[offset]);
          acc2 = sum_reduce(acc2, reinterpret_cast<float *>(
                                      sh)[(sm_offset + kk + value_off)] *
                                      B[(offset + 32)]);
        }
        __syncwarp();
      }
      offset = rid * k + cid;
      C[offset] = acc1;
      C[offset + 32] = acc2;
    } else { // threadIdx.y==blockDim.y-1
      int nout = (k - cid + 31) / 32;
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          reinterpret_cast<float *>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[(sm_offset + kk)] + cid;
          if (nout > 0) {
            acc1 = sum_reduce(acc1, reinterpret_cast<float *>(
                                        sh)[(sm_offset + kk + value_off)] *
                                        B[offset]);
          }
          if (nout > 1) {
            acc2 = sum_reduce(acc2, reinterpret_cast<float *>(
                                        sh)[(sm_offset + kk + value_off)] *
                                        B[(offset + 32)]);
          }
        }
        __syncwarp();
      }
      offset = rid * k + cid;
      if (nout > 0) {
        C[offset] = acc1;
      }
      if (nout > 1) {
        C[offset + 32] = acc2;
      }
    }
  }
}

// TODO: it has bugs(severe bugs when N<32).
void csrGeSpMM(const int M, const int N, const int K, const int *Ap,
               const int *Ai, const float *Ax, const float *Bx, float *Cx) {
  const int *rowindA_csr = Ap;
  const int *colindA = Ai;
  if (N < 32) {
    const int row_per_block = 128 / N;
    const int n_block = (M + row_per_block - 1) / row_per_block;
    topoSimpleSPMMKernel<<<dim3(n_block, 1, 1), dim3(N, row_per_block, 1)>>>(
        M, N, rowindA_csr, colindA, Bx, Cx);
  } else if (N < 64) {
    const int tile_k = (N + 31) / 32;
    const int n_block = (M + 3) / 4;
    topoCacheSPMMKernel<<<dim3(n_block, tile_k, 1), dim3(32, 4, 1),
                          256 * sizeof(int)>>>(M, N, rowindA_csr, colindA, Ax,
                                               Bx, Cx);
  } else {
    const int tile_k = (N + 63) / 64;
    const int n_block = (M + 8 - 1) / 8;
    topoCacheCoarsenSPMMKernel<<<dim3(n_block, tile_k, 1), dim3(32, 8, 1),
                                 2 * 8 * 32 * sizeof(int)>>>(
        M, N, rowindA_csr, colindA, Ax, Bx, Cx);
  }
}

// csrspmm impls from dgSparse

// file: csrspmm_parreduce.cuh
//      Implementation of parallel reduction kernels

// Parallel-reduction algorithm assigns a warp to a non-zero segment
//   and use primitives like parallel-reduction / parallel-scan
//   to compute SpMM.

const int RefThreadPerBlock = 256;
#define CEIL(x, y) (((x) + (y)-1) / (y))

#define FULLMASK 0xffffffff
#define DIV_UP(x, y) (((x) + (y)-1) / (y))
#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a < b) ? b : a)
#define SHFL_DOWN_REDUCE(v)                                                    \
  v += __shfl_down_sync(FULLMASK, v, 16);                                      \
  v += __shfl_down_sync(FULLMASK, v, 8);                                       \
  v += __shfl_down_sync(FULLMASK, v, 4);                                       \
  v += __shfl_down_sync(FULLMASK, v, 2);                                       \
  v += __shfl_down_sync(FULLMASK, v, 1);
#define SEG_SHFL_SCAN(v, tmpv, segid, tmps)                                    \
  tmpv = __shfl_down_sync(FULLMASK, v, 1);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 1);                                 \
  if (tmps == segid && lane_id < 31)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 2);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 2);                                 \
  if (tmps == segid && lane_id < 30)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 4);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 4);                                 \
  if (tmps == segid && lane_id < 28)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 8);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 8);                                 \
  if (tmps == segid && lane_id < 24)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 16);                                    \
  tmps = __shfl_down_sync(FULLMASK, segid, 16);                                \
  if (tmps == segid && lane_id < 16)                                           \
    v += tmpv;

template <typename T>
__device__ __forceinline__ T __guard_load_default_one(const T *base,
                                                      int offset) {
  if (base != nullptr)
    return base[offset];
  else
    return static_cast<T>(1);
}

template <typename index_t>
__device__ __forceinline__ index_t
binary_search_segment_number(const index_t *seg_offsets, const index_t n_seg,
                             const index_t n_elem, const index_t elem_id) {
  index_t lo = 1, hi = n_seg, mid;
  while (lo < hi) {
    mid = (lo + hi) >> 1;
    if (seg_offsets[mid] <= elem_id) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (hi - 1);
}

template <typename access_t>
__global__ void csrspmm_parreduce_rowbalance_kernel(
    const int M, const int N, const int K, const int csr_indptr[],
    const int csr_indices[], const float csr_data[], const float B[],
    float C[]) {
  constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);

  int lane_id = (threadIdx.x & (32 - 1));
  int stride = gridDim.x * blockDim.y;
  int row = blockIdx.x * blockDim.y + threadIdx.y;

  // get the dense column offset
  int col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
  const float *B_panel = B + col_offset;
  float *C_panel = C + col_offset;
  int ldB = N;
  int ldC = N;

  if (col_offset >= N)
    return;
  if (col_offset + CoarsenFactor >= N)
    goto Ndim_Residue;

  for (; row < M; row += stride) {
    // declare accumulators
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor];

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int jj = start + lane_id; jj < end; jj += 32) {
      k = csr_indices[jj];
      v = __guard_load_default_one<float>(csr_data, jj);

      // load B-elements in vector-type
      *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * buffer[i];
      }
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      // row-wise reduction is a simple merge-tree
      SHFL_DOWN_REDUCE(c[i])
    }

    // store to C in vector-type
    if (lane_id == 0) {
      *(access_t *)(C_panel + row * ldC) = *(access_t *)c;
    }
  }
  return;

Ndim_Residue:
  int valid_lane_num = N - col_offset;

  for (; row < M; row += stride) {
    // get row offsets
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor];
    // access_t res = init_zeros<access_t>();

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int jj = start + lane_id; jj < end; jj += 32) {
      k = csr_indices[jj];
      v = __guard_load_default_one<float>(csr_data, jj);

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          buffer[i] = B_panel[k * ldB + i];
        }
      }

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * buffer[i];
      }
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      SHFL_DOWN_REDUCE(c[i])
    }

    if (lane_id == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          C_panel[row * ldC + i] = c[i];
        }
      }
    }
  }
}

template <typename access_t>
__global__ void csrspmm_parreduce_nnzbalance_kernel(
    const int M, const int N, const int K, const int nnz_,
    const int csr_indptr[], const int csr_indices[], const float csr_data[],
    const float B[], float C[]) {
  constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);
  int nnz = nnz_;
  if (nnz < 0)
    nnz = csr_indptr[M];

  int lane_id = (threadIdx.x & (32 - 1));
  int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  int nz_start = Nnzdim_warp_id * 32;
  int stride = gridDim.x * (blockDim.y * 32);

  // get the dense column offset
  int col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
  const float *B_panel = B + col_offset;
  float *C_panel = C + col_offset;
  int ldB = N;
  int ldC = N;

  int k;
  float v;
  float c[CoarsenFactor] = {0};
  float buffer[CoarsenFactor] = {0};

  if (col_offset >= N)
    return;
  if (col_offset + CoarsenFactor >= N)
    goto Ndim_Residue;

  for (int nz_id = nz_start + lane_id;
       nz_id < nnz + lane_id; // make sure NO warp loop-divergence
       nz_id += stride) {
    int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);

    if (nz_id < nnz) {
      k = csr_indices[nz_id];
      v = __guard_load_default_one<float>(csr_data, nz_id);
    } else {
      k = 0;
      v = 0.0f;
    }

    // load B-elements in vector-type
    *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      c[i] = buffer[i] * v;
    }

    // reduction
    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
// if all non-zeros in this warp belong to the same row, use a simple reduction
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    } else {
      // if non-zeros belong to different rows, use a parallel-scan primitive
      // thread that holds the start of each segment are responsible for writing
      // results
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
// atomic add has no vector-type form.
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    }
  }
  return;
Ndim_Residue:
  int valid_lane_num = N - col_offset;

  for (int nz_id = nz_start + lane_id;
       nz_id < nnz + lane_id; // make sure NO warp loop-divergence
       nz_id += stride) {
    int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);

    if (nz_id < nnz) {
      k = csr_indices[nz_id];
      v = __guard_load_default_one<float>(csr_data, nz_id);
    } else {
      k = 0;
      v = 0.0f;
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        c[i] = B_panel[k * ldB + i] * v;
      }
    }

    // reduction
    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    } else {
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    }
  }
  return;
}

void csrspmm_parreduce_rowbalance(const int M, const int N, const int K,
                                  const int *Ap, const int *Ai, const float *Ax,
                                  const float *B, float *C) {
  // factor of thread coarsening
  int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
  // number of parallel warps along M-dimension
  int Mdim_worker = M;
  // partition large-N and map to blockdim.y to help cache performance
  int Ndim_threadblock = CEIL(N, 32);
  int Ndim_warp_per_tb = MIN(N, 32) / coarsen_factor;

  int ref_warp_per_tb = RefThreadPerBlock / 32;
  int Mdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

  // total number of warps
  int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
  int gridDimY = Ndim_threadblock;
  dim3 gridDim(gridDimX, gridDimY, 1);
  dim3 blockDim(Ndim_warp_per_tb * 32, Mdim_warp_per_tb, 1);

  if (coarsen_factor == 4) {
    csrspmm_parreduce_rowbalance_kernel<float4>
        <<<gridDim, blockDim>>>(M, N, K, Ap, Ai, Ax, B, C);
  } else if (coarsen_factor == 2) {
    csrspmm_parreduce_rowbalance_kernel<float2>
        <<<gridDim, blockDim>>>(M, N, K, Ap, Ai, Ax, B, C);
  } else {
    csrspmm_parreduce_rowbalance_kernel<float>
        <<<gridDim, blockDim>>>(M, N, K, Ap, Ai, Ax, B, C);
  }
}

void csrspmm_parreduce_nnzbalance(const int M, const int N, const int K,
                                  const int NNZ, const int *Ap, const int *Ai,
                                  const float *Ax, const float *B, float *C) {
  // factor of thread coarsening
  int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
  // number of parallel warps along M-dimension
  const int segreduce_size_per_warp = 32;
  int Nnzdim_worker = M; // CEIL(spmatA.nnz, segreduce_size_per_warp);
  // partition large-NThreads and map to blockdim.y to help cache performance
  int Ndim_threadblock = CEIL(N, 32);
  int Ndim_warp_per_tb = MIN(N, 32) / coarsen_factor;

  int ref_warp_per_tb = RefThreadPerBlock / 32;
  int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

  // total number of warps
  int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
  int gridDimY = Ndim_threadblock;
  dim3 gridDim(gridDimX, gridDimY, 1);
  dim3 blockDim(Ndim_warp_per_tb * 32, Nnzdim_warp_per_tb, 1);

  if (coarsen_factor == 4) {
    csrspmm_parreduce_nnzbalance_kernel<float4>
        <<<gridDim, blockDim>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
  } else if (coarsen_factor == 2) {
    csrspmm_parreduce_nnzbalance_kernel<float2>
        <<<gridDim, blockDim>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
  } else {
    csrspmm_parreduce_nnzbalance_kernel<float>
        <<<gridDim, blockDim>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
  }
}

// Row-caching strategy pre-loads sparse elements into shared memory
// bucket-by-bucket and share the buffered sparse values within the same warp.
// The __syncwarp() primitive is used to assure shared-memory race safety.

template <int CoarsenFactor>
__global__ void csrspmm_rowcaching_rowbalance_kernel(
    const int M, const int N, const int K, const int csr_indptr[],
    const int csr_indices[], const float csr_data[], const float B[],
    float C[]) {
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;

  extern __shared__ int shared_mem[];
  int *workspace_indices = &shared_mem[(warp_id << 5)];
  float *workspace_data =
      (float *)(workspace_indices +
                blockDim.x); // float and int has the same size

  // get the sparse-value range of this row
  int row_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
  if (row_id >= M)
    return;
  int start = csr_indptr[row_id];
  int end = csr_indptr[row_id + 1];

  // get the dense column offset
  int col_offset = blockIdx.y * 32 * CoarsenFactor;
  const float *B_lanes[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_lanes[i] = B + col_offset + lane_id + i * 32;
  }
  int ldB = N;

  // declare accumulators
  float c[CoarsenFactor] = {0.0f};
  int ldC = N;

  // N-dimension residual handling
  if (blockIdx.y == gridDim.y - 1)
    goto Ndim_Residue;

  // iterate over the sparse row
  for (int p = start; p < end; p += 32) {
    // copy a bucket of sparse row elements into shared memory
    if (p + lane_id < end) {
      workspace_data[lane_id] =
          __guard_load_default_one<float>(csr_data, (p + lane_id));
      workspace_indices[lane_id] = csr_indices[p + lane_id];
    } else {
      workspace_data[lane_id] = 0.0f;
      workspace_indices[lane_id] = 0;
    }
    __syncwarp();
// do MAC computation using buffered elements
#pragma unroll
    for (int pp = 0; pp < 32; pp++) {
      int k = workspace_indices[pp];
      float v = workspace_data[pp];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * B_lanes[i][k * ldB];
      }
    }
  }

// write results
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    float *C_lane = B_lanes[i] - B + (C + row_id * ldC);
    *C_lane = c[i];
  }
  return;

Ndim_Residue:
  int valid_lane_num = CEIL(N - col_offset - lane_id, 32);

  // iterate over the sparse row
  for (int p = start; p < end; p += 32) {
    // copy a bucket of sparse row elements into shared memory
    if (p + lane_id < end) {
      workspace_data[lane_id] =
          __guard_load_default_one<float>(csr_data, (p + lane_id));
      workspace_indices[lane_id] = csr_indices[p + lane_id];
    } else {
      workspace_data[lane_id] = 0.0f;
      workspace_indices[lane_id] = 0;
    }
    __syncwarp();
// do MAC computation using buffered elements
#pragma unroll
    for (int pp = 0; pp < 32; pp++) {
      int k = workspace_indices[pp];
      float v = workspace_data[pp];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          c[i] += v * B_lanes[i][k * ldB];
        }
      }
    }
  }

// write results
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    float *C_lane = B_lanes[i] - B + (C + row_id * ldC);
    if (i < valid_lane_num) {
      *C_lane = c[i];
    }
  }
  return;
}

template <int CoarsenFactor, int ThreadNz>
__global__ void csrspmm_rowcaching_nnzbalance_kernel(
    const int M, const int N, const int K, const int nnz_,
    const int csr_indptr[], const int csr_indices[], const float csr_data[],
    const float B[], float C[]) {
  int nnz = nnz_;
  if (nnz < 0)
    nnz = csr_indptr[M];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;

  extern __shared__ int shared_mem[];
  int *workspace_rowid = &shared_mem[(warp_id << 5)];
  int *workspace_colid = workspace_rowid + blockDim.x;
  float *workspace_data =
      (float *)(workspace_colid +
                blockDim.x); // float and int has the same size

  // get the sparse-value range of this row
  int global_warp_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
  int nz_start = global_warp_id * (ThreadNz * 32);

  // get the dense column offset
  int col_offset = blockIdx.y * 32 * CoarsenFactor;
  const float *B_lanes[CoarsenFactor];
  float *C_lanes[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_lanes[i] = B + col_offset + lane_id + i * 32;
    C_lanes[i] = C + col_offset + lane_id + i * 32;
  }
  int ldB = N;

  // declare accumulators
  float c[CoarsenFactor] = {0.0f};
  int ldC = N;

  int stride = gridDim.x * (blockDim.x >> 5) * ThreadNz * 32;

  if (blockIdx.y == gridDim.y - 1)
    goto Ndim_Residue;

  for (; nz_start < nnz; nz_start += stride) {
    // iterate over the segment of this warp
    for (int tile_base = nz_start;
         tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {
      int thread_nz_id = tile_base + lane_id;
      if (thread_nz_id < nnz) {
        workspace_colid[lane_id] = csr_indices[thread_nz_id];
        workspace_data[lane_id] =
            __guard_load_default_one<float>(csr_data, thread_nz_id);
      } else {
        workspace_colid[lane_id] = 0;
        workspace_data[lane_id] = 0.0f;
      }
      workspace_rowid[lane_id] =
          binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
      __syncwarp();

      // initialize with first value
      int k = workspace_colid[0];
      float v = workspace_data[0];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] = v * B_lanes[i][k * ldB];
      }
      int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
      for (int pp = 1; pp < 32; pp++) {
        next_row = workspace_rowid[pp];
        if (next_row != row_curr) {
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
          }
          row_curr = next_row;
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            c[i] = v * B_lanes[i][k * ldB];
          }
        } else {
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            c[i] = c[i] + v * B_lanes[i][k * ldB];
          }
        }
      }
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
      }
    }
  }
  return;

Ndim_Residue:

  int valid_lane_num = CEIL(N - col_offset - lane_id, 32);

  for (; nz_start < nnz; nz_start += stride) {
    // iterate over the segment of this warp
    for (int tile_base = nz_start;
         tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {
      int thread_nz_id = tile_base + lane_id;
      if (thread_nz_id < nnz) {
        workspace_colid[lane_id] = csr_indices[thread_nz_id];
        workspace_data[lane_id] =
            __guard_load_default_one<float>(csr_data, thread_nz_id);
      } else {
        workspace_colid[lane_id] = 0;
        workspace_data[lane_id] = 0.0f;
      }
      workspace_rowid[lane_id] =
          binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
      __syncwarp();

      // initialize with first value
      int k = workspace_colid[0];
      float v = workspace_data[0];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          c[i] = v * B_lanes[i][k * ldB];
        }
      }
      int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
      for (int pp = 1; pp < 32; pp++) {
        next_row = workspace_rowid[pp];
        if (next_row != row_curr) {
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
            }
          }
          row_curr = next_row;
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              c[i] = v * B_lanes[i][k * ldB];
            }
          }
        } else {
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              c[i] = c[i] + v * B_lanes[i][k * ldB];
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
        }
      }
    }
  }
}

void csrspmm_rowcaching_rowbalance(const int M, const int N, const int K,
                                   const int *Ap, const int *Ai,
                                   const float *Ax, const float *B, float *C) {
  int coarsen_factor = (N >= 512) ? 4 : (N >= 128) ? 2 : 1;
  int Ndim_threadblock = CEIL(N, (32 * coarsen_factor));
  int Mdim_warp_per_tb = RefThreadPerBlock / 32;
  dim3 gridDim(CEIL(M, Mdim_warp_per_tb), Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  size_t smem_size = (sizeof(int) + sizeof(float)) * RefThreadPerBlock;

  if (coarsen_factor == 4) {
    csrspmm_rowcaching_rowbalance_kernel<4>
        <<<gridDim, blockDim, smem_size>>>(M, N, K, Ap, Ai, Ax, B, C);
  } else if (coarsen_factor == 2) {
    csrspmm_rowcaching_rowbalance_kernel<2>
        <<<gridDim, blockDim, smem_size>>>(M, N, K, Ap, Ai, Ax, B, C);
  } else {
    csrspmm_rowcaching_rowbalance_kernel<1>
        <<<gridDim, blockDim, smem_size>>>(M, N, K, Ap, Ai, Ax, B, C);
  }
}

void csrspmm_rowcaching_nnzbalance(const int M, const int N, const int K,
                                   const int NNZ, const int *Ap, const int *Ai,
                                   const float *Ax, const float *B, float *C) {
  int coarsen_factor = (N >= 512) ? 4 : (N >= 128) ? 2 : 1;
  int Ndim_threadblock = CEIL(N, (32 * coarsen_factor));

  // int thread_nz = (spmatA.nnz > 8000 * 128 * 2) ? 2 : 1;
  int thread_nz = 1;
  int Nnzdim_warp_per_tb = RefThreadPerBlock / 32;
  // int Nnzdim_threadblock = CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 *
  // thread_nz );
  int Nnzdim_threadblock = CEIL(
      M,
      Nnzdim_warp_per_tb * thread_nz); // CEIL(spmatA.nnz, Nnzdim_warp_per_tb *
                                       // 32 * thread_nz );

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  size_t smem_size = (2 * sizeof(int) + sizeof(float)) * RefThreadPerBlock;

  // simple heuristic

  if (coarsen_factor == 4) {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<4, 1>
          <<<gridDim, blockDim, smem_size>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<4, 2>
          <<<gridDim, blockDim, smem_size>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<4, 4>
          <<<gridDim, blockDim, smem_size>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
  } else if (coarsen_factor == 2) {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<2, 1>
          <<<gridDim, blockDim, smem_size>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<2, 2>
          <<<gridDim, blockDim, smem_size>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<2, 4>
          <<<gridDim, blockDim, smem_size>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
  } else {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<1, 1>
          <<<gridDim, blockDim, smem_size>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<1, 2>
          <<<gridDim, blockDim, smem_size>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<1, 4>
          <<<gridDim, blockDim, smem_size>>>(M, N, K, NNZ, Ap, Ai, Ax, B, C);
  }
}

// Sequential-reduction algorithm assigns a thread to an output element
// Each thread performs a simple inner-product.
__global__ void
csrspmm_seqreduce_rowbalance_kernel(const int nr, const int nv, const int nc,
                                    const int rowPtr[], const int colIdx[],
                                    const float values[], const float dnInput[],
                                    float dnOutput[]) {
  int row_tile = blockDim.y;
  int subwarp_id = threadIdx.y;
  int stride = row_tile * gridDim.x;
  int row = blockIdx.x * row_tile + subwarp_id;
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  if (v_id < nv) {
    dnInput += v_id;
    dnOutput += v_id;

    float res = 0, val;
    int col;
    for (; row < nr; row += stride) {
      int start = __ldg(rowPtr + row);
      int end = __ldg(rowPtr + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(colIdx + p);
        val = __guard_load_default_one<float>(values, p);
        res += val * __ldg(dnInput + col * nv);
      }
      dnOutput[row * nv] = res;
    }
  }
}

__global__ void csrspmm_seqreduce_rowcoarsened_kernel(
    const int nr, const int nv, const int nc, const int RowPerThread,
    const int rowPtr[], const int colIdx[], const float values[],
    const float dnInput[], float dnOutput[]) {
  int row_tile = blockDim.y * RowPerThread;
  int subwarp_id = threadIdx.y;
//  int stride = row_tile * gridDim.x;
  int row_start = blockIdx.x * row_tile + subwarp_id * RowPerThread;
  int row_end = min(row_start + RowPerThread, nr);
//  printf("row: %d, row_end: %d\n", row, row_end);
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  if (v_id < nv) {
    dnInput += v_id;
    dnOutput += v_id;

    float res, val;
    int col;
    for (int row = row_start; row < row_end; row += 1) {
      res=0;
      int start = __ldg(rowPtr + row);
      int end = __ldg(rowPtr + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(colIdx + p);
        val = __guard_load_default_one<float>(values, p);
        res += val * __ldg(dnInput + col * nv);
      }
      dnOutput[row * nv] = res;
    }
  }
}

__global__ void csr_fusedTile_spmmspmm_seqreduce_rowbalance_kernel(
    const int M, const int N, const int K, const int Ap[], const int Ai[],
    const float Ax[], const float Bx[], float ACx[], float Xx[],
    const int FPtr[], const int FId[]) {
  int row_tile = blockDim.y;
  int subwarp_id = threadIdx.y;
  int stride = row_tile * gridDim.x;
  int row = blockIdx.x * row_tile + subwarp_id;
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  if (v_id < N) {
    Bx += v_id;
    float* aCxTemp = ACx + v_id;
    float res = 0, val;
    int col;
    for (; row < M; row += stride) {
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * __ldg(Bx + col * N);
      }
      aCxTemp[row * N] = res;
    }
  }
  __syncthreads();
  if (v_id < N) {
    Xx += v_id;
    float* aCxTemp = ACx + v_id;
    float res = 0, val;
    int col;
    int rowTileId = blockIdx.x;
    int firstInd = __ldg(FPtr + rowTileId);
    int lastInd = __ldg(FPtr + rowTileId + 1);
    int rowInd = firstInd + threadIdx.y;
    if (rowInd < lastInd) {
      row = __ldg(FId + rowInd);
      res = 0;
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * aCxTemp[col * N];
      }
      Xx[row * N] = res;
    }
  }
}


//TODO: Remove ACx and use shared memory instead -> needs the l1MaxTileSize value to configure the shared memory.
__global__ void csr_redundantFusedTile_multiplerow_seqreduce_rowbalance_kernel(
    const int M, const int N, const int K, const int RowTileSize, const int Ap[],
    const int Ai[], const float Ax[], const float Bx[], float ACx[], float Xx[], const int L1Ptr[],
    const int L1Id[]){
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  int stride = blockDim.y;
  if(v_id < N){
    Bx += v_id;
    float *aCxTemp = ACx + v_id;
    int tileId = blockIdx.x;
    int rowIndStart = threadIdx.y;
    int tileIndStart = __ldg(L1Ptr + tileId);
    int tileIndEnd = __ldg(L1Ptr + tileId + 1);
    int rowInd = tileIndStart + rowIndStart;
    float res, val;
    int col;
    for (; rowInd < tileIndEnd; rowInd+=stride) {
      int row = __ldg(L1Id + rowInd);
      res = 0;
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * __ldg(Bx + col * N);
      }
      aCxTemp[row * N] = res;
    }
  }
  __syncthreads();
  if(v_id < N){
    int subRowId = threadIdx.y;
    int rowStart = blockIdx.x * RowTileSize + subRowId;
    Xx += v_id;
    float* aCxTemp = ACx + v_id;
    float res, val;
    int col;
    int rowEnd = rowStart + RowTileSize;
    for (int row=rowStart; row < rowEnd; row+=stride){
      res = 0;
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * aCxTemp[col * N];
      }
      Xx[row * N] = res;
    }
  }
}


__global__ void csr_fusedTile_multiplerow_seqreduce_rowbalance_kernel(
    const int M, const int N, const int K, const int RowPerThread, const int Ap[], const int Ai[],
    const float Ax[], const float Bx[], float ACx[], float Xx[],
    const int FPtr[], const int FId[]) {
  int row_tile = blockDim.y * RowPerThread;
  int sub_row_id = threadIdx.y;
  int row_start = blockIdx.x * row_tile + sub_row_id * RowPerThread;
  int row_end = min(row_start + RowPerThread, M);
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  if (v_id < N) {
    Bx += v_id;
    float *aCxTemp = ACx + v_id;
    float res, val;
    int col;
    for (int row = row_start; row < row_end; row += 1) {
        res = 0;
        int start = __ldg(Ap + row);
        int end = __ldg(Ap + row + 1);
        for (int p = start; p < end; p++) {
          col = __ldg(Ai + p);
          val = __guard_load_default_one<float>(Ax, p);
          res += val * __ldg(Bx + col * N);
        }
        aCxTemp[row * N] = res;
    }
  }
    __syncthreads();
  if (v_id < N) {
    Xx += v_id;
    float* aCxTemp = ACx + v_id;
    float res = 0, val;
    int col;
    int rowTileId = blockIdx.x;
    int firstInd = __ldg(FPtr + rowTileId);
    int lastInd = __ldg(FPtr + rowTileId + 1);
    int fusedNum = lastInd - firstInd;
    int stride = blockDim.y;
    int rowInd = firstInd + threadIdx.y;
    for (; rowInd < lastInd; rowInd+=stride) {
      int row = __ldg(FId + rowInd);
      res = 0;
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * aCxTemp[col * N];
      }
      Xx[row * N] = res;
    }
  }
}

__global__ void csr_2LfusedTile_multiplerow_seqreduce_rowbalance_kernel(
    const int M, const int N, const int K, const int RowPerThread, const int Ap[], const int Ai[],
    const float Ax[], const float Bx[], float ACx[], float Xx[],
    const int FPtr[], const int FId[]) {
  int row_tile = blockDim.y * RowPerThread;
  int sub_row_id = threadIdx.y;
  int row_start = blockIdx.x * row_tile + sub_row_id * RowPerThread;
//  int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
  int row_end = min(row_start + RowPerThread, M);
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  int l1TilesNum = blockDim.y;
  if (v_id < N) {
    Bx += v_id;
    float *aCxTemp = ACx + v_id;
    float *xxTemp = Xx + v_id;
    float res, val;
    int col;
    for (int row = row_start; row < row_end; row += 1) {
      res = 0;
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * __ldg(Bx + col * N);
      }
      aCxTemp[row * N] = res;
    }
    int rowTileId = blockIdx.x;
    int fOffset = rowTileId *  (l1TilesNum + 1) + sub_row_id;
    int firstInd = __ldg(FPtr + fOffset);
    int lastInd = __ldg(FPtr + fOffset + 1);
    int rowInd = firstInd;
    for (; rowInd < lastInd; rowInd+=1){
      res = 0;
      int row = __ldg(FId + rowInd);
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * aCxTemp[col * N];
      }
      xxTemp[row * N] = res;
    }
  }
  __syncthreads();
  if (v_id < N) {
    float* aCxTemp = ACx + v_id;
    float *xxTemp = Xx + v_id;
    float res = 0, val;
    int col;
    int rowTileId = blockIdx.x;
    int fOffset = rowTileId * (l1TilesNum + 1) + l1TilesNum;
    int firstInd = __ldg(FPtr + fOffset);
    int lastInd = __ldg(FPtr + fOffset  + 1);
    int fusedNum = lastInd - firstInd;
    int stride = blockDim.y;
    int rowInd = firstInd + threadIdx.y;
    for (; rowInd < lastInd; rowInd+=stride) {
      int row = __ldg(FId + rowInd);
      res = 0;
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * aCxTemp[col * N];
      }
      xxTemp[row * N] = res;
    }
  }
}

__global__ void csr_fusedTile_multiplerow_fusedParReduce_rowbalance_kernel(
    const int M, const int N, const int K, const int RowPerThread, const int Ap[], const int Ai[],
    const float Ax[], const float Bx[], float ACx[], float Xx[],
    const int FAp[], const int FAi[], const float FAx[]) {
  int row_tile = blockDim.y * RowPerThread;
  int sub_row_id = threadIdx.y;
  int row_start = blockIdx.x * row_tile + sub_row_id * RowPerThread;
  int row_end = min(row_start + RowPerThread, M);
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  if (v_id < N) {
    Bx += v_id;
    float *aCxTemp = ACx + v_id;
    float *xxTemp = Xx + v_id;
    float res, val;
    int col;
    for (int row = row_start; row < row_end; row += 1) {
      res = 0;
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * __ldg(Bx + col * N);
      }
      int startF = __ldg(FAp + row);
      int endF = __ldg(FAp + row + 1);
      for (int p = startF; p < endF; p++) {
        int rowF = __ldg(FAi + p);
        val = __guard_load_default_one<float>(FAx, p);
        int resF = val * res;
//        xxTemp[rowF * N] += resF;
        atomicAdd_block(xxTemp + rowF * N, resF);
      }
      aCxTemp[row * N] = res;
    }
  }
}


//TODO: test this kernel.
__global__ void csr_fusedTile_multiplecol_seqreduce_rowbalance_kernel(
    const int M, const int N, const int K, const int ColPerThread, const int Ap[], const int Ai[],
    const float Ax[], const float Bx[], float ACx[], float Xx[],
    const int FPtr[], const int FId[]) {
  int sub_row_id = threadIdx.y;
  int row = blockIdx.x * blockDim.y + sub_row_id;
  int v_id_s = (blockIdx.y * blockDim.x * ColPerThread) + threadIdx.x * ColPerThread;

  if (v_id_s < N) {
    int v_id_e = min(v_id_s + ColPerThread, N);
    float val;
    int col;
    int start = __ldg(Ap + row);
    int end = __ldg(Ap + row + 1);
    for (int p = start; p < end; p++) {
      col = __ldg(Ai + p);
      val = __guard_load_default_one<float>(Ax, p);
      for (int v_id = v_id_s; v_id < v_id_e; v_id++) {
        ACx[row * N + v_id] += val * __ldg(Bx + v_id + col * N);
      }
    }
  }
  __syncthreads();
  int rowTileId = blockIdx.x;
  int firstInd = __ldg(FPtr + rowTileId);
  int lastInd = __ldg(FPtr + rowTileId + 1);
  int rowInd = firstInd + threadIdx.y;
  int fusedNum = lastInd - firstInd;
  int fusedColPerThread = ceil(float(N) / ((float(blockDim.x * blockDim.y) / fusedNum)));
  v_id_s = (blockIdx.y * blockDim.x * fusedColPerThread) + threadIdx.x * fusedColPerThread;
  if (v_id_s < N && rowInd < lastInd) {

    int v_id_e = min(v_id_s + fusedColPerThread, N);
    float val;
    int col;
    int row = __ldg(FId + rowInd);
    int start = __ldg(Ap + row);
    int end = __ldg(Ap + row + 1);
    for (int p = start; p < end; p++) {
      val = __guard_load_default_one<float>(Ax, p);
      col = __ldg(Ai + p);
      for (int v_id = v_id_s; v_id < v_id_e; v_id++) {
        Xx[row * N + v_id] += val * ACx[col * N + v_id];
      }
    }
  }
}


//TODO: Assigning all rows to one warp

__global__ void csr_fusedSynchTile_multiplerow_seqreduce_rowbalance_kernel(
    const int M, const int N, const int K, const int RowPerThread, const int Ap[], const int Ai[],
    const float Ax[], const float Bx[], float ACx[], float Xx[],
    const int FPtr[], const int FId[]) {
  int row_tile = blockDim.y * RowPerThread;
  int sub_row_id = threadIdx.y;
  int row_start = blockIdx.x * row_tile + sub_row_id * RowPerThread;
  int row_end = min(row_start + RowPerThread, M);
  int v_id = (blockIdx.y * blockDim.x * blockDim.z) + (threadIdx.z * blockDim.x) + threadIdx.x;
  if (v_id < N) {
    Bx += v_id;
    float *aCxTemp = ACx + v_id;
    float res, val;
    int col;
    for (int row = row_start; row < row_end; row += 1) {
      res = 0;
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * __ldg(Bx + col * N);
      }
      aCxTemp[row * N] = res;
    }
    Xx += v_id;
    int rowTileId = blockIdx.x;
    int firstInd = __ldg(FPtr + rowTileId);
    int lastInd = __ldg(FPtr + rowTileId + 1);
    int fusedNum = lastInd - firstInd;
    int stride = blockDim.y;
    int rowInd = firstInd + threadIdx.y;
    for (; rowInd < lastInd; rowInd += stride) {
      int row = __ldg(FId + rowInd);
      res = 0;
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * aCxTemp[col * N];
      }
      Xx[row * N] = res;
    }
  }
}

__global__ void csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel(
    const int UFDim, const int N, const int K, const int Ap[], const int Ai[],
    const float Ax[], const float ACx[], float Xx[], const int UFPtr[]) {
  int row_tile = blockDim.y;
  int subwarp_id = threadIdx.y;
  int stride = row_tile * gridDim.x;
  int rowInd = blockIdx.x * row_tile + subwarp_id;
  int rowTileId = blockIdx.y * blockDim.x;
  int v_id = (rowTileId) + threadIdx.x;
  if (v_id < N) {
    Xx += v_id;
    ACx += v_id;
    float res = 0, val;
    int col;
    if (rowInd < UFDim) {
      int row = __ldg(UFPtr + rowInd);
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * __ldg(ACx + col * N);
      }
      Xx[row * N] = res;
    }
  }
}

__global__ void csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel(
    const int UFDim, const int N, const int K, const int Ap[], const int Ai[],
    const float Ax[], const float ACx[], float Xx[], const int UFPtr[]) {
  int row_tile = blockDim.y;
  int subwarp_id = threadIdx.y;
  int stride = row_tile * gridDim.x;
  int rowInd = blockIdx.x * row_tile + subwarp_id;
  int rowTileId = blockIdx.y * blockDim.x;
  int v_id = (rowTileId) + threadIdx.x;
  if (v_id < N) {
    Xx += v_id;
    ACx += v_id;
    float res = 0, val;
    int col;
    if (rowInd < UFDim) {
      int row = __ldg(UFPtr + rowInd);
      int start = __ldg(Ap + rowInd);
      int end = __ldg(Ap + rowInd + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * __ldg(ACx + col * N);
      }
      Xx[row * N] = res;
    }
  }
}


// use shared memory for the shared data between two computations.
__global__ void csr_fusedTile_spmmspmm_seqreduce_rowbalance_sm_kernel(
    const int M, const int N, const int K, const int Ap[], const int Ai[],
    const float Ax[], const float Bx[], float ACx[], float Xx[],
    const int FPtr[], const int FId[]) {
  extern __shared__ float sharedMem[];
  int row_tile = blockDim.y;
  int col_tile = blockDim.x;
  int subwarp_id = threadIdx.y;
  int stride = row_tile * gridDim.x;
  int row = blockIdx.x * row_tile + subwarp_id;
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  if (v_id < N) {
    Bx += v_id;
    float* aCxTemp = ACx + v_id;
    float* sharedTemp = sharedMem + threadIdx.x;
    float res = 0, val;
    int col;
    for (; row < M; row += stride) {
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p);
        val = __guard_load_default_one<float>(Ax, p);
        res += val * __ldg(Bx + col * N);
      }
      sharedTemp[subwarp_id * col_tile] = res;
      aCxTemp[row * N] = res;
    }
  }
  __syncthreads();
  if (v_id < N) {
    Xx += v_id;
    float* aCxTemp = ACx + v_id;
    float* sharedTemp = sharedMem + threadIdx.x;
    float res = 0, val;
    int col;
    int rowTileId = blockIdx.x;
    int firstInd = __ldg(FPtr + rowTileId);
    int lastInd = __ldg(FPtr + rowTileId + 1);
    int rowInd = firstInd + threadIdx.y;
    if (rowInd < lastInd) {
      row = __ldg(FId + rowInd);
      res = 0;
      int start = __ldg(Ap + row);
      int end = __ldg(Ap + row + 1);
      for (int p = start; p < end; p++) {
        col = __ldg(Ai + p) % row_tile ;
        val = __guard_load_default_one<float>(Ax, p);
        res += val * sharedTemp[col * col_tile];
      }
      Xx[row * N] = res;
    }
  }
}

__global__ void
csrspmm_seqreduce_nnzbalance_kernel(const int nr, const int nv, const int nc,
                                    const int nnz_, const int rowPtr[],
                                    const int colIdx[], const float values[],
                                    const float dnInput[], float dnOutput[]) {
  int nnz = nnz_;
  if (nnz < 0)
    nnz = rowPtr[nr];

  int Nnzdim_thread = blockDim.y * gridDim.x;
  int NE_PER_THREAD = DIV_UP(nnz, Nnzdim_thread);
  int eid = (blockIdx.x * blockDim.y + threadIdx.y) * NE_PER_THREAD;
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  int col = 0;
  float val = 0.0;
  if (v_id < nv) {
    if (eid < nnz) {
      int row = binary_search_segment_number<int>(rowPtr, nr, nnz, eid);
      int step = __ldg(rowPtr + row + 1) - eid;

      for (int ii = 0; ii < NE_PER_THREAD; ii++) {
        if (eid >= nnz)
          break;
        if (ii < step) {
          col = __ldg(colIdx + eid) * nv;
          val += __guard_load_default_one<float>(values, eid) *
                 __ldg(dnInput + col + v_id);

          eid++;
        } else {
          atomicAdd(&dnOutput[row * nv + v_id], val);

          row = binary_search_segment_number<int>(rowPtr, nr, nnz, eid);
          step = __ldg(rowPtr + row + 1) - eid;
          col = __ldg(colIdx + eid) * nv;
          val = __guard_load_default_one<float>(values, eid) *
                __ldg(dnInput + col + v_id);

          eid++;
        }
      }
      atomicAdd(&dnOutput[row * nv + v_id], val);
    }
  }
}

void csrspmm_seqreduce_rowbalance(const int M, const int N, const int K,
                                  const int *Ap, const int *Ai, const float *Ax,
                                  const float *B, float *C) {
  int Mdim_worker = M;
  int Ndim_worker = N;
  int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
  int Ndim_thread_per_tb = MIN(Ndim_worker, RefThreadPerBlock);
  int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

  csrspmm_seqreduce_rowbalance_kernel<<<gridDim, blockDim>>>(M, N, K, Ap, Ai,
                                                             Ax, B, C);
}

void csrspmm_seqreduce_nnzbalance(const int M, const int N, const int K,
                                  const int NNZ, const int *Ap, const int *Ai,
                                  const float *Ax, const float *B, float *C) {
  // int Nnzdim_worker = spmatA.nnz;
  int Nnzdim_worker = M * 32;
  int Ndim_worker = N;
  int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
  int Ndim_thread_per_tb = MIN(Ndim_worker, RefThreadPerBlock);
  int Nnzdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  int Nnzdim_threadblock = CEIL(Nnzdim_worker, Nnzdim_thread_per_tb);

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Nnzdim_thread_per_tb, 1);

  csrspmm_seqreduce_nnzbalance_kernel<<<gridDim, blockDim>>>(M, N, K, NNZ, Ap,
                                                             Ai, Ax, B, C);
}

// HP-SpMM
//  TODO: Arch Specific. Need to find a pragma or a way to run it on a server.
//   error: Feature 'cp.async' requires .target sm_80 or higher
//
//__global__ void LBSPMMKernel(
//     int m, int k, int NNZ, int nnz_per_warp, const int* __restrict__
//     A_rowind, const int* __restrict__ A_colind, const  float* __restrict__
//     A_value, const  float* __restrict__ B,  float* __restrict__ C
//) {
//     extern __shared__ int sh[];
//     int sm_offset = (threadIdx.y<<5);
//     int thread_idx = sm_offset + threadIdx.x;
//     int cid = (blockIdx.y<<5)+threadIdx.x;
//     int off = blockDim.y * blockDim.x;
//     int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
//     int warp_start = warp_id * nnz_per_warp;
//     if (warp_start > NNZ - 1) {
//       return;
//     }
//     int former_row_id = __ldg(A_rowind + warp_start);
//     int current_rid = former_row_id;
//     int lb = warp_start;
//     int hb = warp_start + nnz_per_warp;
//     if (hb > NNZ){
//       hb = NNZ;
//     }
//     int offset;
//     int ptr = lb + threadIdx.x;
//     float acc1 = sum_init();
//     for(int i = nnz_per_warp; i > 0; i -= 32) {
//       if (ptr < hb) {
//       sh[thread_idx] = __ldg(A_rowind + ptr);
//       sh[thread_idx + off] = __ldg(A_colind + ptr)*k;
//       reinterpret_cast<float*>(sh)[thread_idx + off + off] =  __ldg(A_value +
//       ptr); } else { sh[thread_idx] = 0; sh[thread_idx + off] = 0;
//       sh[thread_idx + off + off] = 0;
//       }
//       __syncwarp();
// #pragma unroll
//       for (int kk=0; kk<32; kk++) {
//       current_rid = sh[sm_offset + kk];
//       if(current_rid != former_row_id) {
//         atomicAdd(&C[former_row_id*k + cid], acc1);
//         acc1 = sum_init();
//         former_row_id = current_rid;
//       }
//       offset = sh[sm_offset + kk + off] + cid;
//       acc1 = sum_reduce(acc1,
//       reinterpret_cast<float*>(sh)[(sm_offset+kk+off+off)] * __ldg(B +
//       offset));
//       }
//       ptr += 32;
//     }
//     __syncwarp();
//     atomicAdd(&C[current_rid*k + cid], acc1);
// }
//
// #define COPY_BYTES 16
// #def ine CP_ASYNC_CG(dst, src, Bytes) \
//    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst),
//     "l"(src), "n"(Bytes))
//
// #define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
// #define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n"
// ::"n"(N))
//
//__global__ void LBSPMMKernel_4_8_float4_float4_async_double_buffer(
//     int m, int k, int NNZ, int nnz_per_warp, const int* __restrict__
//     A_rowind, const int* __restrict__ A_colind, const  float* __restrict__
//     A_value, const  float* __restrict__ B,  float* __restrict__ C
//) {
//     __shared__ int row[2][256];
//     __shared__ int col[2][256];
//     __shared__ float val[2][256];
//     int sm_offset = (threadIdx.y<<5);
//     int thread_sh_idx = sm_offset + (threadIdx.x << 2);
//     int cid = (blockIdx.y<<5)+(threadIdx.x<<2);
//     int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
//     int warp_start = warp_id * nnz_per_warp;
//     if (warp_start > NNZ - 1) {
//       return;
//     }
//     int former_row_id = __ldg(A_rowind + warp_start);
//     int current_rid = former_row_id;
//     int lb = warp_start;
//     int hb = warp_start + nnz_per_warp;
//     if (hb > NNZ){
//       hb = NNZ;
//     }
//     int offset;
//     int ptr = lb + (threadIdx.x<<2);
//     float acc1 = sum_init();
//     float acc2 = sum_init();
//     float acc3 = sum_init();
//     float acc4 = sum_init();
//     uint32_t row_smem_addr = __cvta_generic_to_shared(row[0] +
//     thread_sh_idx); uint32_t col_smem_addr = __cvta_generic_to_shared(col[0]
//     + thread_sh_idx); uint32_t val_smem_addr =
//     __cvta_generic_to_shared(val[0] + thread_sh_idx);
//     CP_ASYNC_CG(row_smem_addr, reinterpret_cast<const int4*>(A_rowind + ptr),
//     COPY_BYTES); CP_ASYNC_CG(col_smem_addr, reinterpret_cast<const
//     int4*>(A_colind + ptr), COPY_BYTES); CP_ASYNC_CG(val_smem_addr,
//     reinterpret_cast<const float4*>(A_value + ptr), COPY_BYTES);
//     CP_ASYNC_COMMIT_GROUP();
//     CP_ASYNC_WAIT_GROUP(0);
//     ptr += 32;
//     int tile_num = (hb-lb) / 32;
//     float4 dense_matrix_fragment[32];
//     for(int j = 1; j < tile_num; j++) {
//       int smem_sel = (j & 1) ^ 1;
//       int smem_sel_next = ( (j - 1) & 1) ^ 1;
// #pragma unroll
//       for (int kk=0; kk<32; kk++) {
//       offset = col[smem_sel][sm_offset + kk]*k + cid;
//       dense_matrix_fragment[kk] = __ldg(reinterpret_cast<const float4*>(B +
//       offset));
//       }
//
//       if (ptr < hb) {
//       uint32_t row_smem_addr = __cvta_generic_to_shared(row[smem_sel_next] +
//       thread_sh_idx); CP_ASYNC_CG(row_smem_addr, reinterpret_cast<const
//       int4*>(A_rowind + ptr), COPY_BYTES); uint32_t col_smem_addr =
//       __cvta_generic_to_shared(col[smem_sel_next] + thread_sh_idx);
//       CP_ASYNC_CG(col_smem_addr, reinterpret_cast<const int4*>(A_colind +
//       ptr), COPY_BYTES); uint32_t val_smem_addr =
//       __cvta_generic_to_shared(val[smem_sel_next] + thread_sh_idx);
//       CP_ASYNC_CG(val_smem_addr, reinterpret_cast<const float4*>(A_value +
//       ptr), COPY_BYTES); } else { break;
//       }
// #pragma unroll
//       for (int kk=0; kk<32; kk++) {
//       current_rid = row[smem_sel][sm_offset + kk];
//       if(current_rid != former_row_id) {
//         atomicAdd(&C[former_row_id*k + cid], acc1);
//         atomicAdd(&C[former_row_id*k + cid + 1], acc2);
//         atomicAdd(&C[former_row_id*k + cid + 2], acc3);
//         atomicAdd(&C[former_row_id*k + cid + 3], acc4);
//         acc1 = sum_init();
//         acc2 = sum_init();
//         acc3 = sum_init();
//         acc4 = sum_init();
//         former_row_id = current_rid;
//       }
//       float v = val[smem_sel][sm_offset+kk];
//       acc1 = sum_reduce(acc1,  v * dense_matrix_fragment[kk].x);
//       acc2 = sum_reduce(acc2, v * dense_matrix_fragment[kk].y);
//       acc3 = sum_reduce(acc3,  v * dense_matrix_fragment[kk].z);
//       acc4 = sum_reduce(acc4, v * dense_matrix_fragment[kk].w);
//       }
//       ptr += 32;
//       CP_ASYNC_COMMIT_GROUP();
//       CP_ASYNC_WAIT_GROUP(0);
//     }
//     int smem_sel = (tile_num & 1) ^ 1;
// #pragma unroll
//     for (int kk=0; kk<32; kk++) {
//       current_rid = row[smem_sel][sm_offset + kk];
//       if(current_rid != former_row_id) {
//       atomicAdd(&C[former_row_id*k + cid], acc1);
//       atomicAdd(&C[former_row_id*k + cid + 1], acc2);
//       atomicAdd(&C[former_row_id*k + cid + 2], acc3);
//       atomicAdd(&C[former_row_id*k + cid + 3], acc4);
//       acc1 = sum_init();
//       acc2 = sum_init();
//       acc3 = sum_init();
//       acc4 = sum_init();
//       former_row_id = current_rid;
//       }
//       float v = val[smem_sel][sm_offset+kk];
//       offset = col[smem_sel][sm_offset + kk]*k + cid;
//       float4 d = __ldg(reinterpret_cast<const float4*>(B + offset));
//       acc1 = sum_reduce(acc1,  v * d.x);
//       acc2 = sum_reduce(acc2, v * d.y);
//       acc3 = sum_reduce(acc3,  v * d.z);
//       acc4 = sum_reduce(acc4, v * d.w);
//     }
//     atomicAdd(&C[current_rid*k + cid], acc1);
//     atomicAdd(&C[current_rid*k + cid + 1], acc2);
//     atomicAdd(&C[current_rid*k + cid + 2], acc3);
//     atomicAdd(&C[current_rid*k + cid + 3], acc4);
// }
//
// void cooLBSpMM(int M, int N, int K1, const int *ARowInd, const int *AColInd ,
// const int ANnz, const float* Ax, const float* Bx, float* Cx) {
//     const int tile_k = (N +31)/32;
//     int nnz_per_warp = 32;
//     if ( (ANnz / M > 256) && (M > 5000)) nnz_per_warp = 128; // 256
//     const int n_block = (((ANnz +nnz_per_warp-1)/nnz_per_warp + 7) / 8);
//       if(nnz_per_warp <= 32)
//       LBSPMMKernel<<< dim3(n_block, tile_k, 1), dim3(32,8,1),
//       768*sizeof(int)>>>(M, N, ANnz, nnz_per_warp, ARowInd,
//                                           AColInd, Ax, Bx, Cx);
//       else
//       LBSPMMKernel_4_8_float4_float4_async_double_buffer<<< dim3(n_block,
//       tile_k, 1), dim3(8,8,1)>>>(
//           M, N, ANnz, nnz_per_warp, ARowInd, AColInd, Ax, Bx, Cx);
// }

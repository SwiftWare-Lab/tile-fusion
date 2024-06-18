//
// Created by salehm32 on 18/06/24.
//
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <aggregation/def.h>

__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

__device__ __forceinline__ float sum_init() {
  return 0.0;
}


__global__ void topoSimpleSPMMKernel(
    int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C
) {
  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    float acc1 = sum_init();
    int offset;
    for (int ptr=lb; ptr<hb; ptr++) {
      offset = A_indices[ptr]*k+threadIdx.x;
      acc1 = sum_reduce(acc1, B[offset]);
    }
    C[(rid*k+threadIdx.x)] = acc1;
  }
}

__global__ void topoCacheSPMMKernel(
    int m, int k, const int* A_indptr, const int* A_indices, const float* A_value, const float* B, float* C
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset + threadIdx.x;

  int cid = (blockIdx.y<<5)+threadIdx.x;
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  int value_off = blockDim.y * blockDim.x;

  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    int ptr = lb+threadIdx.x;
    float acc1 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[sm_offset+kk]+cid;
          acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)]*B[offset]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
            acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)]*B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
        C[offset] = acc1;}
    }
  }
}

__global__ void topoCacheCoarsenSPMMKernel(
    int m, int k, const int* A_indptr, const int* A_indices, const float* A_value, const float* B, float* C
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset+threadIdx.x;
  int value_off = blockDim.y * blockDim.x;


  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {

    int cid = (blockIdx.y<<6)+threadIdx.x;
    int lb = A_indptr[rid];
    int hb = A_indptr[rid+1];
    int ptr = lb+threadIdx.x;
    int offset;
    float acc1 = sum_init();
    float acc2 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)] * B[offset]);
          acc2 = sum_reduce(acc2, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)] * B[(offset+32)]);
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
      C[offset+32] = acc2;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
            acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)] * B[offset]);}
          if (nout>1) {
            acc2 = sum_reduce(acc2, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)] * B[(offset+32)]);}
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
        C[offset] = acc1;}
      if (nout>1) {
        C[offset+32] = acc2;}
    }
  }
}

void csrGeSpMM(const int M, const int N, const int K, const int* Ap,
            const int* Ai, const float* Ax,
            const float *Bx, float *Cx) {
  const int *rowindA_csr = Ap;
  const int *colindA = Ai;
    if (N<32) {
      const int row_per_block = 128/N;
      const int n_block = (M +row_per_block-1)/row_per_block;
      topoSimpleSPMMKernel<<< dim3(n_block,1,1),dim3(N, row_per_block, 1)>>>(
          M, N, rowindA_csr, colindA, Bx, Cx);
    } else if (N < 64) {
      const int tile_k = (N+31)/32;
      const int n_block = (M +3)/4;
      topoCacheSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,4,1), 256*sizeof(int)>>>(M, N, rowindA_csr, colindA, Ax,
                                                 Bx, Cx);
    } else {
      const int tile_k = (N+63)/64;
      const int n_block = (M +8-1)/8;
      topoCacheCoarsenSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,8,1), 2*8*32*sizeof(int)>>>(
          M, N, rowindA_csr, colindA, Ax, Bx, Cx);
    }
}
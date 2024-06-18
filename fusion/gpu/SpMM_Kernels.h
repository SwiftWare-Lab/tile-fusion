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

//Ge-SpMM

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




//TODO: it has bugs(severe bugs when N<32).
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
      const int n_block = (M+3)/4;
      topoCacheSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,4,1), 256*sizeof(int)>>>(M, N, rowindA_csr, colindA, Ax,
                                                 Bx, Cx);
    } else {
      const int tile_k = (N+63)/64;
      const int n_block = (M +8-1)/8;
      topoCacheCoarsenSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,8,1), 2*8*32*sizeof(int)>>>(
          M, N, rowindA_csr, colindA, Ax, Bx, Cx);
    }
}


//HP-SpMM
// TODO: Arch Specific. Need to find a pragma or a way to run it on a server.
//  error: Feature 'cp.async' requires .target sm_80 or higher
//
//__global__ void LBSPMMKernel(
//    int m, int k, int NNZ, int nnz_per_warp, const int* __restrict__ A_rowind, const int* __restrict__ A_colind, const  float* __restrict__ A_value, const  float* __restrict__ B,  float* __restrict__ C
//) {
//    extern __shared__ int sh[];
//    int sm_offset = (threadIdx.y<<5);
//    int thread_idx = sm_offset + threadIdx.x;
//    int cid = (blockIdx.y<<5)+threadIdx.x;
//    int off = blockDim.y * blockDim.x;
//    int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
//    int warp_start = warp_id * nnz_per_warp;
//    if (warp_start > NNZ - 1) {
//      return;
//    }
//    int former_row_id = __ldg(A_rowind + warp_start);
//    int current_rid = former_row_id;
//    int lb = warp_start;
//    int hb = warp_start + nnz_per_warp;
//    if (hb > NNZ){
//      hb = NNZ;
//    }
//    int offset;
//    int ptr = lb + threadIdx.x;
//    float acc1 = sum_init();
//    for(int i = nnz_per_warp; i > 0; i -= 32) {
//      if (ptr < hb) {
//      sh[thread_idx] = __ldg(A_rowind + ptr);
//      sh[thread_idx + off] = __ldg(A_colind + ptr)*k;
//      reinterpret_cast<float*>(sh)[thread_idx + off + off] =  __ldg(A_value + ptr);
//      } else {
//      sh[thread_idx] = 0;
//      sh[thread_idx + off] = 0;
//      sh[thread_idx + off + off] = 0;
//      }
//      __syncwarp();
//#pragma unroll
//      for (int kk=0; kk<32; kk++) {
//      current_rid = sh[sm_offset + kk];
//      if(current_rid != former_row_id) {
//        atomicAdd(&C[former_row_id*k + cid], acc1);
//        acc1 = sum_init();
//        former_row_id = current_rid;
//      }
//      offset = sh[sm_offset + kk + off] + cid;
//      acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+off+off)] * __ldg(B + offset));
//      }
//      ptr += 32;
//    }
//    __syncwarp();
//    atomicAdd(&C[current_rid*k + cid], acc1);
//}
//
//#define COPY_BYTES 16
//#define CP_ASYNC_CG(dst, src, Bytes) \
//    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
//
//#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
//#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
//
//__global__ void LBSPMMKernel_4_8_float4_float4_async_double_buffer(
//    int m, int k, int NNZ, int nnz_per_warp, const int* __restrict__ A_rowind, const int* __restrict__ A_colind, const  float* __restrict__ A_value, const  float* __restrict__ B,  float* __restrict__ C
//) {
//    __shared__ int row[2][256];
//    __shared__ int col[2][256];
//    __shared__ float val[2][256];
//    int sm_offset = (threadIdx.y<<5);
//    int thread_sh_idx = sm_offset + (threadIdx.x << 2);
//    int cid = (blockIdx.y<<5)+(threadIdx.x<<2);
//    int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
//    int warp_start = warp_id * nnz_per_warp;
//    if (warp_start > NNZ - 1) {
//      return;
//    }
//    int former_row_id = __ldg(A_rowind + warp_start);
//    int current_rid = former_row_id;
//    int lb = warp_start;
//    int hb = warp_start + nnz_per_warp;
//    if (hb > NNZ){
//      hb = NNZ;
//    }
//    int offset;
//    int ptr = lb + (threadIdx.x<<2);
//    float acc1 = sum_init();
//    float acc2 = sum_init();
//    float acc3 = sum_init();
//    float acc4 = sum_init();
//    uint32_t row_smem_addr = __cvta_generic_to_shared(row[0] + thread_sh_idx);
//    uint32_t col_smem_addr = __cvta_generic_to_shared(col[0] + thread_sh_idx);
//    uint32_t val_smem_addr = __cvta_generic_to_shared(val[0] + thread_sh_idx);
//    CP_ASYNC_CG(row_smem_addr, reinterpret_cast<const int4*>(A_rowind + ptr), COPY_BYTES);
//    CP_ASYNC_CG(col_smem_addr, reinterpret_cast<const int4*>(A_colind + ptr), COPY_BYTES);
//    CP_ASYNC_CG(val_smem_addr, reinterpret_cast<const float4*>(A_value + ptr), COPY_BYTES);
//    CP_ASYNC_COMMIT_GROUP();
//    CP_ASYNC_WAIT_GROUP(0);
//    ptr += 32;
//    int tile_num = (hb-lb) / 32;
//    float4 dense_matrix_fragment[32];
//    for(int j = 1; j < tile_num; j++) {
//      int smem_sel = (j & 1) ^ 1;
//      int smem_sel_next = ( (j - 1) & 1) ^ 1;
//#pragma unroll
//      for (int kk=0; kk<32; kk++) {
//      offset = col[smem_sel][sm_offset + kk]*k + cid;
//      dense_matrix_fragment[kk] = __ldg(reinterpret_cast<const float4*>(B + offset));
//      }
//
//      if (ptr < hb) {
//      uint32_t row_smem_addr = __cvta_generic_to_shared(row[smem_sel_next] + thread_sh_idx);
//      CP_ASYNC_CG(row_smem_addr, reinterpret_cast<const int4*>(A_rowind + ptr), COPY_BYTES);
//      uint32_t col_smem_addr = __cvta_generic_to_shared(col[smem_sel_next] + thread_sh_idx);
//      CP_ASYNC_CG(col_smem_addr, reinterpret_cast<const int4*>(A_colind + ptr), COPY_BYTES);
//      uint32_t val_smem_addr = __cvta_generic_to_shared(val[smem_sel_next] + thread_sh_idx);
//      CP_ASYNC_CG(val_smem_addr, reinterpret_cast<const float4*>(A_value + ptr), COPY_BYTES);
//      } else {
//      break;
//      }
//#pragma unroll
//      for (int kk=0; kk<32; kk++) {
//      current_rid = row[smem_sel][sm_offset + kk];
//      if(current_rid != former_row_id) {
//        atomicAdd(&C[former_row_id*k + cid], acc1);
//        atomicAdd(&C[former_row_id*k + cid + 1], acc2);
//        atomicAdd(&C[former_row_id*k + cid + 2], acc3);
//        atomicAdd(&C[former_row_id*k + cid + 3], acc4);
//        acc1 = sum_init();
//        acc2 = sum_init();
//        acc3 = sum_init();
//        acc4 = sum_init();
//        former_row_id = current_rid;
//      }
//      float v = val[smem_sel][sm_offset+kk];
//      acc1 = sum_reduce(acc1,  v * dense_matrix_fragment[kk].x);
//      acc2 = sum_reduce(acc2, v * dense_matrix_fragment[kk].y);
//      acc3 = sum_reduce(acc3,  v * dense_matrix_fragment[kk].z);
//      acc4 = sum_reduce(acc4, v * dense_matrix_fragment[kk].w);
//      }
//      ptr += 32;
//      CP_ASYNC_COMMIT_GROUP();
//      CP_ASYNC_WAIT_GROUP(0);
//    }
//    int smem_sel = (tile_num & 1) ^ 1;
//#pragma unroll
//    for (int kk=0; kk<32; kk++) {
//      current_rid = row[smem_sel][sm_offset + kk];
//      if(current_rid != former_row_id) {
//      atomicAdd(&C[former_row_id*k + cid], acc1);
//      atomicAdd(&C[former_row_id*k + cid + 1], acc2);
//      atomicAdd(&C[former_row_id*k + cid + 2], acc3);
//      atomicAdd(&C[former_row_id*k + cid + 3], acc4);
//      acc1 = sum_init();
//      acc2 = sum_init();
//      acc3 = sum_init();
//      acc4 = sum_init();
//      former_row_id = current_rid;
//      }
//      float v = val[smem_sel][sm_offset+kk];
//      offset = col[smem_sel][sm_offset + kk]*k + cid;
//      float4 d = __ldg(reinterpret_cast<const float4*>(B + offset));
//      acc1 = sum_reduce(acc1,  v * d.x);
//      acc2 = sum_reduce(acc2, v * d.y);
//      acc3 = sum_reduce(acc3,  v * d.z);
//      acc4 = sum_reduce(acc4, v * d.w);
//    }
//    atomicAdd(&C[current_rid*k + cid], acc1);
//    atomicAdd(&C[current_rid*k + cid + 1], acc2);
//    atomicAdd(&C[current_rid*k + cid + 2], acc3);
//    atomicAdd(&C[current_rid*k + cid + 3], acc4);
//}
//
//void cooLBSpMM(int M, int N, int K1, const int *ARowInd, const int *AColInd , const int ANnz, const float* Ax, const float* Bx, float* Cx) {
//    const int tile_k = (N +31)/32;
//    int nnz_per_warp = 32;
//    if ( (ANnz / M > 256) && (M > 5000)) nnz_per_warp = 128; // 256
//    const int n_block = (((ANnz +nnz_per_warp-1)/nnz_per_warp + 7) / 8);
//      if(nnz_per_warp <= 32)
//      LBSPMMKernel<<< dim3(n_block, tile_k, 1), dim3(32,8,1), 768*sizeof(int)>>>(M, N, ANnz, nnz_per_warp, ARowInd,
//                                          AColInd, Ax, Bx, Cx);
//      else
//      LBSPMMKernel_4_8_float4_float4_async_double_buffer<<< dim3(n_block, tile_k, 1), dim3(8,8,1)>>>(
//          M, N, ANnz, nnz_per_warp, ARowInd, AColInd, Ax, Bx, Cx);
//}


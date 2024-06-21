//
// Created by salehm32 on 21/06/24.
//

#ifndef SPARSE_FUSION_SW_SPMM_KERNELS_H
#define SPARSE_FUSION_SW_SPMM_KERNELS_H

__global__ void spmmSeqReduceRowBalance(const int M, const int N,
                                        const int K, const int* Ap,
                                        const int* Ai,
                                        const float* Ax,
                                        const float* Bx,
                                        float* Cx) {
  int row_tile = blockDim.y;
  int subwarp_id = threadIdx.y;
  int stride = row_tile * gridDim.x;
  int row = blockIdx.x * row_tile + subwarp_id;
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  if (v_id < N) {
    Bx += v_id;
    Cx += v_id;

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
      Cx[row * N] = res;
    }
  }
}

#endif // SPARSE_FUSION_SW_SPMM_KERNELS_H

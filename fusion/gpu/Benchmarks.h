//
// Created by salehm32 on 09/01/25.
//

#ifndef SPARSE_FUSION_BENCHMARKS_H
#define SPARSE_FUSION_BENCHMARKS_H

#include "Cuda_SpMM_Demo_Utils.h"
#include "SWTensorBench.h"
#include "Timer.h"

using namespace swiftware::benchmark;

__global__
    void filter_k(int *dst, int *nres, const int *src, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
//  printf("i = %d\n", i);
  if(i < n && src[i] > 0)
    atomicAdd(nres, 1);
}

__global__
    void filter_k_no_atomic(int *dst, int *nres, const int *src, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n && src[i] > 0) {
    nres[0] = nres[0] + 1;
  }

}



class FilterKBench : public SWTensorBench<float> {
protected:
  int N;
  int *h_src, *h_dst, *d_src, *d_dst, *d_nres;

  void setup() override {
    h_src = new int [N];
    for (int i = 0; i < N; ++i) {
      if (i % 2 == 0)
        h_src[i] = -1;
      else
        h_src[i] = i;
    }
    h_dst = new int [N];
    cudaMalloc(&d_src, N * sizeof(int));
    cudaMalloc(&d_dst, N * sizeof(int));
    cudaMalloc(&d_nres, sizeof(int));
    cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice);
  }

  void preExecute() override {}

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    Timer t;
    t.startGPU();
    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    filter_k<<<grid, block>>>(d_dst, d_nres, d_src, N);
    cudaDeviceSynchronize();
    t.stopGPU("FilterK");
    return t;
  }

public:
  FilterKBench(CudaTensorInputs *In1, Stats *Stat1, int N1)
      : SWTensorBench<float>(In1, Stat1), N(N1) {}

  ~FilterKBench(){
    delete[] h_src;
    delete[] h_dst;
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_nres);
  }

};


class FilterKNoAtomicBench : public FilterKBench {
protected:
  void setup() override {
    FilterKBench::setup();
  }
  Timer execute() override {
    Timer t;
    t.startGPU();
    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    filter_k_no_atomic<<<grid, block>>>(d_dst, d_nres, d_src, N);
    cudaDeviceSynchronize();
    t.stopGPU("FilterKNoAtomic");
    return t;
  }

public:
  FilterKNoAtomicBench(CudaTensorInputs *In1, Stats *Stat1, int N1)
      : FilterKBench(In1, Stat1, N1) {}
};
#endif // SPARSE_FUSION_BENCHMARKS_H

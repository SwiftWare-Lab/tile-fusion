//
// Created by salehm32 on 30/01/25.
//

#ifndef SPARSE_FUSION_CUDA_GEMM_DEMO_UTILS_H
#define SPARSE_FUSION_CUDA_GEMM_DEMO_UTILS_H

#include "../example/SpMM_SpMM_Demo_Utils.h"
#include "Cuda_GeMM_SpMM_Demo_Utils.h"
#include "Cuda_SpMM_Demo_Utils.h"
#include "GeMM_Kernels.cuh"
#include "Stats.h"
#include "Timer.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Defs.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/GeMM.h"

using namespace sym_lib;

class GeMMCPUParallel : public SWTensorBench<float> {
protected:
  CudaGeMMSpMMTensorInputs *InTensor;

  void setup() override {}

  void preExecute() override {}

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::dense::geMMParallel<float>(InTensor->M, InTensor->N, InTensor->K,
                                          InTensor->HCx, InTensor->Bx,
                                          OutTensor->ACx, InTensor->NumThreads);
    t.stop();
    return t;
  }

  bool verify(double &Error) override {
    bool retValue = true;
    if (!InTensor->IsSolProvided) {
      Error = 0;
      return true;
    }
    double infNorm = 0;
    // Since For now This is For SpMM I'm Using ACx and its dimensions.
    // TODO: Later on I need to have separate classes for SpMM and SpMM-SpMM or
    // think of an other way.
    for (int i = 0; i < InTensor->M * InTensor->N; ++i) {
      if (std::abs(OutTensor->ACx[i] - InTensor->CorrectSol[i]) > infNorm) {
        infNorm = std::abs(OutTensor->ACx[i] - InTensor->CorrectSol[i]);
      }
    }
    Error = (double)infNorm;
    if (infNorm > InTensor->Threshold) {
      retValue = false;
    }
    return retValue;
  }

public:
  CudaTensorOutputs *OutTensor;
  GeMMCPUParallel(CudaGeMMSpMMTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<float>(In1, Stat1) {
    OutTensor = new CudaTensorOutputs(In1->M, In1->N, In1->L);
    InTensor = In1;
  }

  ~GeMMCPUParallel() { delete OutTensor; }
};

class GeMMCuda2DBlocking : public GeMMCPUParallel {
  Timer execute() override {
    OutTensor->reset();
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(InTensor->N) + block_dim.x - 1U) /
            block_dim.x,
        (static_cast<unsigned int>(InTensor->M) + block_dim.y - 1U) /
            block_dim.y,
        1U};
//    std::cout << grid_dim.x << " " << grid_dim.y << " " << grid_dim.z << std::endl;
//    std::cout << block_dim.x << " " << block_dim.y << " " << block_dim.z << std::endl;
    Timer t;
    t.startGPU();
    gemm2DBlocking<float, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                   BLOCK_TILE_SIZE_K><<<grid_dim, block_dim>>>(
        InTensor->M, InTensor->N, InTensor->K, (float)1., InTensor->DBx,
        InTensor->K, InTensor->DCx, InTensor->N, (float)0., OutTensor->DACx,
        InTensor->N);
    t.stopGPU("gemm_2DTiling");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  GeMMCuda2DBlocking(CudaGeMMSpMMTensorInputs *In1, Stats *Stat1)
      : GeMMCPUParallel(In1, Stat1) {}
};

#endif // SPARSE_FUSION_CUDA_GEMM_DEMO_UTILS_H

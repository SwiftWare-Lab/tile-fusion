//
// Created by salehm32 on 31/01/25.
//

#ifndef SPARSE_FUSION_CUDA_GEMM_SPMM_DEMO_UTILS_H
#define SPARSE_FUSION_CUDA_GEMM_SPMM_DEMO_UTILS_H

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

struct CudaGeMMSpMMTensorInputs : public TensorInputs<float> {
  int *DACsrAp;
  int *DACsrI;
  float *DACsrVal;
  float *HACsrVal;

  float *DBx;
  float *HCx;
  float *DCx;

  CudaGeMMSpMMTensorInputs(int M1, int N1, int K1, int L1, sym_lib::CSC *A1,
                           sym_lib::CSC *B1, int NumThreads1, int NumTrial1,
                           std::string ExpN)
      : TensorInputs<float>(M1, N1, K1, L1, A1, B1, NumThreads1, NumTrial1,
                            ExpN) {
    size_t rPtrSize = (ACsr->m + 1) * sizeof(int);
    size_t cIndexSize = ACsr->nnz * sizeof(int);
    size_t valSize = ACsr->nnz * sizeof(float);
    size_t denseSize1 = K * N * sizeof(float);
    size_t denseSize2 = K * L * sizeof(float);
    HCx = new float[denseSize2];
    for (int i = 0; i < denseSize2; i++) {
      HCx[i] = 1.0;
    }
    cudaMalloc(&DACsrAp, rPtrSize);
    cudaMalloc(&DACsrI, cIndexSize);
    cudaMalloc(&DACsrVal, valSize);
    cudaMalloc(&DBx, denseSize1);
    cudaMalloc(&DCx, denseSize2);
    HACsrVal = new float[ACsr->nnz];
    for (int i = 0; i < ACsr->nnz; i++) {
      HACsrVal[i] = (float)ACsr->x[i];
    }
    cudaMemcpy(DACsrAp, ACsr->p, rPtrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(DACsrI, ACsr->i, cIndexSize, cudaMemcpyHostToDevice);
    cudaMemcpy(DACsrVal, HACsrVal, valSize, cudaMemcpyHostToDevice);
    cudaMemcpy(DBx, Bx, denseSize1, cudaMemcpyHostToDevice);
    cudaMemcpy(DCx, HCx, denseSize2, cudaMemcpyHostToDevice);
  }

  ~CudaGeMMSpMMTensorInputs() {
    delete[] HACsrVal;
    delete[] HCx;
    cudaFree(DACsrAp);
    cudaFree(DACsrI);
    cudaFree(DACsrVal);
    cudaFree(DBx);
    cudaFree(DCx);
  }
};

class GeMMSpMMCPU : public SWTensorBench<float> {
protected:
  CudaGeMMSpMMTensorInputs *InTensor;

  void setup() override {
    this->St->OtherStats["Number of Fused Rows"] = {0.};
    this->St->OtherStats["Number of Fused Nnz"] = {0.};
  }

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
    swiftware::sparse::spmmCsrParallel<float>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->HACsrVal, OutTensor->ACx, OutTensor->Xx,
        InTensor->NumThreads);
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
      if (std::abs(OutTensor->Xx[i] - InTensor->CorrectSol[i]) > infNorm) {
        infNorm = std::abs(OutTensor->Xx[i] - InTensor->CorrectSol[i]);
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
  GeMMSpMMCPU(CudaGeMMSpMMTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<float>(In1, Stat1) {
    OutTensor = new CudaTensorOutputs(In1->M, In1->N, In1->L);
    InTensor = In1;
  }

  ~GeMMSpMMCPU() { delete OutTensor; }
};

class UnfusedGeMMSpMMGPU : public GeMMSpMMCPU {
protected:
  int SpMM_MGridDim;
  int SpMM_NGridDim;
  int SpMM_MBlockDim;
  int SpMM_NBlockDim;
  int SpMM_ThreadPerBlock;
  int GeMM_MGridDim;
  int GeMM_NGridDim;
  int GeMM_MBlockDim;
  int GeMM_NBlockDim;

  static constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
  static constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
  static constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
  static constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X *
                                            BLOCK_TILE_SIZE_Y};

  void setup() override {
    GeMMSpMMCPU::setup();
    SpMM_NGridDim = CEIL(InTensor->N, SpMM_ThreadPerBlock);
    SpMM_NBlockDim = MIN(InTensor->N, SpMM_ThreadPerBlock);
    SpMM_MBlockDim = CEIL(SpMM_ThreadPerBlock, SpMM_NBlockDim);
    SpMM_MGridDim = CEIL(InTensor->M, SpMM_MBlockDim);
    GeMM_NGridDim = CEIL(InTensor->N, BLOCK_TILE_SIZE_X);
    GeMM_MGridDim = CEIL(InTensor->M, BLOCK_TILE_SIZE_Y);
    GeMM_NBlockDim = BLOCK_TILE_SIZE_X;
    GeMM_MBlockDim = BLOCK_TILE_SIZE_Y;
  }

  Timer execute() override {
    OutTensor->reset();
    Timer t1;
    dim3 spMMGridDim(SpMM_MGridDim, SpMM_NGridDim, 1);
    dim3 spMMBlockDim(SpMM_NBlockDim, SpMM_MBlockDim, 1);
    dim3 geMMGridDim(GeMM_NGridDim, GeMM_MGridDim, 1);
    dim3 geMMBlockDim(GeMM_NBlockDim, GeMM_MBlockDim, 1);
    t1.startGPU();
    gemm2DBlocking<float, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                   BLOCK_TILE_SIZE_K><<<geMMGridDim, geMMBlockDim>>>(
        InTensor->L, InTensor->N, InTensor->K, (float)1., InTensor->DCx,
        InTensor->K, InTensor->DBx, InTensor->N, (float)0., OutTensor->DACx,
        InTensor->N);
    // TODO: Since I know kernel calls are nonBlocking I used this. Is this the
    // right way?
    //  I might need to test without synchronization to make sure nonBlocking
    //  property of kernel calls is valid.
    cudaDeviceSynchronize();
    csrspmm_seqreduce_rowbalance_kernel<<<spMMGridDim, spMMBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->L, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx);
    cudaDeviceSynchronize();
    t1.stopGPU("UnfusedGeMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  UnfusedGeMMSpMMGPU(CudaGeMMSpMMTensorInputs *In1, Stats *Stat1)
      : GeMMSpMMCPU(In1, Stat1) {
    SpMM_ThreadPerBlock = 128;
  }
};


class FusedGeMMSpMMGPU : public UnfusedGeMMSpMMGPU {
protected:
  int *HROAp;
  int *HROAi;
  float *HROAx;
  int *HFPtr;
  int *HROId;
  int *DROAp;
  int *DROAi;
  float *DROAx;
  int *DFPtr;
  int *DROId;
  int UFDim;
  int UFMGridDim;

  //TODO: Fix the analysis function with respect to the new fused and unfused structure.
  Timer analysis() override {
    Timer t1;
    t1.start();
    std::vector<int> ufRows;
    std::vector<std::vector<int>> fRows(GeMM_MGridDim);
    int rowTile = GeMM_MBlockDim;
    int *ap = InTensor->ACsr->p;
    int *ai = InTensor->ACsr->i;
    float *ax = InTensor->HACsrVal;
    int fNnzCount = 0;
    for (int i = 0; i < InTensor->M; i += rowTile) {
      int t = i / rowTile;
      int end = MIN(i + rowTile, InTensor->M);
      for (int ii = i; ii < end; ii++) {
        if (ai[ap[ii]] < i || ai[ap[ii+1]-1] >= end) {
          ufRows.push_back(ii);
        }
        else {
          fRows[t].push_back(ii);
          fNnzCount += ap[ii + 1] - ap[ii];
        }
      }
    }
    int ufCount = ufRows.size();
    int nnzCount = InTensor->ACsr->nnz;
//    int uFNnzCount = nnzCount - fNnzCount;
    int fIdCount = InTensor->M - ufCount;
    int fPtrCount = GeMM_MGridDim + 1;
    int aRows = InTensor->M;
    HROId = new int[aRows];
    HROAp = new int[aRows+1];
    HROAi = new int[nnzCount];
    HROAx = new float[nnzCount];
    HFPtr = new int[fPtrCount];
    HFPtr[0] = 0;
    HROAp[0] = 0;
    int j = HFPtr[0];
    int p = HROAp[0];
    for (int i = 0; i < fRows.size(); i++) {
      HFPtr[i + 1] = HFPtr[i] + fRows[i].size();
      for (; j < HFPtr[i + 1]; j++) {
        HROId[j] = fRows[i][j - HFPtr[i]];
        HROAp[j+1] = HROAp[j] + ap[HROId[j]+1] - ap[HROId[j]];
        for (; p < HROAp[j+1]; p++){
          HROAi[p] = ai[p - HROAp[j] + ap[HROId[j]]];
          HROAx[p] = ax[p - HROAp[j] + ap[HROId[j]]];
        }
      }
    }
    for (int i = 0; i < ufRows.size(); i++) {
      HROId[i+j] = ufRows[i];
      int rowNnzCount = ap[ufRows[i] + 1] - ap[ufRows[i]];
      HROAp[i + j + 1] = rowNnzCount + HROAp[i + j];
      for (int k = 0; k < rowNnzCount; k++) {
        HROAi[HROAp[i + j] + k] = ai[ap[ufRows[i]] + k];
        HROAx[HROAp[i + j] + k] = ax[ap[ufRows[i]] + k];
      }
    }
    this->St->OtherStats["Number of Fused Rows"] = {(double)fIdCount};
    this->St->OtherStats["Number of Fused Nnz"] = {(double)fNnzCount};
    UFDim = ufCount;
    UFMGridDim = CEIL(UFDim, SpMM_MBlockDim);
    cudaMalloc(&DROId, (aRows) * sizeof(int));
    cudaMalloc(&DROAp, (aRows+1) * sizeof(int));
    cudaMalloc(&DROAi, nnzCount * sizeof(int));
    cudaMalloc(&DROAx, nnzCount * sizeof(float));
    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    cudaMemcpy(DROId, HROId, aRows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (aRows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, nnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, nnzCount * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DFPtr, HFPtr, fPtrCount * sizeof(int), cudaMemcpyHostToDevice);
    t1.stop();
    return t1;
  }

  Timer execute() override {
    OutTensor->reset();
    Timer t1;
    dim3 fGridDim(GeMM_NGridDim, GeMM_MGridDim, 1);
    dim3 fBlockDim(GeMM_NBlockDim, GeMM_MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, SpMM_NGridDim, 1);
    dim3 ufBlockDim(SpMM_NBlockDim, SpMM_MBlockDim, 1);
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    int fusedNum = this->InTensor->M - UFDim;
    t1.startGPU();
    gemm2DBlockingSpMMSeqRedFused<float, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                  BLOCK_TILE_SIZE_K><<<fGridDim,
                                                            fBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->K,
        DROAp, DROAi, DROAx, DFPtr, DROId,(float)1., InTensor->DCx,
        InTensor->K, InTensor->DBx, InTensor->N, (float)0., OutTensor->DACx,
        OutTensor->DXx);
    cudaDeviceSynchronize();
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N, InTensor->K, DROAp + fusedNum, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DROId + fusedNum);
    cudaDeviceSynchronize();
    t1.stopGPU("FusedGeMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedGeMMSpMMGPU(CudaGeMMSpMMTensorInputs *In1, Stats *Stat1)
      : UnfusedGeMMSpMMGPU(In1, Stat1) {};
  ~FusedGeMMSpMMGPU() {
    delete[] HROId;
    delete[] HROAp;
    delete[] HROAi;
    delete[] HROAx;
    delete[] HFPtr;
    cudaFree(DROId);
    cudaFree(DROAp);
    cudaFree(DROAi);
    cudaFree(DROAx);
    cudaFree(DFPtr);
  }
};


class FusedSpMM1DGeMM2DGPU : public GeMMSpMMCPU {
protected:
  int GeMM_MGridDim;
  int GeMM_NGridDim;
  int GeMM_MBlockDim;
  int GeMM_NBlockDim;

  static constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
  static constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
  static constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
  static constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X *
                                            BLOCK_TILE_SIZE_Y};

  void setup() override {
    GeMMSpMMCPU::setup();
    GeMM_NGridDim = CEIL(InTensor->N, BLOCK_TILE_SIZE_X);
    GeMM_MGridDim = CEIL(InTensor->M, BLOCK_TILE_SIZE_Y);
    GeMM_NBlockDim = BLOCK_TILE_SIZE_X;
    GeMM_MBlockDim = BLOCK_TILE_SIZE_Y;
  }

  Timer execute() override {
    OutTensor->reset();
    Timer t1;
    dim3 geMMGridDim(GeMM_NGridDim, GeMM_MGridDim, 1);
    dim3 geMMBlockDim(GeMM_NBlockDim, GeMM_MBlockDim, 1);
    t1.startGPU();
    fusedSpMM1DGeMM2DBlocking<float, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                              BLOCK_TILE_SIZE_K><<<geMMGridDim,
                                                   geMMBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->K,
        InTensor->DACsrAp, InTensor->DACsrI, InTensor->DACsrVal,(float)1., InTensor->DCx,
        InTensor->K, InTensor->DBx, InTensor->N, (float)0.,
        OutTensor->DACx, OutTensor->DXx);
    cudaDeviceSynchronize();
    t1.stopGPU("UnfusedGeMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMM1DGeMM2DGPU(CudaGeMMSpMMTensorInputs *In1, Stats *Stat1)
      : GeMMSpMMCPU(In1, Stat1) {}
};

class FusedSpMM2DGeMM2DGPU : public GeMMSpMMCPU {
protected:
  int GeMM_MGridDim;
  int GeMM_NGridDim;
  int GeMM_MBlockDim;
  int GeMM_NBlockDim;

  static constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
  static constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
  static constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
  static constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X *
                                            BLOCK_TILE_SIZE_Y};

  void setup() override {
    GeMMSpMMCPU::setup();
    GeMM_NGridDim = CEIL(InTensor->N, BLOCK_TILE_SIZE_X);
    GeMM_MGridDim = CEIL(InTensor->M, BLOCK_TILE_SIZE_Y);
    GeMM_NBlockDim = BLOCK_TILE_SIZE_X;
    GeMM_MBlockDim = BLOCK_TILE_SIZE_Y;
  }

  Timer execute() override {
    OutTensor->reset();
    Timer t1;
    dim3 geMMGridDim(GeMM_NGridDim, GeMM_MGridDim, 1);
    dim3 geMMBlockDim(GeMM_NBlockDim, GeMM_MBlockDim, 1);
    t1.startGPU();
    fusedSpMM2DGeMM2DBlocking<float, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                  BLOCK_TILE_SIZE_K><<<geMMGridDim,
                                                           geMMBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->K,
        InTensor->DACsrAp, InTensor->DACsrI, InTensor->DACsrVal,(float)1., InTensor->DCx,
        InTensor->K, InTensor->DBx, InTensor->N, (float)0.,
        OutTensor->DACx, OutTensor->DXx);
    cudaDeviceSynchronize();
    t1.stopGPU("UnfusedGeMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMM2DGeMM2DGPU(CudaGeMMSpMMTensorInputs *In1, Stats *Stat1)
      : GeMMSpMMCPU(In1, Stat1) {}
};

class FusedSpMM1DSMGeMM2DGPU : public GeMMSpMMCPU {
protected:
  int GeMM_MGridDim;
  int GeMM_NGridDim;
  int GeMM_MBlockDim;
  int GeMM_NBlockDim;

  static constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
  static constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
  static constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
  static constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X *
                                            BLOCK_TILE_SIZE_Y};

  void setup() override {
    GeMMSpMMCPU::setup();
    GeMM_NGridDim = CEIL(InTensor->N, BLOCK_TILE_SIZE_X);
    GeMM_MGridDim = CEIL(InTensor->M, BLOCK_TILE_SIZE_Y);
    GeMM_NBlockDim = BLOCK_TILE_SIZE_X;
    GeMM_MBlockDim = BLOCK_TILE_SIZE_Y;
  }

  Timer execute() override {
    OutTensor->reset();
    Timer t1;
    dim3 geMMGridDim(GeMM_NGridDim, GeMM_MGridDim, 1);
    dim3 geMMBlockDim(GeMM_NBlockDim, GeMM_MBlockDim, 1);
    t1.startGPU();
    fusedSpMM1DSMGeMM2DBlocking<float, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                              BLOCK_TILE_SIZE_K><<<geMMGridDim,
                                                   geMMBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->K,
        InTensor->DACsrAp, InTensor->DACsrI, InTensor->DACsrVal,(float)1., InTensor->DCx,
        InTensor->K, InTensor->DBx, InTensor->N, (float)0.,
        OutTensor->DACx, OutTensor->DXx);
    cudaDeviceSynchronize();
    t1.stopGPU("UnfusedGeMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMM1DSMGeMM2DGPU(CudaGeMMSpMMTensorInputs *In1, Stats *Stat1)
      : GeMMSpMMCPU(In1, Stat1) {}
};

class FusedSpMM2DGeMM2DAStationaryGPU : public GeMMSpMMCPU {
protected:
  int GeMM_MGridDim;
  int GeMM_NGridDim;
  int GeMM_MBlockDim;
  int GeMM_NBlockDim;

  static constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
  static constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
  static constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
  static constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X *
                                            BLOCK_TILE_SIZE_Y};

  void setup() override {
    GeMMSpMMCPU::setup();
    GeMM_NGridDim = CEIL(InTensor->N, BLOCK_TILE_SIZE_X);
    GeMM_MGridDim = CEIL(InTensor->M, BLOCK_TILE_SIZE_Y);
    GeMM_NBlockDim = BLOCK_TILE_SIZE_X;
    GeMM_MBlockDim = BLOCK_TILE_SIZE_Y;
  }

  Timer execute() override {
    OutTensor->reset();
    Timer t1;
    dim3 geMMGridDim(GeMM_NGridDim, GeMM_MGridDim, 1);
    dim3 geMMBlockDim(GeMM_NBlockDim, GeMM_MBlockDim, 1);
    t1.startGPU();
    fusedSpMM2DGeMMAStationary<float, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                BLOCK_TILE_SIZE_K><<<geMMGridDim,
                                                     geMMBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->K,
        InTensor->DACsrAp, InTensor->DACsrI, InTensor->DACsrVal,(float)1., InTensor->DCx,
        InTensor->K, InTensor->DBx, InTensor->N, (float)0.,
        OutTensor->DACx, OutTensor->DXx);
    cudaDeviceSynchronize();
    t1.stopGPU("UnfusedGeMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMM2DGeMM2DAStationaryGPU(CudaGeMMSpMMTensorInputs *In1, Stats *Stat1)
      : GeMMSpMMCPU(In1, Stat1) {}
};

#endif // SPARSE_FUSION_CUDA_GEMM_SPMM_DEMO_UTILS_H

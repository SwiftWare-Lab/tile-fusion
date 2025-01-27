//
// Created by salehm32 on 22/01/25.
//

#ifndef SPARSE_FUSION_CUDA_SPMM_SPMM_FP16_DEMO_UTILS_H
#define SPARSE_FUSION_CUDA_SPMM_SPMM_FP16_DEMO_UTILS_H
#include "Stats.h"
#include "../example/SpMM_SpMM_Demo_Utils.h"
#include "SpMM_Kernels.h"
#include "Timer.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Defs.h"
#include "sparse-fusion/Fusion_Utils.h"

struct CudaTensorInputsFP16: public TensorInputs<float>{
  int* DACsrAp;
  int* DACsrI;
  __half* DACsrVal;
  __half* HACsrVal;

  __half2* HBx;
  __half2* DBx;

  CudaTensorInputsFP16(int M1, int N1, int K1, int L1, sym_lib::CSC *A1,
                   sym_lib::CSC *B1, int NumThreads1, int NumTrial1,
                   std::string ExpN): TensorInputs<float>(M1, N1, K1, L1, A1, B1, NumThreads1, NumTrial1, ExpN){

    size_t rPtrSize = (ACsr->m + 1) * sizeof(int);
    size_t cIndexSize = ACsr->nnz * sizeof(int);
    size_t valSize = ACsr->nnz * sizeof(__half);
    size_t denseSize = K * N/2 * sizeof(__half2);
    cudaMalloc(&DACsrAp, rPtrSize);
    cudaMalloc(&DACsrI, cIndexSize);
    cudaMalloc(&DACsrVal, valSize);
    cudaMalloc(&DBx, denseSize);
    HACsrVal = new __half[ACsr->nnz];
    for (int i = 0; i < ACsr->nnz; i++){
      HACsrVal[i] = __float2half((float) ACsr->x[i]);
    }
    HBx = new __half2[(K*N)/2]();
    for (int i = 0; i < (K * N)/2; ++i) {
      HBx[i] = __float2half2_rn(1); //(double)rand()/RAND_MAX;
    }
    cudaMemcpy(DACsrAp, ACsr->p, rPtrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(DACsrI, ACsr->i, cIndexSize, cudaMemcpyHostToDevice);
    cudaMemcpy(DACsrVal, HACsrVal, valSize, cudaMemcpyHostToDevice);
    cudaMemcpy(DBx, HBx, denseSize, cudaMemcpyHostToDevice);
  }

  ~CudaTensorInputsFP16(){
    delete[] HACsrVal;
    delete[] HBx;
    cudaFree(DACsrAp);
    cudaFree(DACsrI);
    cudaFree(DACsrVal);
    cudaFree(DBx);
  }
};

struct CudaTensorOutputsFP16{
  int M,N,L;
  __half2* HACx;
  __half2* HXx;
  __half2* DACx;
  __half2* DXx;

  CudaTensorOutputsFP16(int M, int N, int L): M(M), N(N), L(L){
    HXx = new __half2[(L * N)/2]();
    HACx = new __half2[(M * N)/2]();
    size_t aCxSize = (M * N)/2 * sizeof(__half2);
    size_t xxSize = (M * N)/2 * sizeof(__half2);
    for (int i = 0; i < (M * N)/2; i++){
      HXx[i] = __float2half2_rn(0.0);
    }
    for (int i = 0; i < (M * N)/2; i++){
      HACx[i] = __float2half2_rn(0.0);
    }
    cudaMalloc(&DACx, aCxSize);
    cudaMalloc(&DXx, xxSize);
    //    cudaMemcpy(DACx, ACx, aCxSize, cudaMemcpyHostToDevice);
  }

  void copyDeviceToHost(){
    size_t aCxSize = M * N * sizeof(__half);
    size_t xxSize = L * N * sizeof(__half);
    cudaMemcpy(HACx, DACx, aCxSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(HXx, DXx, xxSize, cudaMemcpyDeviceToHost);
  }

  void reset(){
    std::fill_n(HXx, (L * N)/2, __float2half2_rn(0.0));
    std::fill_n(HACx, (M * N)/2, __float2half2_rn(0.0));
    cudaMemset(DACx, 0, (M * N/2) * sizeof(__half2));
    cudaMemset(DXx, 0, (L * N/2) * sizeof(__half2));
  }

  ~CudaTensorOutputsFP16(){
    delete [] HACx;
    delete [] HXx;
    cudaFree(DACx);
    cudaFree(DXx);
  }
};

class SpMMSpMMSeqReduceRowBalanceFP16 : public SWTensorBench<float> {
protected:
  CudaTensorInputsFP16 *InTensor;
  int MGridDim;
  int NGridDim;
  int MBlockDim;
  int NBlockDim;
  int ThreadPerBlock = 128;

  void setup() override {
    this->St->OtherStats["Number of Fused Rows"] = {0.};
    this->St->OtherStats["Number of Fused Nnz"] = {0.};
    NGridDim = CEIL(InTensor->N/2, ThreadPerBlock);
    NBlockDim = MIN(InTensor->N/2, ThreadPerBlock);
    MBlockDim = CEIL(ThreadPerBlock, NBlockDim);
    MGridDim = CEIL(InTensor->M, MBlockDim);
  }

  void preExecute() override {}


  // TODO: I assumed that we have one sparse matrix(or two graph adj matrices).
  //  This should be fix for two sparse matrices with different dimensions.

  Timer execute() override {
    OutTensor->reset();
    Timer t1;
    dim3 gridDim(MGridDim, NGridDim, 1);
    dim3 blockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csrspmm_seqreduce_rowbalance_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N/2, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    // TODO: Since I know kernel calls are nonBlocking I used this. Is this the
    // right way?
    //  I might need to test without synchronization to make sure nonBlocking
    //  property of kernel calls is valid.
    cudaDeviceSynchronize();
    csrspmm_seqreduce_rowbalance_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N/2, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx);
    cudaDeviceSynchronize();
    t1.stopGPU("UnfusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
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
    for (int i = 0; i < InTensor->M * InTensor->N; i+=2) {
      if (std::abs(__low2float(OutTensor->HXx[i/2]) - InTensor->CorrectSol[i]) > infNorm) {
        infNorm = std::abs(__low2float(OutTensor->HXx[i/2]) - InTensor->CorrectSol[i]);
      }
      if (std::abs(__high2float(OutTensor->HXx[i/2]) - InTensor->CorrectSol[i+1]) > infNorm) {
        infNorm = std::abs(__high2float(OutTensor->HXx[i/2]) - InTensor->CorrectSol[i+1]);
      }
    }
    Error = (double)infNorm;
    if (infNorm > InTensor->Threshold) {
      retValue = false;
    }
    return retValue;
  }

public:
  CudaTensorOutputsFP16 *OutTensor;
  SpMMSpMMSeqReduceRowBalanceFP16(CudaTensorInputsFP16 *In1, Stats *Stat1)
      : SWTensorBench<float>(In1, Stat1) {
    OutTensor = new CudaTensorOutputsFP16(In1->M, In1->N, In1->L);
    InTensor = In1;
  }

  ~SpMMSpMMSeqReduceRowBalanceFP16() { delete OutTensor; }
};

class FusedSpMMSpMMSeqReduceRowBalanceFP16 : public SpMMSpMMSeqReduceRowBalanceFP16 {
protected:
  int *HFPtr;
  int *DFPtr;
  int *HFId;
  int *DFId;
  int *HUFPtr;
  int *DUFPtr;
  int UFDim;
  int UFMGridDim;

  Timer analysis() override {
    Timer t1;
    t1.start();
    std::vector<int> ufRows;
    std::vector<std::vector<int>> fRows(MGridDim);
    int rowTile = MBlockDim;
    int *ap = InTensor->ACsr->p;
    int *ai = InTensor->ACsr->i;
    int NnzCount = 0;
    for (int i = 0; i < InTensor->M; i += rowTile) {
      int t = i / rowTile;
      int end = MIN(i + rowTile, InTensor->M);
      for (int ii = i; ii < end; ii++) {
        if (ai[ap[ii]] < i || ai[ap[ii+1]-1] >= end) {
          ufRows.push_back(ii);
        }
        else {
          fRows[t].push_back(ii);
          NnzCount += ap[ii + 1] - ap[ii];
        }
      }
    }
    int ufCount = ufRows.size();
    int fIdCount = InTensor->M - ufCount;
    int fPtrCount = MGridDim + 1;
    HUFPtr = new int[ufCount];
    HFPtr = new int[fPtrCount];
    HFId = new int[fIdCount];
    for (int i = 0; i < ufRows.size(); i++) {
      HUFPtr[i] = ufRows[i];
    }
    HFPtr[0] = 0;
    for (int i = 0; i < fRows.size(); i++) {
      HFPtr[i + 1] = HFPtr[i] + fRows[i].size();
      for (int j = HFPtr[i]; j < HFPtr[i + 1]; j++) {
        HFId[j] = fRows[i][j - HFPtr[i]];
      }
    }
    this->St->OtherStats["Number of Fused Rows"] = {(double)fIdCount};
    this->St->OtherStats["Number of Fused Nnz"] = {(double)NnzCount};
    UFDim = ufCount;
    UFMGridDim = CEIL(UFDim, MBlockDim);
    cudaMalloc(&DUFPtr, ufCount * sizeof(int));
    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFPtr, HFPtr, fPtrCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFId, HFId, fIdCount * sizeof(int), cudaMemcpyHostToDevice);
    t1.stop();
    return t1;
  }

  Timer execute() override {
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, NGridDim, 1);
    dim3 ufBlockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<fGridDim, fBlockDim>>>(
        InTensor->M, InTensor->N/2, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    cudaDeviceSynchronize();
    csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<ufGridDim,
                                                           ufBlockDim>>>(
        UFDim, InTensor->N/2, InTensor->K, InTensor->DACsrAp, InTensor->DACsrI,
        InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx, DUFPtr);
    cudaDeviceSynchronize();
    t1.stopGPU("FusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMSeqReduceRowBalanceFP16(CudaTensorInputsFP16 *In1, Stats *Stat1)
      : SpMMSpMMSeqReduceRowBalanceFP16(In1, Stat1) {}

  ~FusedSpMMSpMMSeqReduceRowBalanceFP16() {
    delete[] HFPtr;
    delete[] HUFPtr;
    delete[] HFId;
    cudaFree(DFPtr);
    cudaFree(DUFPtr);
    cudaFree(DFId);
  }
};

class FusedSpMMSpMMSeqReduceRowBalanceReorderedFP16
    : public FusedSpMMSpMMSeqReduceRowBalanceFP16 {
protected:
  int *HROAp;
  int *HROAi;
  __half *HROAx;
  int *DROAp;
  int *DROAi;
  __half *DROAx;

  Timer analysis() override {
    Timer t1;
    t1.start();
    std::vector<int> ufRows;
    std::vector<std::vector<int>> fRows(MGridDim);
    int rowTile = MBlockDim;
    int *ap = InTensor->ACsr->p;
    int *ai = InTensor->ACsr->i;
    __half *ax = InTensor->HACsrVal;
    int NnzCount = 0;
    for (int i = 0; i < InTensor->M; i += rowTile) {
      int t = i / rowTile;
      int end = MIN(i + rowTile, InTensor->M);
      for (int ii = i; ii < end; ii++) {
        if (ai[ap[ii]] < i || ai[ap[ii+1]-1] >= end) {
          ufRows.push_back(ii);
        }
        else {
          fRows[t].push_back(ii);
          NnzCount += ap[ii + 1] - ap[ii];
        }
      }
    }
    int ufCount = ufRows.size();
    int uFNnzCount = InTensor->ACsr->nnz - NnzCount;
    int fIdCount = InTensor->M - ufCount;
    int fPtrCount = MGridDim + 1;
    HUFPtr = new int[ufCount];
    HROAp = new int[ufCount + 1];
    HROAi = new int[uFNnzCount];
    HROAx = new __half[uFNnzCount];
    HFPtr = new int[fPtrCount];
    HFId = new int[fIdCount];
    HROAp[0] = 0;
    for (int i = 0; i < ufRows.size(); i++) {
      HUFPtr[i] = ufRows[i];
      int rowNnzCount = ap[ufRows[i] + 1] - ap[ufRows[i]];
      HROAp[i + 1] = rowNnzCount + HROAp[i];
      for (int j = 0; j < rowNnzCount; j++) {
        HROAi[HROAp[i] + j] = ai[ap[ufRows[i]] + j];
        HROAx[HROAp[i] + j] = ax[ap[ufRows[i]] + j];
      }
    }
    HFPtr[0] = 0;
    for (int i = 0; i < fRows.size(); i++) {
      HFPtr[i + 1] = HFPtr[i] + fRows[i].size();
      for (int j = HFPtr[i]; j < HFPtr[i + 1]; j++) {
        HFId[j] = fRows[i][j - HFPtr[i]];
      }
    }
    this->St->OtherStats["Number of Fused Rows"] = {(double)fIdCount};
    this->St->OtherStats["Number of Fused Nnz"] = {(double)NnzCount};
    UFDim = ufCount;
    UFMGridDim = CEIL(UFDim, MBlockDim);
    cudaMalloc(&DUFPtr, ufCount * sizeof(int));
    cudaMalloc(&DROAp, (ufCount + 1) * sizeof(int));
    cudaMalloc(&DROAi, uFNnzCount * sizeof(int));
    cudaMalloc(&DROAx, uFNnzCount * sizeof(__half));
    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (ufCount + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, uFNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, uFNnzCount * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DFPtr, HFPtr, fPtrCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFId, HFId, fIdCount * sizeof(int), cudaMemcpyHostToDevice);
    t1.stop();
    return t1;
  }

  Timer execute() override {
    Timer t1;
    Timer t2;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, NGridDim, 1);
    dim3 ufBlockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<fGridDim, fBlockDim>>>(
        InTensor->M, InTensor->N/2, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    cudaDeviceSynchronize();
    //    t1.stopGPU("ROFusedTileSpMMSpMM");
    //    std::cout << "ROFusedTileSpMMSpMM: " << t1.printTimeCsv(0) <<
    //    std::endl; t2.startGPU();
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N/2, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    cudaDeviceSynchronize();
    t1.stopGPU("ROUnFusedTileSpMMSpMM");
    //    std::cout << "ROUnFusedTileSpMMSpMM: " << t2.printTimeCsv(0) <<
    //    std::endl;
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMSeqReduceRowBalanceReorderedFP16(CudaTensorInputsFP16 *In1, Stats *Stat1)
      : FusedSpMMSpMMSeqReduceRowBalanceFP16(In1, Stat1) {}

  ~FusedSpMMSpMMSeqReduceRowBalanceReorderedFP16() {
    delete[] HROAp;
    delete[] HROAi;
    delete[] HROAx;
    cudaFree(DROAp);
    cudaFree(DROAi);
    cudaFree(DROAx);
  }
};

class FusedSpMMSpMMHighFusionRatioFP16
    : public FusedSpMMSpMMSeqReduceRowBalanceReorderedFP16 {
protected:
  int ThreadWorkReps;
  int FusedThreadsPerBlock = 128;
  int RowTile;
  int UFMBlockDim;
  int UFNBlockDim;
  int UFNGridDim;

  void setup() override {
    SpMMSpMMSeqReduceRowBalanceFP16::setup();
    //    if (InTensor->N == 32){
    //      RowTile = 256;
    //    } else if (InTensor ->N == 64){
    //      RowTile = 128;
    //    } else if (InTensor ->N == 128){
    //      RowTile = 64;
    //    }
    UFNGridDim = CEIL(InTensor->N/2, ThreadPerBlock);
    UFNBlockDim = MIN(InTensor->N/2, ThreadPerBlock);
    UFMBlockDim = CEIL(ThreadPerBlock, UFNBlockDim);
    // assert that bCols >= 32 and bCols is a product of 32.
    NGridDim = CEIL(InTensor->N/2, FusedThreadsPerBlock);
    NBlockDim = MIN(InTensor->N/2, FusedThreadsPerBlock);
    MBlockDim = CEIL(FusedThreadsPerBlock, NBlockDim);
    MGridDim = CEIL(InTensor->M, RowTile);
    ThreadWorkReps = CEIL(RowTile, MBlockDim);
  }

  Timer analysis() override {
    Timer t1;
    t1.start();
    std::vector<int> ufRows;
    std::vector<std::vector<int>> fRows(MGridDim);
    int rowTile = RowTile;
    int *ap = InTensor->ACsr->p;
    int *ai = InTensor->ACsr->i;
    __half *ax = InTensor->HACsrVal;
    int NnzCount = 0;
    for (int i = 0; i < InTensor->M; i += rowTile) {
      int t = i / rowTile;
      int end = MIN(i + rowTile, InTensor->M);
      for (int ii = i; ii < end; ii++) {
        if (ai[ap[ii]] < i || ai[ap[ii+1]-1] >= end) {
          ufRows.push_back(ii);
        }
        else {
          fRows[t].push_back(ii);
          NnzCount += ap[ii + 1] - ap[ii];
        }
      }
    }
    int ufCount = ufRows.size();
    int uFNnzCount = InTensor->ACsr->nnz - NnzCount;
    int fIdCount = InTensor->M - ufCount;
    int fPtrCount = MGridDim + 1;
    HUFPtr = new int[ufCount];
    HROAp = new int[ufCount + 1];
    HROAi = new int[uFNnzCount];
    HROAx = new __half[uFNnzCount];
    HFPtr = new int[fPtrCount];
    HFId = new int[fIdCount];
    HROAp[0] = 0;
    for (int i = 0; i < ufRows.size(); i++) {
      HUFPtr[i] = ufRows[i];
      int rowNnzCount = ap[ufRows[i] + 1] - ap[ufRows[i]];
      HROAp[i + 1] = rowNnzCount + HROAp[i];
      for (int j = 0; j < rowNnzCount; j++) {
        HROAi[HROAp[i] + j] = ai[ap[ufRows[i]] + j];
        HROAx[HROAp[i] + j] = ax[ap[ufRows[i]] + j];
      }
    }
    HFPtr[0] = 0;
    for (int i = 0; i < fRows.size(); i++) {
      HFPtr[i + 1] = HFPtr[i] + fRows[i].size();
      for (int j = HFPtr[i]; j < HFPtr[i + 1]; j++) {
        HFId[j] = fRows[i][j - HFPtr[i]];
      }
    }
    this->St->OtherStats["Number of Fused Rows"] = {(double)fIdCount};
    this->St->OtherStats["Number of Fused Nnz"] = {(double)NnzCount};
    UFDim = ufCount;
    UFMGridDim = CEIL(UFDim, UFMBlockDim);
    cudaMalloc(&DUFPtr, ufCount * sizeof(int));
    cudaMalloc(&DROAp, (ufCount + 1) * sizeof(int));
    cudaMalloc(&DROAi, uFNnzCount * sizeof(int));
    cudaMalloc(&DROAx, uFNnzCount * sizeof(__half));
    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (ufCount + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, uFNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, uFNnzCount * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DFPtr, HFPtr, fPtrCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFId, HFId, fIdCount * sizeof(int), cudaMemcpyHostToDevice);
    t1.stop();
    return t1;
  }

  Timer execute() override {
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, UFNGridDim, 1);
    dim3 ufBlockDim(UFNBlockDim, UFMBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_multiplerow_seqreduce_rowbalance_kernel<<<fGridDim,
                                                            fBlockDim>>>(
        InTensor->M, InTensor->N/2, InTensor->K, ThreadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    cudaDeviceSynchronize();
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N/2, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    cudaDeviceSynchronize();
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }


public:
  FusedSpMMSpMMHighFusionRatioFP16(CudaTensorInputsFP16 *In1, Stats *Stat1,
                               int RowTile)
      : FusedSpMMSpMMSeqReduceRowBalanceReorderedFP16(In1, Stat1),
        RowTile(RowTile) {}
};

class SpMMSpMMSeqReduceRowBalanceCoarsenedRowFP16:
    public SpMMSpMMSeqReduceRowBalanceFP16{
protected:
  int RowTile;
  void setup() override {
    SpMMSpMMSeqReduceRowBalanceFP16::setup();
    NGridDim = CEIL(InTensor->N/2, ThreadPerBlock);
    NBlockDim = MIN(InTensor->N/2, ThreadPerBlock);
    MBlockDim = CEIL(ThreadPerBlock, NBlockDim);
    MGridDim = CEIL(InTensor->M, RowTile);
  }
  Timer execute() override {
    OutTensor->reset();
    int threadWorkReps = CEIL(RowTile, MBlockDim);
    Timer t1;
    dim3 gridDim(MGridDim, NGridDim, 1);
    dim3 blockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csrspmm_seqreduce_rowcoarsened_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N/2, InTensor->K, threadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    // TODO: Since I know kernel calls are nonBlocking I used this. Is this the
    // right way?
    //  I might need to test without synchronization to make sure nonBlocking
    //  property of kernel calls is valid.
    cudaDeviceSynchronize();
    csrspmm_seqreduce_rowcoarsened_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N/2, InTensor->K, threadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx);
    cudaDeviceSynchronize();
    t1.stopGPU("UnfusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  SpMMSpMMSeqReduceRowBalanceCoarsenedRowFP16(CudaTensorInputsFP16 *In1, Stats *Stat1,
                                          int RowTile1)
      : SpMMSpMMSeqReduceRowBalanceFP16(In1, Stat1), RowTile(RowTile1)
  {}
};

class FusedSpMMSpMMHighFusionRatio2LevelFP16: public FusedSpMMSpMMHighFusionRatioFP16{
protected:

  //TODO: Should not work for bcol=16
  Timer analysis() override {
    Timer t1;
    t1.start();
    std::vector<int> ufRows;
    std::vector<std::vector<int>> fRows(MGridDim);
    int rowTile = RowTile;
    int *ap = InTensor->ACsr->p;
    int *ai = InTensor->ACsr->i;
    __half *ax = InTensor->HACsrVal;
    int l1TilesNum = MBlockDim;
    int l1TileSize = ThreadWorkReps;
    int NnzCount = 0;
    for (int i = 0; i < InTensor->M; i += rowTile) {
      int t = i / rowTile;
      int end = MIN(i + rowTile, InTensor->M);
      for (int ii = i; ii < end; ii++) {
        if (ai[ap[ii]] < i || ai[ap[ii+1]-1] >= end) {
          ufRows.push_back(ii);
        }
        else {
          fRows[t].push_back(ii);
          NnzCount += ap[ii + 1] - ap[ii];
        }
      }
    }
    int ufCount = ufRows.size();
    int uFNnzCount = InTensor->ACsr->nnz - NnzCount;
    int fIdCount = InTensor->M - ufCount;
    int fPtrCount = MGridDim * (l1TilesNum+1) + 1;
    HUFPtr = new int[ufCount];
    HROAp = new int[ufCount + 1];
    HROAi = new int[uFNnzCount];
    HROAx = new __half[uFNnzCount];
    HFPtr = new int[fPtrCount];
    HFId = new int[fIdCount];
    HROAp[0] = 0;
    for (int i = 0; i < ufRows.size(); i++) {
      HUFPtr[i] = ufRows[i];
      int rowNnzCount = ap[ufRows[i] + 1] - ap[ufRows[i]];
      HROAp[i + 1] = rowNnzCount + HROAp[i];
      for (int j = 0; j < rowNnzCount; j++) {
        HROAi[HROAp[i] + j] = ai[ap[ufRows[i]] + j];
        HROAx[HROAp[i] + j] = ax[ap[ufRows[i]] + j];
      }
    }
    HFPtr[0] = 0;
    for (int i = 0; i < fRows.size(); i++) {
      std::vector<std::vector<int>> innerTiles(l1TilesNum+1);
      for (int k = 0; k < fRows[i].size(); k++){
        bool l1Fused = false;
        int row = fRows[i][k];
        for(int j = 0; j < l1TilesNum; j++){
          int s = i * RowTile + j * ThreadWorkReps;
          int e = i * RowTile + (j+1) * (ThreadWorkReps);
          //          std::cout << s << " " << e << std::endl;
          if (ai[ap[row]] >= s && ai[ap[row+1]-1] < e){
            innerTiles[j].push_back(row);
            l1Fused = true;
          }
        }
        if (!l1Fused){
          innerTiles[l1TilesNum].push_back(row);
        }
      }
      for(int j = 0; j < l1TilesNum+1; j++){
        int sp = i*(l1TilesNum+1) + j;
        HFPtr[sp+1] = HFPtr[sp] + innerTiles[j].size();
        for (int k = HFPtr[sp]; k < HFPtr[sp+1]; k++) {
          HFId[k] = innerTiles[j][k - HFPtr[sp]];
        }
      }
    }
    this->St->OtherStats["Number of Fused Rows"] = {(double)fIdCount};
    this->St->OtherStats["Number of Fused Nnz"] = {(double)NnzCount};
    UFDim = ufCount;
    UFMGridDim = CEIL(UFDim, MBlockDim);
    cudaMalloc(&DUFPtr, ufCount * sizeof(int));
    cudaMalloc(&DROAp, (ufCount + 1) * sizeof(int));
    cudaMalloc(&DROAi, uFNnzCount * sizeof(int));
    cudaMalloc(&DROAx, uFNnzCount * sizeof(__half));
    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (ufCount + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, uFNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, uFNnzCount * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DFPtr, HFPtr, fPtrCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFId, HFId, fIdCount * sizeof(int), cudaMemcpyHostToDevice);
    t1.stop();
    return t1;
  }

  Timer execute() override {
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, UFNGridDim, 1);
    dim3 ufBlockDim(UFNBlockDim, UFMBlockDim, 1);
    t1.startGPU();
    csr_2LfusedTile_multiplerow_seqreduce_rowbalance_kernel<<<fGridDim,
                                                              fBlockDim>>>(
        InTensor->M, InTensor->N/2, InTensor->K, ThreadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    cudaDeviceSynchronize();
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N/2, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    cudaDeviceSynchronize();
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMHighFusionRatio2LevelFP16(CudaTensorInputsFP16 *In1, Stats *Stat1,
                                     int RowTile)
      : FusedSpMMSpMMHighFusionRatioFP16(In1, Stat1, RowTile)
  {}

};

class FusedSpMMSpMMFusedParReduceFP16: public FusedSpMMSpMMHighFusionRatioFP16{
protected:
  int* HFAp;
  int* HFAi;
  __half* HFAx;
  int* DFAp;
  int* DFAi;
  __half* DFAx;


  //TODO: Complete the code and clean if necessary.
  Timer analysis() override {
    Timer t1;
    t1.start();
    std::vector<int> ufRows;
    std::vector<std::vector<int>> fRows(MGridDim);
    int rowTile = RowTile;
    int *ap = InTensor->ACsr->p;
    int *ai = InTensor->ACsr->i;
    __half *ax = InTensor->HACsrVal;
    int NnzCount = 0;
    std::vector<int> fusedPtr;
    std::vector<int> fusedRowInd;
    std::vector<__half> fusedRowVal;
    fusedPtr.push_back(0);
    int cntr = 0;
    for (int i = 0; i < InTensor->M; i += rowTile) {
      std::vector<std::vector<int>> colNnzRow(rowTile);
      std::vector<std::vector<__half>> colNnzVal(rowTile);
      int t = i / rowTile;
      int end = MIN(i + rowTile, InTensor->M);
      for (int ii = i; ii < end; ii++) {
        if (ai[ap[ii]] < i || ai[ap[ii+1]-1] >= end) {
          ufRows.push_back(ii);
        }
        else {
          fRows[t].push_back(ii);
          for (int j = ap[ii]; j < ap[ii+1]; j++){
            colNnzRow[ai[j]-i].push_back(ii);
            colNnzVal[ai[j]-i].push_back(ax[j]);
          }
          NnzCount += ap[ii + 1] - ap[ii];
        }
      }
      for (int ii = 0; ii < rowTile; ii++) {
        for(int j = 0; j < colNnzRow[ii].size(); j++){
          fusedRowInd.push_back(colNnzRow[ii][j]);
          fusedRowVal.push_back(colNnzVal[ii][j]);
          cntr++;
        }
        fusedPtr.push_back(cntr);
      }
    }

    //TODO:Copy fused packing vectors to packing arrays.
    int ufCount = ufRows.size();
    int fApCount = fusedPtr.size();
    int fNnzCount = fusedRowVal.size();
    int uFNnzCount = InTensor->ACsr->nnz - NnzCount;
    HFAp = new int[fApCount];
    HFAi = new int[fNnzCount];
    HFAx = new __half[fNnzCount];
    HUFPtr = new int[ufCount];
    HROAp = new int[ufCount + 1];
    HROAi = new int[uFNnzCount];
    HROAx = new __half[uFNnzCount];
    HFPtr = nullptr; //TODO: Fix these.
    HFId = nullptr;
    HROAp[0] = 0;
    for (int i = 0; i < ufRows.size(); i++) {
      HUFPtr[i] = ufRows[i];
      int rowNnzCount = ap[ufRows[i] + 1] - ap[ufRows[i]];
      HROAp[i + 1] = rowNnzCount + HROAp[i];
      for (int j = 0; j < rowNnzCount; j++) {
        HROAi[HROAp[i] + j] = ai[ap[ufRows[i]] + j];
        HROAx[HROAp[i] + j] = ax[ap[ufRows[i]] + j];
      }
    }
    //    HFPtr[0] = 0;
    //    for (int i = 0; i < fRows.size(); i++) {
    //      HFPtr[i + 1] = HFPtr[i] + fRows[i].size();
    //      for (int j = HFPtr[i]; j < HFPtr[i + 1]; j++) {
    //        HFId[j] = fRows[i][j - HFPtr[i]];
    //      }
    //    }
    for(int i = 0; i < fApCount; i++){
      HFAp[i] = fusedPtr[i];
    }
    for(int i = 0; i < fNnzCount; i++){
      HFAi[i] = fusedRowInd[i];
      HFAx[i] = fusedRowVal[i];
    }
    this->St->OtherStats["Number of Fused Rows"] = {(double)fApCount};
    this->St->OtherStats["Number of Fused Nnz"] = {(double)NnzCount};
    UFDim = ufCount;
    UFMGridDim = CEIL(UFDim, MBlockDim);
    cudaMalloc(&DUFPtr, ufCount * sizeof(int));
    cudaMalloc(&DROAp, (ufCount + 1) * sizeof(int));
    cudaMalloc(&DROAi, uFNnzCount * sizeof(int));
    cudaMalloc(&DROAx, uFNnzCount * sizeof(__half));
    //    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    //    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMalloc(&DFAp, fApCount * sizeof(int));
    cudaMalloc(&DFAi, fNnzCount * sizeof(int));
    cudaMalloc(&DFAx, fNnzCount * sizeof(__half));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (ufCount + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, uFNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, uFNnzCount * sizeof(__half),
               cudaMemcpyHostToDevice);
    //    cudaMemcpy(DFPtr, HFPtr, fPtrCount * sizeof(int), cudaMemcpyHostToDevice);
    //    cudaMemcpy(DFId, HFId, fIdCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFAp, HFAp, fApCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFAi, HFAi, fNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFAx, HFAx, fNnzCount * sizeof(__half), cudaMemcpyHostToDevice);
    t1.stop();
    return t1;
  }

  Timer execute() override {
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, UFNGridDim, 1);
    dim3 ufBlockDim(UFNBlockDim, UFMBlockDim, 1);
    OutTensor->reset();
    t1.startGPU();
    csr_fusedTile_multiplerow_fusedParReduce_rowbalance_kernel<<<fGridDim,
                                                                 fBlockDim>>>(
        InTensor->M, InTensor->N/2, InTensor->K, ThreadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFAp, DFAi, DFAx);
    cudaDeviceSynchronize();
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N/2, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    cudaDeviceSynchronize();
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:

  FusedSpMMSpMMFusedParReduceFP16(CudaTensorInputsFP16 *In1, Stats *Stat1,
                              int RowTile)
      : FusedSpMMSpMMHighFusionRatioFP16(In1, Stat1, RowTile){}

  ~FusedSpMMSpMMFusedParReduceFP16(){
    delete[] HFAp;
    delete[] HFAi;
    delete[] HFAx;
    cudaFree(DFAp);
    cudaFree(DFAi);
    cudaFree(DFAx);
  }
};


class FusedSpMMSpMMCSRCSCFP16: public SpMMSpMMSeqReduceRowBalanceFP16{
protected:

  Timer execute() override {
    OutTensor->reset();
    Timer t1;
    dim3 gridDim(MGridDim, NGridDim, 1);
    dim3 blockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_multiplerow_1v1fusedParReduceAtomic_rowbalance_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N/2, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal);
    t1.stopGPU("UnfusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMCSRCSCFP16(CudaTensorInputsFP16 *In1, Stats *Stat1)
      : SpMMSpMMSeqReduceRowBalanceFP16(In1, Stat1) {}
};


#endif // SPARSE_FUSION_CUDA_SPMM_SPMM_FP16_DEMO_UTILS_H



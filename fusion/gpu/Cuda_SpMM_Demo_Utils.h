//
// Created by salehm32 on 17/06/24.
//

#ifndef SPARSE_FUSION_CUDA_SPMM_DEMO_UTILS_H
#define SPARSE_FUSION_CUDA_SPMM_DEMO_UTILS_H

#include "Stats.h"
#include "../example/SpMM_SpMM_Demo_Utils.h"
#include "SpMM_Kernels.h"
#include "Timer.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Defs.h"
#include "sparse-fusion/Fusion_Utils.h"

using namespace sym_lib;

struct CudaTensorInputs: public TensorInputs<float>{
  int* DACsrAp;
  int* DACsrI;
  float* DACsrVal;
  float* HACsrVal;

  float* DBx;

  // Don't Forget to delete the output whenever used.
  int* getCOORowIndForDevice(){
    int* hARowInd = new int [ACsr->nnz];
    int * dARowInd;
    size_t rowIndSize = ACsr->nnz * sizeof(int);
    cudaMalloc(&dARowInd, rowIndSize);
    for(int i = 0; i < ACsr->m; i++){
      for(int j = ACsr->p[i]; j < ACsr->p[i+1]; j++){
        hARowInd[j] = i;
      }
    }
    cudaMemcpy(dARowInd, hARowInd, rowIndSize, cudaMemcpyHostToDevice);
    return dARowInd;
  }

  CudaTensorInputs(int M1, int N1, int K1, int L1, sym_lib::CSC *A1,
                   sym_lib::CSC *B1, int NumThreads1, int NumTrial1,
                   std::string ExpN): TensorInputs<float>(M1, N1, K1, L1, A1, B1, NumThreads1, NumTrial1, ExpN){

    size_t rPtrSize = (ACsr->m + 1) * sizeof(int);
    size_t cIndexSize = ACsr->nnz * sizeof(int);
    size_t valSize = ACsr->nnz * sizeof(float);
    size_t denseSize = K * N * sizeof(float);
    cudaMalloc(&DACsrAp, rPtrSize);
    cudaMalloc(&DACsrI, cIndexSize);
    cudaMalloc(&DACsrVal, valSize);
    cudaMalloc(&DBx, denseSize);
    HACsrVal = new float[ACsr->nnz];
    for (int i = 0; i < ACsr->nnz; i++){
      HACsrVal[i] = (float) ACsr->x[i];
    }
    cudaMemcpy(DACsrAp, ACsr->p, rPtrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(DACsrI, ACsr->i, cIndexSize, cudaMemcpyHostToDevice);
    cudaMemcpy(DACsrVal, HACsrVal, valSize, cudaMemcpyHostToDevice);
    cudaMemcpy(DBx, Bx, denseSize, cudaMemcpyHostToDevice);
  }

  ~CudaTensorInputs(){
    delete[] HACsrVal;
    cudaFree(DACsrAp);
    cudaFree(DACsrI);
    cudaFree(DACsrVal);
    cudaFree(DBx);
  }
};

struct CudaTensorOutputs: public TensorOutputs<float>{

  float* DACx;
  float* DXx;

  CudaTensorOutputs(int M, int N, int L) : TensorOutputs<float>(M, N, L) {
    size_t aCxSize = M * N * sizeof(float);
    size_t xxSize = M * N * sizeof(float);
    cudaMalloc(&DACx, aCxSize);
    cudaMalloc(&DXx, xxSize);
//    cudaMemcpy(DACx, ACx, aCxSize, cudaMemcpyHostToDevice);
  }

  void copyDeviceToHost(){
    size_t aCxSize = M * N * sizeof(float);
    size_t xxSize = L * N * sizeof(float);
    cudaMemcpy(ACx, DACx, aCxSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(Xx, DXx, xxSize, cudaMemcpyDeviceToHost);
  }

  void reset() override{
    std::fill_n(Xx, L * N, 0.0);
    std::fill_n(ACx, M * N, 0.0);
    cudaMemset(DACx, 0, M * N * sizeof(float));
    cudaMemset(DXx, 0, L * N * sizeof(float));
  }

  ~CudaTensorOutputs(){
    cudaFree(DACx);
    cudaFree(DXx);
  }
};


class CpuSpMM : public SWTensorBench<float> {
protected:

  CudaTensorInputs *InTensor;

  void setup() override {}

  void preExecute() override {}

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSequential<float>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->HACsrVal, InTensor->Bx, OutTensor->ACx);
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
    // TODO: Later on I need to have separate classes for SpMM and SpMM-SpMM or think of an other way.
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
  CpuSpMM(CudaTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<float>(In1, Stat1) {
    OutTensor = new CudaTensorOutputs(In1->M, In1->N, In1->L);
    InTensor = In1;
  }

  ~CpuSpMM(){
    delete OutTensor;
  }

};

class GpuGeSpMM: public CpuSpMM{

  Timer execute() override {
    OutTensor->reset();
    Timer t;
    t.startGPU();
    csrGeSpMM(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    t.stopGPU("GpSpMM");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  GpuGeSpMM(CudaTensorInputs *In1, Stats* Stat1)
      : CpuSpMM(In1, Stat1){}
};

class GpuParReduceRowBalance: public CpuSpMM{

  Timer execute() override {
    OutTensor->reset();
    Timer t;
    t.startGPU();
    csrspmm_parreduce_rowbalance(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    t.stopGPU("csrspmm_parreduce_rowbalance");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  GpuParReduceRowBalance(CudaTensorInputs *In1, Stats* Stat1)
      : CpuSpMM(In1, Stat1){}
};

class GpuParReduceNnzBalance: public CpuSpMM{

  Timer execute() override {
    OutTensor->reset();
    Timer t;
    t.startGPU();
    csrspmm_parreduce_nnzbalance(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->nnz, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    t.stopGPU("csrspmm_parreduce_nnzbalance");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  GpuParReduceNnzBalance(CudaTensorInputs *In1, Stats* Stat1)
      : CpuSpMM(In1, Stat1){}
};

class GpuSeqReduceRowBalance: public CpuSpMM{

  Timer execute() override {
    OutTensor->reset();
    Timer t;
    t.startGPU();
    csrspmm_seqreduce_rowbalance(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    t.stopGPU("csrspmm_seqreduce_rowbalance");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  GpuSeqReduceRowBalance(CudaTensorInputs *In1, Stats* Stat1)
      : CpuSpMM(In1, Stat1){}
};


class GpuSeqReduceRowBalanceVariableThreadPerBlock: public CpuSpMM{
protected:
  int ThreadsPerBlock;
  int MGridDim;
  int NGridDim;
  int MBlockDim;
  int NBlockDim;

  void setup() override {
    NGridDim = CEIL(InTensor->N, ThreadsPerBlock);
    NBlockDim = MIN(InTensor->N, ThreadsPerBlock);
    MBlockDim = CEIL(ThreadsPerBlock, NBlockDim);
    MGridDim = CEIL(InTensor->M, MBlockDim);
  }

  Timer execute() override{
    OutTensor->reset();
    dim3 gridDim(MGridDim, NGridDim, 1);
    dim3 blockDim(NBlockDim, MBlockDim, 1);
    Timer t1;
    t1.startGPU();
    csrspmm_seqreduce_rowbalance_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    t1.stopGPU("spmm_seqreduce_rowbalance_tb_" + std::to_string(ThreadsPerBlock));
    OutTensor->copyDeviceToHost();
    return t1;
  }
public:
  GpuSeqReduceRowBalanceVariableThreadPerBlock(CudaTensorInputs *In1, Stats* Stat1, int ThreadsPerBlock)
      : CpuSpMM(In1, Stat1), ThreadsPerBlock(ThreadsPerBlock) {}
};

class GpuSeqReduceNnzBalance: public CpuSpMM{

  Timer execute() override {
    OutTensor->reset();
    Timer t;
    t.startGPU();
    csrspmm_seqreduce_nnzbalance(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->nnz, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    t.stopGPU("csrspmm_seqreduce_nnzbalance");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  GpuSeqReduceNnzBalance(CudaTensorInputs *In1, Stats* Stat1)
      : CpuSpMM(In1, Stat1){}
};

class GpuRowCachingRowBalance: public CpuSpMM{

  Timer execute() override {
    OutTensor->reset();
    Timer t;
    t.startGPU();
    csrspmm_rowcaching_rowbalance(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    t.stopGPU("csrspmm_rowcaching_rowbalance");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  GpuRowCachingRowBalance(CudaTensorInputs *In1, Stats* Stat1)
      : CpuSpMM(In1, Stat1){}
};

class GpuRowCachingNnzBalance : public CpuSpMM{

  Timer execute() override {
    OutTensor->reset();
    Timer t;
    t.startGPU();
    csrspmm_rowcaching_nnzbalance(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->nnz, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    t.stopGPU("csrspmm_rowcaching_nnzbalance");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  GpuRowCachingNnzBalance(CudaTensorInputs *In1, Stats* Stat1)
      : CpuSpMM(In1, Stat1){}
};


//class GpuHPSpMM: public CpuSpMM{
//
//  Timer execute() override {
//    OutTensor->reset();
//    Timer t;
//    t.startGPU();
//    cooLBSpMM(
//        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
//        InTensor->ACsr->i,  InTensor->ACsr->nnz, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
//    t.stopGPU("GpSpMM");
//    OutTensor->copyDeviceToHost();
//    return t;
//  }
//
//public:
//  GpuHPSpMM(CudaTensorInputs *In1, Stats* Stat1)
//      : CpuSpMM(In1, Stat1){}
//};

class GpuSpMMCuSparse : public CpuSpMM {
protected:
  void *Workspace;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  float alpha = 1.0;
  float beta = 0.0;
  cusparseSpMMAlg_t Alg;
  cusparseHandle_t cusparse_handle = 0;

  void setup() override {
    cusparseCreateCsr(&matA,
                      InTensor->M, InTensor->K, InTensor->ACsr->nnz,
                      InTensor->DACsrAp,
                      InTensor->DACsrI,
                      InTensor->DACsrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB,
                        InTensor->K, InTensor->N, InTensor->N, InTensor->DBx, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC,
                        InTensor->M, InTensor->N, InTensor->N, OutTensor->DACx, CUDA_R_32F, CUSPARSE_ORDER_ROW);
  }

  Timer analysis() override {
    //    if(algid == -1){
    //      Alg = CUSPARSE_SPMM_ALG_DEFAULT;
    //    } else if(algid == 2){
    //      Alg = CUSPARSE_SPMM_CSR_ALG2;
    //    } else if (algid == 3) {
    //      Alg = CUSPARSE_SPMM_CSR_ALG3;
    //    }
    cusparseCreate(&cusparse_handle);
    Timer t;
    t.startGPU();
    size_t workspaceSize;
    cusparseSpMM_bufferSize(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, Alg,
        &workspaceSize);
    cudaMalloc(&Workspace, workspaceSize);
    t.stopGPU("CuSparseSpMM_CSR_workspace");
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.startGPU();
    cusparseSpMM(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, Alg,
        Workspace);
    t.stopGPU("CuSparseSpMM_CSR");
    OutTensor->copyDeviceToHost();
    return t;
  }


public:
  GpuSpMMCuSparse(CudaTensorInputs *In1, Stats *Stat1, cusparseSpMMAlg_t Alg1)
      : CpuSpMM(In1, Stat1), Alg(Alg1) {}

  ~GpuSpMMCuSparse(){
    cudaFree(Workspace);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
  }
};

// Only Works With Alg3 for CSR
class GpuSpMMCuSparsePreProcessing : public GpuSpMMCuSparse {
  Timer analysis() override {
    Timer t;
    t.startGPU();
    cusparseCreate(&cusparse_handle);
    size_t workspaceSize;
    cusparseSpMM_bufferSize(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, Alg,
        &workspaceSize);
    cudaMalloc(&Workspace, workspaceSize);
//    cusparseSpMM_preprocess(cusparse_handle, transA, transB,
//                            &alpha, matA, matB, &beta, matC,
//                            CUDA_R_32F, Alg,
//                            Workspace);
    t.stopGPU("CuSparseSpMM_CSR_preprocess");
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.startGPU();
    cusparseSpMM(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, Alg,
        Workspace);
    t.stopGPU("CuSparseSpMM_WPP_CSR");
    OutTensor->copyDeviceToHost();
    return t;
  }
public:
  GpuSpMMCuSparsePreProcessing(CudaTensorInputs *In1, Stats *Stat1)
    : GpuSpMMCuSparse(In1, Stat1, CUSPARSE_SPMM_CSR_ALG3){}
};

#endif // SPARSE_FUSION_CUDA_SPMM_DEMO_UTILS_H
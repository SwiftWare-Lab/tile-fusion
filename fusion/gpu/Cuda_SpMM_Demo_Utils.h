//
// Created by salehm32 on 17/06/24.
//

#ifndef SPARSE_FUSION_CUDA_SPMM_DEMO_UTILS_H
#define SPARSE_FUSION_CUDA_SPMM_DEMO_UTILS_H

#include "../example/SpMM_SpMM_Demo_Utils.h"
#include "SpMM_Kernels.h"
#include "Timer.h"
#endif // SPARSE_FUSION_CUDA_SPMM_DEMO_UTILS_H


struct CudaTensorInputs: public TensorInputs<float>{
  int* DACsrAp;
  int* DACsrI;
  float* DACsrVal;
  float* HACsrVal;


  float* DBx;

  CudaTensorInputs(int M1, int N1, int K1, int L1, sym_lib::CSC *A1,
                   sym_lib::CSC *B1, int NumThreads1, int NumTrial1,
                   std::string ExpN): TensorInputs<float>(M1, N1, K1, L1, A1, B1, NumThreads1, NumTrial1, ExpN){
    size_t rPtrSize = ACsr->m * sizeof(int);
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

  CudaTensorOutputs(int M, int N, int L) : TensorOutputs<float>(M, N, L) {
    size_t outputSize = M * N * sizeof(float);
    cudaMalloc(&DACx, outputSize);
//    cudaMemcpy(DACx, ACx, outputSize, cudaMemcpyHostToDevice);
  }

  void copyDeviceToHost(){
    size_t outputSize = M * N * sizeof(float);
    cudaMemcpy(ACx, DACx, outputSize, cudaMemcpyDeviceToHost);
  }

  void reset() override{
    std::fill_n(Xx, L * N, 0.0);
    std::fill_n(ACx, M * N, 0.0);
    cudaMemset(DACx, 0, M * N * sizeof(float));
  }

  ~CudaTensorOutputs(){
    cudaFree(DACx);
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

};

class GpuGeSpMM: public CpuSpMM{

  Timer execute() override {
    OutTensor->reset();
    Timer t;
    t.startGPU();
    csrGeSpMM(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->HACsrVal, InTensor->Bx, OutTensor->ACx);
    t.stopGPU("GpSpMM");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  GpuGeSpMM(CudaTensorInputs *In1, Stats* Stat1)
      : CpuSpMM(In1, Stat1){}
};

class GpuSpMMCuSparse : public CpuSpMM {
protected:
  void *Workspace;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  float alpha = 1.0;
  float beta = 0.0;
  cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;
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
    //      alg = CUSPARSE_SPMM_ALG_DEFAULT;
    //    } else if(algid == 2){
    //      alg = CUSPARSE_SPMM_CSR_ALG2;
    //    } else if (algid == 3) {
    //      alg = CUSPARSE_SPMM_CSR_ALG3;
    //    }
    cusparseCreate(&cusparse_handle);
    Timer t;
    t.startGPU();
    size_t workspaceSize;
    cusparseSpMM_bufferSize(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, alg,
        &workspaceSize);
    cudaMalloc(&Workspace, workspaceSize);
    t.stopGPU("CuSparseSpMM_CSR_workspace");
    return t;
  }

//  Timer analysis() override {
//
//  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.startGPU();
    cusparseSpMM(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, alg,
        Workspace);
    t.stopGPU("CuSparseSpMM_CSR");
    OutTensor->copyDeviceToHost();
    return t;
  }


public:
  GpuSpMMCuSparse(CudaTensorInputs *In1, Stats *Stat1)
      : CpuSpMM(In1, Stat1) {}
};


class GpuSpMMCuSparsePreProcessing : public GpuSpMMCuSparse {
  Timer analysis() override {
    alg = CUSPARSE_SPMM_CSR_ALG3;
    Timer t;
    t.startGPU();
    cusparseCreate(&cusparse_handle);
    size_t workspaceSize;
    cusparseSpMM_bufferSize(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, alg,
        &workspaceSize);
    cudaMalloc(&Workspace, workspaceSize);
    cusparseSpMM_preprocess(cusparse_handle, transA, transB,
                            &alpha, matA, matB, &beta, matC,
                            CUDA_R_32F, alg,
                            Workspace);
    t.stopGPU("CuSparseSpMM_CSR_preprocess");
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    alg = CUSPARSE_SPMM_CSR_ALG3;
    OutTensor->reset();
    Timer t;
    t.startGPU();
    cusparseSpMM(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, alg,
        Workspace);
    t.stopGPU("CuSparseSpMM_WPP_CSR");
    OutTensor->copyDeviceToHost();
    return t;
  }
public:
  GpuSpMMCuSparsePreProcessing(CudaTensorInputs *In1, Stats *Stat1)
    : GpuSpMMCuSparse(In1, Stat1){}
};
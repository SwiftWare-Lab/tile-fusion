//
// Created by salehm32 on 17/06/24.
//

#ifndef SPARSE_FUSION_CUDA_SPMM_DEMO_UTILS_H
#define SPARSE_FUSION_CUDA_SPMM_DEMO_UTILS_H

#include "../example/SpMM_SpMM_Demo_Utils.h"

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

class GpuSpMMCuBlas : public SWTensorBench<float> {
protected:
  CudaTensorInputs *InTensor;
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

  void preExecute() override {
    int algid = -1;
    size_t workspaceSize;
    //    if(algid == -1){
    //      alg = CUSPARSE_SPMM_ALG_DEFAULT;
    //    } else if(algid == 2){
    //      alg = CUSPARSE_SPMM_CSR_ALG2;
    //    } else if (algid == 3) {
    //      alg = CUSPARSE_SPMM_CSR_ALG3;
    //    }
    cusparseCreate(&cusparse_handle);
    cusparseSpMM_bufferSize(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, alg,
        &workspaceSize);

    cudaMalloc(&Workspace, workspaceSize);
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
    t.stopGPU(St->OperationName);
    OutTensor->copyDeviceToHost();
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
  GpuSpMMCuBlas(CudaTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<float>(In1, Stat1) {
    OutTensor = new CudaTensorOutputs(In1->M, In1->N, In1->L);
    InTensor = In1;
  }

  ~GpuSpMMCuBlas() { delete OutTensor; }
};
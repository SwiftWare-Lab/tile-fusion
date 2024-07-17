//
// Created by salehm32 on 20/06/24.
//

#ifndef SPARSE_FUSION_CUDA_SPMM_SPMM_DEMO_UTILS_H
#define SPARSE_FUSION_CUDA_SPMM_SPMM_DEMO_UTILS_H
#include "Cuda_SpMM_Demo_Utils.h"
//#include "SW_SpMM_Kernels.h"

#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(a, b) ((a < b) ? a : b)

class SeqSpMMSpMM : public SWTensorBench<float> {
protected:
  CudaTensorInputs *InTensor;

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
    swiftware::sparse::spmmCsrSequential<float>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->HACsrVal, InTensor->Bx, OutTensor->ACx);
    swiftware::sparse::spmmCsrSequential<float>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->HACsrVal, OutTensor->ACx, OutTensor->Xx);
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
  SeqSpMMSpMM(CudaTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<float>(In1, Stat1) {
    OutTensor = new CudaTensorOutputs(In1->M, In1->N, In1->L);
    InTensor = In1;
  }

  ~SeqSpMMSpMM() { delete OutTensor; }
};

class SpMMSpMMCuSparse : public SeqSpMMSpMM {
protected:
  void *WorkspaceMul1;
  void *WorkspaceMul2;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC, matD;
  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  float alpha = 1.0;
  float beta = 0.0;
  cusparseSpMMAlg_t Alg;
  cusparseHandle_t cusparse_handle = 0;

  void setup() override {
    SeqSpMMSpMM::setup();
    cusparseCreateCsr(&matA, InTensor->M, InTensor->K, InTensor->ACsr->nnz,
                      InTensor->DACsrAp, InTensor->DACsrI, InTensor->DACsrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, InTensor->K, InTensor->N, InTensor->N,
                        InTensor->DBx, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, InTensor->M, InTensor->N, InTensor->N,
                        OutTensor->DACx, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matD, InTensor->L, InTensor->N, InTensor->N,
                        OutTensor->DXx, CUDA_R_32F, CUSPARSE_ORDER_ROW);
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
    size_t workspace1Size;
    cusparseSpMM_bufferSize(cusparse_handle, transA, transB, &alpha, matA, matB,
                            &beta, matC, CUDA_R_32F, Alg, &workspace1Size);
    cudaMalloc(&WorkspaceMul1, workspace1Size);
    size_t workspace2Size;
    cusparseSpMM_bufferSize(cusparse_handle, transA, transB, &alpha, matA, matC,
                            &beta, matD, CUDA_R_32F, Alg, &workspace2Size);
    cudaMalloc(&WorkspaceMul2, workspace2Size);
    t.stopGPU("CuSparseSpMMSpMM_CSR_workspace");
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.startGPU();
    cusparseSpMM(cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
                 matC, CUDA_R_32F, Alg, WorkspaceMul1);
    cudaDeviceSynchronize();
    cusparseSpMM(cusparse_handle, transA, transB, &alpha, matA, matC, &beta,
                 matD, CUDA_R_32F, Alg, WorkspaceMul2);
    t.stopGPU("CuSparseSpMMSpMM_CSR");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  SpMMSpMMCuSparse(CudaTensorInputs *In1, Stats *Stat1, cusparseSpMMAlg_t Alg1)
      : SeqSpMMSpMM(In1, Stat1), Alg(Alg1) {}

  ~SpMMSpMMCuSparse() {
    cudaFree(WorkspaceMul1);
    cudaFree(WorkspaceMul2);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroyDnMat(matD);
  }
};

class SpMMSpMMSeqReduceRowBalance : public SeqSpMMSpMM {
protected:
  int MGridDim;
  int NGridDim;
  int MBlockDim;
  int NBlockDim;
  int ThreadPerBlock;

  // TODO: I assumed that we have one sparse matrix(or two graph adj matrices).
  //  This should be fix for two sparse matrices with different dimensions.
  void setup() override {
    SeqSpMMSpMM::setup();
    NGridDim = CEIL(InTensor->N, ThreadPerBlock);
    NBlockDim = MIN(InTensor->N, ThreadPerBlock);
    MBlockDim = CEIL(ThreadPerBlock, NBlockDim);
    MGridDim = CEIL(InTensor->M, MBlockDim);
  }

  Timer execute() override{
    OutTensor->reset();
    Timer t1;
    dim3 gridDim(MGridDim, NGridDim, 1);
    dim3 blockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csrspmm_seqreduce_rowbalance_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    // TODO: Since I know kernel calls are nonBlocking I used this. Is this the
    // right way?
    //  I might need to test without synchronization to make sure nonBlocking
    //  property of kernel calls is valid.
    cudaDeviceSynchronize();
    csrspmm_seqreduce_rowbalance_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx);
    cudaDeviceSynchronize();
    t1.stopGPU("UnfusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }
public:
  SpMMSpMMSeqReduceRowBalance(CudaTensorInputs *In1, Stats *Stat1,
                              int ThreadPerBlock = 256)
      : SeqSpMMSpMM(In1, Stat1), ThreadPerBlock(ThreadPerBlock) {
  }
};

// only for tri-banded now.
class FusedSpMMSpMMSeqReduceRowBalance : public SpMMSpMMSeqReduceRowBalance {
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
    int* ap = InTensor->ACsr->p;
    int* ai = InTensor->ACsr->i;
    int NnzCount = 0;
    for (int i = 0; i < InTensor->M; i+=rowTile) {
      int t = i / rowTile;
      int end = MIN(i + rowTile, InTensor->M);
      for (int ii = i; ii < end; ii++){
        bool isUnfused = false;
        for (int j = ap[ii]; j < ap[ii + 1]; j++){
          if (ai[j] < i || ai[j] >= end){
            ufRows.push_back(ii);
            isUnfused = true;
            break;
          }
        }
        if (!isUnfused){
          fRows[t].push_back(ii);
          NnzCount += ap[ii+1] - ap[ii];
        }
      }
    }
    int ufCount = ufRows.size();
    int fIdCount = InTensor->M - ufCount;
    int fPtrCount = MGridDim + 1;
    HUFPtr = new int[ufCount];
    HFPtr = new int[fPtrCount];
    HFId = new int[fIdCount];
    for (int i = 0; i < ufRows.size(); i++){
      HUFPtr[i] = ufRows[i];
    }
    HFPtr[0] = 0;
    for (int i = 0; i < fRows.size(); i++){
      HFPtr[i+1] = HFPtr[i] + fRows[i].size();
      for (int j = HFPtr[i]; j < HFPtr[i+1]; j++){
        HFId[j] = fRows[i][j-HFPtr[i]];
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

  Timer execute() override{
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, NGridDim, 1);
    dim3 ufBlockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<fGridDim, fBlockDim>>>
        (InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
         InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
         OutTensor->DXx, DFPtr, DFId);
    cudaDeviceSynchronize();
    csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<ufGridDim, ufBlockDim>>>
        (UFDim, InTensor->N, InTensor->K, InTensor->DACsrAp,
         InTensor->DACsrI, InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx,
         DUFPtr);
    cudaDeviceSynchronize();
    t1.stopGPU("FusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMSeqReduceRowBalance(CudaTensorInputs *In1, Stats *Stat1,
                                   int ThreadPerBlock = 256)
      : SpMMSpMMSeqReduceRowBalance(In1, Stat1, ThreadPerBlock) {}

  ~FusedSpMMSpMMSeqReduceRowBalance() {
    delete[] HFPtr;
    delete[] HUFPtr;
    delete[] HFId;
    cudaFree(DFPtr);
    cudaFree(DUFPtr);
    cudaFree(DFId);
  }
};


class FusedSpMMSpMMSeqReduceRowBalanceReordered: public FusedSpMMSpMMSeqReduceRowBalance
{
protected:
  int *HROAp;
  int *HROAi;
  float *HROAx;
  int *DROAp;
  int *DROAi;
  float *DROAx;

  Timer analysis() override {
    Timer t1;
    t1.start();
    std::vector<int> ufRows;
    std::vector<std::vector<int>> fRows(MGridDim);
    int rowTile = MBlockDim;
    int* ap = InTensor->ACsr->p;
    int* ai = InTensor->ACsr->i;
    float* ax = InTensor->HACsrVal;
    int NnzCount = 0;
    for (int i = 0; i < InTensor->M; i+=rowTile) {
      int t = i / rowTile;
      int end = MIN(i + rowTile, InTensor->M);
      for (int ii = i; ii < end; ii++){
        bool isUnfused = false;
        for (int j = ap[ii]; j < ap[ii + 1]; j++){
          if (ai[j] < i || ai[j] >= end){
            ufRows.push_back(ii);
            isUnfused = true;
            break;
          }
        }
        if (!isUnfused){
          fRows[t].push_back(ii);
          NnzCount += ap[ii+1] - ap[ii];
        }
      }
    }
    int ufCount = ufRows.size();
    int uFNnzCount = InTensor->ACsr->nnz - NnzCount;
    int fIdCount = InTensor->M - ufCount;
    int fPtrCount = MGridDim + 1;
    HUFPtr = new int[ufCount];
    HROAp = new int[ufCount+1];
    HROAi = new int[uFNnzCount];
    HROAx = new float[uFNnzCount];
    HFPtr = new int[fPtrCount];
    HFId = new int[fIdCount];
    HROAp[0] = 0;
    for (int i = 0; i < ufRows.size(); i++){
      HUFPtr[i] = ufRows[i];
      int rowNnzCount = ap[ufRows[i]+1] - ap[ufRows[i]];
      HROAp[i+1] = rowNnzCount + HROAp[i];
      for(int j = 0; j < rowNnzCount; j++){
        HROAi[HROAp[i]+j] = ai[ap[ufRows[i]] + j];
        HROAx[HROAp[i]+j] = ax[ap[ufRows[i]] + j];
      }
    }
    HFPtr[0] = 0;
    for (int i = 0; i < fRows.size(); i++){
      HFPtr[i+1] = HFPtr[i] + fRows[i].size();
      for (int j = HFPtr[i]; j < HFPtr[i+1]; j++){
        HFId[j] = fRows[i][j-HFPtr[i]];
      }
    }
    this->St->OtherStats["Number of Fused Rows"] = {(double)fIdCount};
    this->St->OtherStats["Number of Fused Nnz"] = {(double)NnzCount};
    UFDim = ufCount;
    UFMGridDim = CEIL(UFDim, MBlockDim);
    cudaMalloc(&DUFPtr, ufCount * sizeof(int));
    cudaMalloc(&DROAp, (ufCount+1) * sizeof(int));
    cudaMalloc(&DROAi, uFNnzCount * sizeof(int));
    cudaMalloc(&DROAx, uFNnzCount * sizeof(float));
    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (ufCount+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, uFNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, uFNnzCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DFPtr, HFPtr, fPtrCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFId, HFId, fIdCount * sizeof(int), cudaMemcpyHostToDevice);
    t1.stop();
    return t1;
  }

  Timer execute() override{
    Timer t1;
    Timer t2;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, NGridDim, 1);
    dim3 ufBlockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<fGridDim, fBlockDim>>>
        (InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
         InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
         OutTensor->DXx, DFPtr, DFId);
    cudaDeviceSynchronize();
//    t1.stopGPU("ROFusedTileSpMMSpMM");
//    std::cout << "ROFusedTileSpMMSpMM: " << t1.printTimeCsv(0) << std::endl;
//    t2.startGPU();
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<ufGridDim, ufBlockDim>>>
        (UFDim, InTensor->N, InTensor->K, DROAp,
         DROAi, DROAx, OutTensor->DACx, OutTensor->DXx,
         DUFPtr);
    cudaDeviceSynchronize();
    t1.stopGPU("ROUnFusedTileSpMMSpMM");
//    std::cout << "ROUnFusedTileSpMMSpMM: " << t2.printTimeCsv(0) << std::endl;
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMSeqReduceRowBalanceReordered(CudaTensorInputs *In1, Stats *Stat1,
                                   int ThreadPerBlock = 256)
      : FusedSpMMSpMMSeqReduceRowBalance(In1, Stat1, ThreadPerBlock) {}

  ~FusedSpMMSpMMSeqReduceRowBalanceReordered() {
    delete[] HROAp;
    delete[] HROAi;
    delete[] HROAx;
    cudaFree(DROAp);
    cudaFree(DROAi);
    cudaFree(DROAx);
  }
};


class FusedSpMMSpMMHighFusionRatio
    : public FusedSpMMSpMMSeqReduceRowBalanceReordered
{
  int RowPerThread;
  int RowTile;
protected:
  void setup() override {
    SeqSpMMSpMM::setup();
//    if (InTensor->N == 32){
//      RowTile = 256;
//    } else if (InTensor ->N == 64){
//      RowTile = 128;
//    } else if (InTensor ->N == 128){
//      RowTile = 64;
//    }
    //assert that bCols >= 32 and bCols is a product of 32.
    NGridDim = CEIL(InTensor->N, ThreadPerBlock);
    NBlockDim = MIN(InTensor->N, ThreadPerBlock);
    MBlockDim = CEIL(ThreadPerBlock, NBlockDim);
    MGridDim = CEIL(InTensor->M, RowTile);
    RowPerThread = CEIL(RowTile, MBlockDim);
  }


  Timer analysis() override {
    Timer t1;
    t1.start();
    std::vector<int> ufRows;
    std::vector<std::vector<int>> fRows(MGridDim);
    int rowTile = RowTile;
    int* ap = InTensor->ACsr->p;
    int* ai = InTensor->ACsr->i;
    float* ax = InTensor->HACsrVal;
    int NnzCount = 0;
    for (int i = 0; i < InTensor->M; i+=rowTile) {
      int t = i / rowTile;
      int end = MIN(i + rowTile, InTensor->M);
      for (int ii = i; ii < end; ii++){
        bool isUnfused = false;
        for (int j = ap[ii]; j < ap[ii + 1]; j++){
          if (ai[j] < i || ai[j] >= end){
            ufRows.push_back(ii);
            isUnfused = true;
            break;
          }
        }
        if (!isUnfused){
          fRows[t].push_back(ii);
          NnzCount += ap[ii+1] - ap[ii];
        }
      }
    }
    int ufCount = ufRows.size();
    int uFNnzCount = InTensor->ACsr->nnz - NnzCount;
    int fIdCount = InTensor->M - ufCount;
    int fPtrCount = MGridDim + 1;
    HUFPtr = new int[ufCount];
    HROAp = new int[ufCount+1];
    HROAi = new int[uFNnzCount];
    HROAx = new float[uFNnzCount];
    HFPtr = new int[fPtrCount];
    HFId = new int[fIdCount];
    HROAp[0] = 0;
    for (int i = 0; i < ufRows.size(); i++){
      HUFPtr[i] = ufRows[i];
      int rowNnzCount = ap[ufRows[i]+1] - ap[ufRows[i]];
      HROAp[i+1] = rowNnzCount + HROAp[i];
      for(int j = 0; j < rowNnzCount; j++){
        HROAi[HROAp[i]+j] = ai[ap[ufRows[i]] + j];
        HROAx[HROAp[i]+j] = ax[ap[ufRows[i]] + j];
      }
    }
    HFPtr[0] = 0;
    for (int i = 0; i < fRows.size(); i++){
      HFPtr[i+1] = HFPtr[i] + fRows[i].size();
      for (int j = HFPtr[i]; j < HFPtr[i+1]; j++){
        HFId[j] = fRows[i][j-HFPtr[i]];
      }
    }
    this->St->OtherStats["Number of Fused Rows"] = {(double)fIdCount};
    this->St->OtherStats["Number of Fused Nnz"] = {(double)NnzCount};
    UFDim = ufCount;
    UFMGridDim = CEIL(UFDim, MBlockDim);
    cudaMalloc(&DUFPtr, ufCount * sizeof(int));
    cudaMalloc(&DROAp, (ufCount+1) * sizeof(int));
    cudaMalloc(&DROAi, uFNnzCount * sizeof(int));
    cudaMalloc(&DROAx, uFNnzCount * sizeof(float));
    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (ufCount+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, uFNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, uFNnzCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DFPtr, HFPtr, fPtrCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFId, HFId, fIdCount * sizeof(int), cudaMemcpyHostToDevice);
    t1.stop();
    return t1;
  }


  Timer execute() override{
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, NGridDim, 1);
    dim3 ufBlockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_multiplerow_seqreduce_rowbalance_kernel<<<fGridDim, fBlockDim>>>
        (InTensor->M, InTensor->N, InTensor->K, RowPerThread, InTensor->DACsrAp,
         InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
         OutTensor->DXx, DFPtr, DFId);
    cudaDeviceSynchronize();
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<ufGridDim, ufBlockDim>>>
        (UFDim, InTensor->N, InTensor->K, DROAp,
         DROAi, DROAx, OutTensor->DACx, OutTensor->DXx,
         DUFPtr);
    cudaDeviceSynchronize();
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }
public:
  FusedSpMMSpMMHighFusionRatio(CudaTensorInputs *In1, Stats *Stat1,
                                            int ThreadPerBlock, int RowTile)
      : FusedSpMMSpMMSeqReduceRowBalanceReordered(In1, Stat1, ThreadPerBlock), RowTile(RowTile) {}
};


class FusedSpMMSpMMSeqReduceBColsBlocking: public FusedSpMMSpMMSeqReduceRowBalance
{
protected:

  int NTile;
  void setup() override {
    NGridDim = CEIL(InTensor->N, NTile);
    NBlockDim = MIN(InTensor->N, NTile);
    MBlockDim = CEIL(ThreadPerBlock, NBlockDim);
    MGridDim = CEIL(InTensor->M, MBlockDim);
  }


public:
  FusedSpMMSpMMSeqReduceBColsBlocking(CudaTensorInputs *In1, Stats *Stat1,
                                      int ThreadPerBlock, int NTile)
      : FusedSpMMSpMMSeqReduceRowBalance(In1, Stat1, ThreadPerBlock), NTile(NTile) {}
};


class FusedSpMMSpMMSeqReduceBColsBlockingWithSharedMem: public FusedSpMMSpMMSeqReduceRowBalance
{
protected:
  int SharedMemSize;
  int NTile;
  void setup() override {
    NGridDim = CEIL(InTensor->N, NTile);
    NBlockDim = MIN(InTensor->N, NTile);
    MBlockDim = CEIL(ThreadPerBlock, NBlockDim);
    MGridDim = CEIL(InTensor->M, MBlockDim);
    SharedMemSize = NBlockDim * MBlockDim * sizeof(float);
  }

  Timer execute() override{
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, NGridDim, 1);
    dim3 ufBlockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_spmmspmm_seqreduce_rowbalance_sm_kernel<<<fGridDim, fBlockDim, SharedMemSize>>>
        (InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
         InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
         OutTensor->DXx, DFPtr, DFId);
    cudaDeviceSynchronize();
    csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<ufGridDim, ufBlockDim>>>
        (UFDim, InTensor->N, InTensor->K, InTensor->DACsrAp,
         InTensor->DACsrI, InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx,
         DUFPtr);
    cudaDeviceSynchronize();
    t1.stopGPU("FusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }


public:
  FusedSpMMSpMMSeqReduceBColsBlockingWithSharedMem(CudaTensorInputs *In1, Stats *Stat1,
                                      int ThreadPerBlock, int NTile)
      : FusedSpMMSpMMSeqReduceRowBalance(In1, Stat1, ThreadPerBlock), NTile(NTile) {}
};
#endif // SPARSE_FUSION_CUDA_SPMM_SPMM_DEMO_UTILS_H

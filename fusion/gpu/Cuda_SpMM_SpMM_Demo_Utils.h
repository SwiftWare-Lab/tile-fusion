//
// Created by salehm32 on 20/06/24.
//

#ifndef SPARSE_FUSION_CUDA_SPMM_SPMM_DEMO_UTILS_H
#define SPARSE_FUSION_CUDA_SPMM_SPMM_DEMO_UTILS_H
#include "Cuda_SpMM_Demo_Utils.h"
#include <algorithm>
#include <stdexcept>
#include <vector>
// #include "SW_SpMM_Kernels.h"

#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(a, b) ((a < b) ? a : b)

// Launches and GPU timing events below use the default stream. Its in-order
// execution enforces kernel dependencies, and stopGPU waits for completion.
class SeqSpMMSpMM : public SWTensorBench<float> {
protected:
  CudaTensorInputs *InTensor;

  void setup() override {
    this->St->OtherStats["Number of Fused Rows"] = {0.};
    this->St->OtherStats["Number of Fused Nnz"] = {0.};
    this->St->OtherStats["Color Count"] = {0.};
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

  Timer execute() override {
    OutTensor->reset();
    Timer t1;
    dim3 gridDim(MGridDim, NGridDim, 1);
    dim3 blockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csrspmm_seqreduce_rowbalance_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    csrspmm_seqreduce_rowbalance_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx);
    t1.stopGPU("UnfusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  SpMMSpMMSeqReduceRowBalance(CudaTensorInputs *In1, Stats *Stat1,
                              int ThreadPerBlock = 256)
      : SeqSpMMSpMM(In1, Stat1), ThreadPerBlock(ThreadPerBlock) {}
};

class SpMMSpMMSeqReduceRowBalanceCoarsenedRow :
    public SpMMSpMMSeqReduceRowBalance{
protected:
  int RowTile;
  void setup() override {
    SeqSpMMSpMM::setup();
    NGridDim = CEIL(InTensor->N, ThreadPerBlock);
    NBlockDim = MIN(InTensor->N, ThreadPerBlock);
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
        InTensor->M, InTensor->N, InTensor->K, threadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    csrspmm_seqreduce_rowcoarsened_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, threadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx);
    t1.stopGPU("UnfusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  SpMMSpMMSeqReduceRowBalanceCoarsenedRow(CudaTensorInputs *In1, Stats *Stat1,
                              int ThreadPerBlock, int RowTile1)
      : SpMMSpMMSeqReduceRowBalance(In1, Stat1, ThreadPerBlock), RowTile(RowTile1)
  {}
};

class SpMMSpMMSeqReduceRowBalanceCoarsenedRowStride
    : public SpMMSpMMSeqReduceRowBalanceCoarsenedRow{
protected:
  Timer execute() override{
    OutTensor->reset();
    int threadWorkReps = CEIL(RowTile, MBlockDim);
    Timer t1;
    dim3 gridDim(MGridDim, NGridDim, 1);
    dim3 blockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csrspmm_seqreduce_striderowcoarsened_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, threadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
    csrspmm_seqreduce_striderowcoarsened_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, threadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx);
    t1.stopGPU("UnfusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }
public:
  SpMMSpMMSeqReduceRowBalanceCoarsenedRowStride(CudaTensorInputs *In1, Stats *Stat1,
                              int ThreadPerBlock, int RowTile1)
      : SpMMSpMMSeqReduceRowBalanceCoarsenedRow(In1, Stat1, ThreadPerBlock, RowTile1)
  {}
};

class FusedSpMMSpMMCSRCSC: public SpMMSpMMSeqReduceRowBalance{
protected:

  Timer execute() override {
    OutTensor->reset();
    Timer t1;
    dim3 gridDim(MGridDim, NGridDim, 1);
    dim3 blockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_multiplerow_1v1fusedParReduceAtomic_rowbalance_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal);
    t1.stopGPU("UnfusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMCSRCSC(CudaTensorInputs *In1, Stats *Stat1,
                              int ThreadPerBlock = 256)
      : SpMMSpMMSeqReduceRowBalance(In1, Stat1, ThreadPerBlock) {}
};

class FusedSpMMSpMMCSRCSCNoAtomic: public SpMMSpMMSeqReduceRowBalance{
protected:

  Timer execute() override {
    OutTensor->reset();
    Timer t1;
    dim3 gridDim(MGridDim, NGridDim, 1);
    dim3 blockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_multiplerow_1v1fusedParReduceNoAtomic_rowbalance_kernel<<<gridDim, blockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal);
    t1.stopGPU("UnfusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMCSRCSCNoAtomic(CudaTensorInputs *In1, Stats *Stat1,
                      int ThreadPerBlock = 256)
      : SpMMSpMMSeqReduceRowBalance(In1, Stat1, ThreadPerBlock) {}
};

// A true CSR producer / CSC consumer pair. The consumer CSC is constructed
// once in analysis(), not in the measured execution loop.
class FusedSpMMSpMMCSRCSCPrepared : public SeqSpMMSpMM {
protected:
  sym_lib::CSC *ConsumerCSC = nullptr;
  int *DConsumerPtr = nullptr;
  int *DConsumerIdx = nullptr;
  float *DConsumerVal = nullptr;
  int ThreadsPerBlock;

  void prepareConsumer() {
    ConsumerCSC = sym_lib::csr_to_csc(InTensor->BCsr);
    std::vector<float> values(ConsumerCSC->nnz);
    for (int p = 0; p < ConsumerCSC->nnz; ++p)
      values[p] = static_cast<float>(ConsumerCSC->x[p]);
    cudaMalloc(&DConsumerPtr, (ConsumerCSC->n + 1) * sizeof(int));
    cudaMalloc(&DConsumerIdx, ConsumerCSC->nnz * sizeof(int));
    cudaMalloc(&DConsumerVal, ConsumerCSC->nnz * sizeof(float));
    cudaMemcpy(DConsumerPtr, ConsumerCSC->p,
               (ConsumerCSC->n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DConsumerIdx, ConsumerCSC->i,
               ConsumerCSC->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DConsumerVal, values.data(),
               ConsumerCSC->nnz * sizeof(float), cudaMemcpyHostToDevice);
  }

public:
  FusedSpMMSpMMCSRCSCPrepared(CudaTensorInputs *inputs, Stats *stats,
                              int threadsPerBlock)
      : SeqSpMMSpMM(inputs, stats), ThreadsPerBlock(threadsPerBlock) {}

  ~FusedSpMMSpMMCSRCSCPrepared() {
    cudaFree(DConsumerPtr);
    cudaFree(DConsumerIdx);
    cudaFree(DConsumerVal);
    delete ConsumerCSC;
  }
};

class FusedSpMMSpMMCSRCSCAtomic : public FusedSpMMSpMMCSRCSCPrepared {
protected:
  int RowsPerBlock;

  Timer analysis() override {
    Timer timer;
    timer.start();
    prepareConsumer();
    timer.stop("CSRCSC_Convert");
    return timer;
  }

  Timer execute() override {
    OutTensor->reset();
    // Other benchmark variants may leave an unrelated launch error pending.
    cudaGetLastError();
    Timer timer;
    timer.startGPU();
    const int featureThreads = std::min(
        ThreadsPerBlock, std::max(32, 32 * CEIL(InTensor->N, 32)));
    const int blockThreads = featureThreads * RowsPerBlock;
    if (blockThreads > 1024)
      throw std::runtime_error("CSR-CSC atomic block exceeds 1024 threads");
    if (InTensor->M > 0)
      spmm_spmm_csr_csc_atomic_kernel<<<CEIL(InTensor->M, RowsPerBlock),
                                        blockThreads>>>(
          InTensor->M, InTensor->N, RowsPerBlock, featureThreads,
          InTensor->DACsrAp, InTensor->DACsrI,
          InTensor->DACsrVal, DConsumerPtr, DConsumerIdx, DConsumerVal,
          InTensor->DBx, OutTensor->DACx, OutTensor->DXx);
    const cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess)
      throw std::runtime_error(std::string("atomic kernel launch: ") +
                               cudaGetErrorString(launchStatus));
    timer.stopGPU("CSRCSC_Atomic");
    OutTensor->copyDeviceToHost();
    return timer;
  }

public:
  FusedSpMMSpMMCSRCSCAtomic(CudaTensorInputs *inputs, Stats *stats,
                            int threadsPerBlock = 128, int rowsPerBlock = 1)
      : FusedSpMMSpMMCSRCSCPrepared(inputs, stats, threadsPerBlock),
        RowsPerBlock(rowsPerBlock) {
    if (rowsPerBlock < 1)
      throw std::invalid_argument("CSR-CSC atomic rowsPerBlock must be positive");
  }
};

class FusedSpMMSpMMCSRCSCConflictReduce
    : public FusedSpMMSpMMCSRCSCPrepared {
protected:
  struct ConflictEdge {
    int Destination;
    int LocalRow;
    int OriginalPosition;
    float Value;
  };

  int RowsPerBlock;
  int MinSavingsPercent;
  int *DBlockGroupPtr = nullptr;
  int *DConsumerEdgeGroup = nullptr;
  int *DGroupDst = nullptr;
  int *DGroupEdgePtr = nullptr;
  int *DGroupEdgeLocalRow = nullptr;
  float *DGroupEdgeVal = nullptr;

  template <typename T>
  static void uploadVector(T *&device, const std::vector<T> &host,
                           const char *description) {
    if (host.empty()) return;
    cudaError_t status = cudaMalloc(&device, host.size() * sizeof(T));
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("cudaMalloc ") + description +
                               ": " + cudaGetErrorString(status));
    status = cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("cudaMemcpy ") + description +
                               ": " + cudaGetErrorString(status));
  }

  void prepareConflictGroups() {
    prepareConsumer();
    const int blockCount = CEIL(InTensor->M, RowsPerBlock);
    std::vector<int> blockGroupPtr(1, 0);
    std::vector<int> consumerEdgeGroup(ConsumerCSC->nnz, -1);
    std::vector<int> groupDst;
    std::vector<int> groupEdgePtr(1, 0);
    std::vector<int> groupEdgeLocalRow;
    std::vector<float> groupEdgeVal;
    long long groupedRows = 0;
    long long eliminatedEdges = 0;

    for (int block = 0; block < blockCount; ++block) {
      const int firstRow = block * RowsPerBlock;
      const int lastRow = std::min(firstRow + RowsPerBlock, InTensor->M);
      int edgeCount = 0;
      for (int row = firstRow; row < lastRow; ++row)
        edgeCount += ConsumerCSC->p[row + 1] - ConsumerCSC->p[row];
      std::vector<ConflictEdge> edges;
      edges.reserve(edgeCount);
      for (int row = firstRow; row < lastRow; ++row)
        for (int p = ConsumerCSC->p[row]; p < ConsumerCSC->p[row + 1]; ++p)
          edges.push_back({ConsumerCSC->i[p], row - firstRow,
                           p,
                           static_cast<float>(ConsumerCSC->x[p])});
      std::sort(edges.begin(), edges.end(),
                [](const ConflictEdge &left, const ConflictEdge &right) {
                  if (left.Destination != right.Destination)
                    return left.Destination < right.Destination;
                  return left.LocalRow < right.LocalRow;
                });

      int uniqueDestinations = 0;
      for (size_t p = 0; p < edges.size();) {
        ++uniqueDestinations;
        const int destination = edges[p].Destination;
        while (p < edges.size() && edges[p].Destination == destination) ++p;
      }
      const int savings = edgeCount - uniqueDestinations;
      const bool useGroups = savings > 0 &&
          100LL * savings >=
              static_cast<long long>(MinSavingsPercent) * edgeCount;
      if (useGroups) {
        groupedRows += lastRow - firstRow;
        eliminatedEdges += savings;
        for (size_t p = 0; p < edges.size();) {
          const int destination = edges[p].Destination;
          size_t end = p + 1;
          while (end < edges.size() &&
                 edges[end].Destination == destination)
            ++end;
          if (end - p == 1) {
            p = end;
            continue;
          }
          const int group = static_cast<int>(groupDst.size());
          groupDst.push_back(destination);
          do {
            consumerEdgeGroup[edges[p].OriginalPosition] = group;
            groupEdgeLocalRow.push_back(edges[p].LocalRow);
            groupEdgeVal.push_back(edges[p].Value);
            ++p;
          } while (p < end);
          groupEdgePtr.push_back(static_cast<int>(groupEdgeVal.size()));
        }
      }
      blockGroupPtr.push_back(static_cast<int>(groupDst.size()));
    }

    uploadVector(DBlockGroupPtr, blockGroupPtr, "conflict block offsets");
    uploadVector(DConsumerEdgeGroup, consumerEdgeGroup,
                 "consumer conflict groups");
    uploadVector(DGroupDst, groupDst, "conflict group destinations");
    uploadVector(DGroupEdgePtr, groupEdgePtr, "conflict group offsets");
    uploadVector(DGroupEdgeLocalRow, groupEdgeLocalRow,
                 "conflict edge local rows");
    uploadVector(DGroupEdgeVal, groupEdgeVal, "conflict edge values");
    this->St->OtherStats["Number of Fused Rows"] = {
        static_cast<double>(groupedRows)};
    this->St->OtherStats["Number of Fused Nnz"] = {
        static_cast<double>(eliminatedEdges)};
  }

  Timer analysis() override {
    Timer timer;
    timer.start();
    prepareConflictGroups();
    timer.stop("CSRCSC_Conflict_Preprocess");
    return timer;
  }

  Timer execute() override {
    OutTensor->reset();
    cudaGetLastError();
    Timer timer;
    timer.startGPU();
    const int featureThreads = std::min(
        ThreadsPerBlock, std::max(32, 32 * CEIL(InTensor->N, 32)));
    const int blockThreads = featureThreads * RowsPerBlock;
    if (blockThreads > 1024)
      throw std::runtime_error(
          "CSR-CSC conflict-reduction block exceeds 1024 threads");
    const size_t sharedBytes = static_cast<size_t>(RowsPerBlock) *
                               InTensor->N * sizeof(float);
    if (InTensor->M > 0)
      spmm_spmm_csr_csc_conflict_reduce_kernel
          <<<CEIL(InTensor->M, RowsPerBlock), blockThreads, sharedBytes>>>(
              InTensor->M, InTensor->N, RowsPerBlock, featureThreads,
              InTensor->DACsrAp, InTensor->DACsrI, InTensor->DACsrVal,
              DConsumerPtr, DConsumerIdx, DConsumerVal, DBlockGroupPtr,
              DConsumerEdgeGroup, DGroupDst, DGroupEdgePtr,
              DGroupEdgeLocalRow, DGroupEdgeVal, InTensor->DBx,
              OutTensor->DACx, OutTensor->DXx);
    const cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess)
      throw std::runtime_error(std::string("conflict-reduction kernel launch: ") +
                               cudaGetErrorString(launchStatus));
    timer.stopGPU("CSRCSC_ConflictReduce");
    OutTensor->copyDeviceToHost();
    return timer;
  }

public:
  FusedSpMMSpMMCSRCSCConflictReduce(
      CudaTensorInputs *inputs, Stats *stats, int threadsPerBlock = 128,
      int rowsPerBlock = 4, int minSavingsPercent = 40)
      : FusedSpMMSpMMCSRCSCPrepared(inputs, stats, threadsPerBlock),
        RowsPerBlock(rowsPerBlock), MinSavingsPercent(minSavingsPercent) {
    if (rowsPerBlock < 2)
      throw std::invalid_argument(
          "CSR-CSC conflict reduction needs at least two rows per block");
    if (minSavingsPercent < 0 || minSavingsPercent > 100)
      throw std::invalid_argument(
          "CSR-CSC conflict min savings must be between 0 and 100");
  }

  ~FusedSpMMSpMMCSRCSCConflictReduce() {
    cudaFree(DBlockGroupPtr);
    cudaFree(DConsumerEdgeGroup);
    cudaFree(DGroupDst);
    cudaFree(DGroupEdgePtr);
    cudaFree(DGroupEdgeLocalRow);
    cudaFree(DGroupEdgeVal);
  }
};

class FusedSpMMSpMMCSRCSCColored : public FusedSpMMSpMMCSRCSCPrepared {
protected:
  int TileSize;
  int *DColorTileIds = nullptr;
  std::vector<int> ColorOffsets;

  void prepareColored() {
    prepareConsumer();

    // Size-balanced greedy coloring of the implicit tile-conflict graph.
    // Two tiles conflict iff their CSC columns reach the same output row.
    const int tileCount = CEIL(InTensor->M, TileSize);
    std::vector<std::vector<int>> rowColors(InTensor->L);
    std::vector<std::vector<int>> colorTiles;
    std::vector<int> lastSeen(InTensor->L, -1);
    std::vector<int> forbidden;
    std::vector<int> touched;
    for (int tile = 0; tile < tileCount; ++tile) {
      touched.clear();
      for (int row = tile * TileSize;
           row < std::min((tile + 1) * TileSize, InTensor->M); ++row) {
        for (int p = ConsumerCSC->p[row]; p < ConsumerCSC->p[row + 1]; ++p) {
          int outputRow = ConsumerCSC->i[p];
          if (lastSeen[outputRow] != tile) {
            lastSeen[outputRow] = tile;
            touched.push_back(outputRow);
          }
        }
      }
      if (forbidden.size() < colorTiles.size())
        forbidden.resize(colorTiles.size(), -1);
      for (int outputRow : touched)
        for (int color : rowColors[outputRow])
          forbidden[color] = tile;
      const int existingColors = static_cast<int>(colorTiles.size());
      int color = existingColors;
      for (int candidate = 0; candidate < existingColors; ++candidate)
        if (forbidden[candidate] != tile &&
            (color == existingColors ||
             colorTiles[candidate].size() < colorTiles[color].size()))
          color = candidate;
      if (color == existingColors)
        colorTiles.emplace_back();
      colorTiles[color].push_back(tile);
      for (int outputRow : touched)
        rowColors[outputRow].push_back(color);
    }

    std::vector<int> tileIds;
    ColorOffsets.push_back(0);
    for (const auto &tiles : colorTiles) {
      tileIds.insert(tileIds.end(), tiles.begin(), tiles.end());
      ColorOffsets.push_back(static_cast<int>(tileIds.size()));
    }
    if (!tileIds.empty()) {
      cudaMalloc(&DColorTileIds, tileIds.size() * sizeof(int));
      cudaMemcpy(DColorTileIds, tileIds.data(), tileIds.size() * sizeof(int),
                 cudaMemcpyHostToDevice);
    }
    this->St->OtherStats["Color Count"] = {
        static_cast<double>(colorTiles.size())};
  }

  void launchColors(cudaStream_t stream) {
    const int featureThreads = std::min(
        ThreadsPerBlock, std::max(32, 32 * CEIL(InTensor->N, 32)));
    for (size_t color = 0; color + 1 < ColorOffsets.size(); ++color) {
      int offset = ColorOffsets[color];
      int count = ColorOffsets[color + 1] - offset;
      spmm_spmm_csr_csc_colored_kernel<<<count, featureThreads, 0, stream>>>(
          InTensor->M, InTensor->N, TileSize, DColorTileIds + offset,
          InTensor->DACsrAp, InTensor->DACsrI, InTensor->DACsrVal,
          DConsumerPtr, DConsumerIdx, DConsumerVal, InTensor->DBx,
          OutTensor->DACx, OutTensor->DXx);
    }
  }

  Timer analysis() override {
    Timer timer;
    timer.start();
    prepareColored();
    timer.stop("CSRCSC_Coloring");
    return timer;
  }

  Timer execute() override {
    OutTensor->reset();
    cudaGetLastError();
    Timer timer;
    timer.startGPU();
    launchColors(0);
    const cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess)
      throw std::runtime_error(std::string("direct colored kernel launch: ") +
                               cudaGetErrorString(launchStatus));
    timer.stopGPU("CSRCSC_Colored");
    OutTensor->copyDeviceToHost();
    return timer;
  }

public:
  FusedSpMMSpMMCSRCSCColored(CudaTensorInputs *inputs, Stats *stats,
                             int threadsPerBlock, int tileSize)
      : FusedSpMMSpMMCSRCSCPrepared(inputs, stats, threadsPerBlock),
        TileSize(tileSize) {}

  ~FusedSpMMSpMMCSRCSCColored() { cudaFree(DColorTileIds); }
};

// Fuse only producer rows with 1..MaxRowNnz entries. The remaining producer
// rows and their consumer contributions use separate CSR kernels. Coloring is
// built solely from the compact list of fused producer rows.
class FusedSpMMSpMMCSRCSCHybridColored
    : public FusedSpMMSpMMCSRCSCPrepared {
  int TileSize;
  int MaxRowNnz;
  bool RowParallel128;
  bool Cooperative;
  int CooperativeGridBlocks = 0;
  int FusedRowCount = 0;
  int UnfusedRowCount = 0;
  int *DFusedRows = nullptr;
  int *DUnfusedRows = nullptr;
  int *DFusedMask = nullptr;
  int *DColorTileIds = nullptr;
  int *DColorOffsets = nullptr;
  int *DConsumerCsrPtr = nullptr;
  int *DConsumerCsrIdx = nullptr;
  float *DConsumerCsrVal = nullptr;
  std::vector<int> ColorOffsets;

  static void checkCuda(cudaError_t result, const char *operation) {
    if (result != cudaSuccess)
      throw std::runtime_error(std::string(operation) + ": " +
                               cudaGetErrorString(result));
  }

  Timer analysis() override {
    Timer timer;
    timer.start();
    prepareConsumer();

    // The existing CUDA inputs upload the producer CSR only. Keep a separate
    // consumer CSR for the unfused gather, which also supports A != B.
    const auto *consumer = InTensor->BCsr;
    std::vector<float> consumerValues(consumer->nnz);
    for (int p = 0; p < consumer->nnz; ++p)
      consumerValues[p] = static_cast<float>(consumer->x[p]);
    checkCuda(cudaMalloc(&DConsumerCsrPtr, (consumer->m + 1) * sizeof(int)),
              "cudaMalloc consumer CSR ptr");
    checkCuda(cudaMemcpy(DConsumerCsrPtr, consumer->p,
                         (consumer->m + 1) * sizeof(int),
                         cudaMemcpyHostToDevice), "cudaMemcpy consumer CSR ptr");
    if (consumer->nnz > 0) {
      checkCuda(cudaMalloc(&DConsumerCsrIdx, consumer->nnz * sizeof(int)),
                "cudaMalloc consumer CSR idx");
      checkCuda(cudaMalloc(&DConsumerCsrVal, consumer->nnz * sizeof(float)),
                "cudaMalloc consumer CSR values");
      checkCuda(cudaMemcpy(DConsumerCsrIdx, consumer->i,
                           consumer->nnz * sizeof(int), cudaMemcpyHostToDevice),
                "cudaMemcpy consumer CSR idx");
      checkCuda(cudaMemcpy(DConsumerCsrVal, consumerValues.data(),
                           consumer->nnz * sizeof(float), cudaMemcpyHostToDevice),
                "cudaMemcpy consumer CSR values");
    }

    std::vector<int> fusedRows, unfusedRows, fusedMask(InTensor->M, 0);
    int fusedNnz = 0;
    for (int row = 0; row < InTensor->M; ++row) {
      const int nnz = InTensor->ACsr->p[row + 1] - InTensor->ACsr->p[row];
      if (nnz > 0 && nnz <= MaxRowNnz) {
        fusedMask[row] = 1;
        fusedRows.push_back(row);
        fusedNnz += nnz;
      } else {
        unfusedRows.push_back(row);
      }
    }
    FusedRowCount = static_cast<int>(fusedRows.size());
    UnfusedRowCount = static_cast<int>(unfusedRows.size());
    if (FusedRowCount) {
      checkCuda(cudaMalloc(&DFusedRows, FusedRowCount * sizeof(int)),
                "cudaMalloc fused rows");
      checkCuda(cudaMemcpy(DFusedRows, fusedRows.data(),
                           FusedRowCount * sizeof(int), cudaMemcpyHostToDevice),
                "cudaMemcpy fused rows");
    }
    if (UnfusedRowCount) {
      checkCuda(cudaMalloc(&DUnfusedRows, UnfusedRowCount * sizeof(int)),
                "cudaMalloc unfused rows");
      checkCuda(cudaMemcpy(DUnfusedRows, unfusedRows.data(),
                           UnfusedRowCount * sizeof(int), cudaMemcpyHostToDevice),
                "cudaMemcpy unfused rows");
      checkCuda(cudaMalloc(&DFusedMask, InTensor->M * sizeof(int)),
                "cudaMalloc fused mask");
      checkCuda(cudaMemcpy(DFusedMask, fusedMask.data(),
                           InTensor->M * sizeof(int), cudaMemcpyHostToDevice),
                "cudaMemcpy fused mask");
    }

    const int tileCount = CEIL(FusedRowCount, TileSize);
    std::vector<std::vector<int>> rowColors(InTensor->L);
    std::vector<std::vector<int>> colorTiles;
    std::vector<int> lastSeen(InTensor->L, -1), forbidden, touched;
    for (int tile = 0; tile < tileCount; ++tile) {
      touched.clear();
      const int end = std::min((tile + 1) * TileSize, FusedRowCount);
      for (int position = tile * TileSize; position < end; ++position) {
        const int row = fusedRows[position];
        for (int p = ConsumerCSC->p[row]; p < ConsumerCSC->p[row + 1]; ++p) {
          const int outputRow = ConsumerCSC->i[p];
          if (lastSeen[outputRow] != tile) {
            lastSeen[outputRow] = tile;
            touched.push_back(outputRow);
          }
        }
      }
      if (forbidden.size() < colorTiles.size())
        forbidden.resize(colorTiles.size(), -1);
      for (int outputRow : touched)
        for (int color : rowColors[outputRow])
          forbidden[color] = tile;
      const int existingColors = static_cast<int>(colorTiles.size());
      int color = existingColors;
      for (int candidate = 0; candidate < existingColors; ++candidate)
        if (forbidden[candidate] != tile &&
            (color == existingColors ||
             colorTiles[candidate].size() < colorTiles[color].size()))
          color = candidate;
      if (color == existingColors)
        colorTiles.emplace_back();
      colorTiles[color].push_back(tile);
      for (int outputRow : touched)
        rowColors[outputRow].push_back(color);
    }
    std::vector<int> tileIds;
    ColorOffsets.push_back(0);
    for (const auto &tiles : colorTiles) {
      tileIds.insert(tileIds.end(), tiles.begin(), tiles.end());
      ColorOffsets.push_back(static_cast<int>(tileIds.size()));
    }
    if (!tileIds.empty()) {
      checkCuda(cudaMalloc(&DColorTileIds, tileIds.size() * sizeof(int)),
                "cudaMalloc colored tile ids");
      checkCuda(cudaMemcpy(DColorTileIds, tileIds.data(),
                           tileIds.size() * sizeof(int), cudaMemcpyHostToDevice),
                "cudaMemcpy colored tile ids");
    }
    if (Cooperative) {
      int device = 0, supportsCooperative = 0, multiprocessors = 0;
      int residentBlocksPerSm = 0;
      checkCuda(cudaGetDevice(&device), "cudaGetDevice");
      checkCuda(cudaDeviceGetAttribute(&supportsCooperative,
                                       cudaDevAttrCooperativeLaunch, device),
                "cudaDevAttrCooperativeLaunch");
      if (!supportsCooperative)
        throw std::runtime_error("device does not support cooperative launches");
      checkCuda(cudaDeviceGetAttribute(&multiprocessors,
                                       cudaDevAttrMultiProcessorCount, device),
                "cudaDevAttrMultiProcessorCount");
      checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &residentBlocksPerSm,
                    spmm_spmm_csr_csc_hybrid_cooperative_kernel, 128, 0),
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
      if (residentBlocksPerSm < 1)
        throw std::runtime_error("cooperative kernel has no resident blocks");
      int largestColor = 0;
      for (size_t color = 0; color + 1 < ColorOffsets.size(); ++color)
        largestColor = std::max(
            largestColor, ColorOffsets[color + 1] - ColorOffsets[color]);
      CooperativeGridBlocks = std::min(
          multiprocessors * residentBlocksPerSm,
          std::max(multiprocessors,
                   std::min(multiprocessors * 4, largestColor)));
      checkCuda(cudaMalloc(&DColorOffsets,
                           ColorOffsets.size() * sizeof(int)),
                "cudaMalloc color offsets");
      checkCuda(cudaMemcpy(DColorOffsets, ColorOffsets.data(),
                           ColorOffsets.size() * sizeof(int),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy color offsets");
    }
    this->St->OtherStats["Number of Fused Rows"] = {
        static_cast<double>(FusedRowCount)};
    this->St->OtherStats["Number of Fused Nnz"] = {
        static_cast<double>(fusedNnz)};
    this->St->OtherStats["Color Count"] = {
        static_cast<double>(colorTiles.size())};
    timer.stop("CSRCSC_HybridColoring");
    return timer;
  }

  Timer execute() override {
    if (!Cooperative)
      OutTensor->reset();
    cudaGetLastError();
    Timer timer;
    timer.startGPU();
    if (Cooperative) {
      SpmmSpmmHybridCooperativeArgs args = {
          FusedRowCount, UnfusedRowCount, InTensor->L, InTensor->N, TileSize,
          static_cast<int>(ColorOffsets.size()) - 1,
          DColorOffsets, DColorTileIds, DFusedRows, DUnfusedRows, DFusedMask,
          InTensor->DACsrAp, InTensor->DACsrI, DConsumerPtr, DConsumerIdx,
          DConsumerCsrPtr, DConsumerCsrIdx, InTensor->DACsrVal,
          DConsumerVal, DConsumerCsrVal, InTensor->DBx, OutTensor->DACx,
          OutTensor->DXx};
      void *launchArgs[] = {&args};
      checkCuda(cudaLaunchCooperativeKernel(
                    reinterpret_cast<const void *>(
                        spmm_spmm_csr_csc_hybrid_cooperative_kernel),
                    CooperativeGridBlocks, 128, launchArgs, 0, 0),
                "cudaLaunchCooperativeKernel");
    } else if (RowParallel128) {
      if (UnfusedRowCount)
        spmm_spmm_hybrid_unfused_producer_rows4_kernel
            <<<CEIL(UnfusedRowCount, 4), 128>>>(
                UnfusedRowCount, InTensor->N, DUnfusedRows,
                InTensor->DACsrAp, InTensor->DACsrI, InTensor->DACsrVal,
                InTensor->DBx, OutTensor->DACx);
      for (size_t color = 0; color + 1 < ColorOffsets.size(); ++color) {
        const int offset = ColorOffsets[color];
        const int count = ColorOffsets[color + 1] - offset;
        spmm_spmm_csr_csc_hybrid_colored_rows4_kernel<<<count, 128>>>(
            FusedRowCount, InTensor->N, TileSize, DColorTileIds + offset,
            DFusedRows, InTensor->DACsrAp, InTensor->DACsrI,
            InTensor->DACsrVal, DConsumerPtr, DConsumerIdx, DConsumerVal,
            InTensor->DBx, OutTensor->DACx, OutTensor->DXx);
      }
      if (UnfusedRowCount && InTensor->L > 0)
        spmm_spmm_hybrid_unfused_consumer_rows4_kernel
            <<<CEIL(InTensor->L, 4), 128>>>(
                InTensor->L, InTensor->N, DConsumerCsrPtr, DConsumerCsrIdx,
                DConsumerCsrVal, DFusedMask, OutTensor->DACx,
                OutTensor->DXx);
    } else {
      const int featureThreads = std::min(
          ThreadsPerBlock, std::max(32, 32 * CEIL(InTensor->N, 32)));
      if (UnfusedRowCount)
        spmm_spmm_hybrid_unfused_producer_kernel
            <<<UnfusedRowCount, featureThreads>>>(
                InTensor->N, DUnfusedRows, InTensor->DACsrAp,
                InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx,
                OutTensor->DACx);
      for (size_t color = 0; color + 1 < ColorOffsets.size(); ++color) {
        const int offset = ColorOffsets[color];
        const int count = ColorOffsets[color + 1] - offset;
        spmm_spmm_csr_csc_hybrid_colored_kernel<<<count, featureThreads>>>(
            FusedRowCount, InTensor->N, TileSize, DColorTileIds + offset,
            DFusedRows, InTensor->DACsrAp, InTensor->DACsrI,
            InTensor->DACsrVal, DConsumerPtr, DConsumerIdx, DConsumerVal,
            InTensor->DBx, OutTensor->DACx, OutTensor->DXx);
      }
      if (UnfusedRowCount && InTensor->L > 0)
        spmm_spmm_hybrid_unfused_consumer_kernel
            <<<InTensor->L, featureThreads>>>(
                InTensor->L, InTensor->N, DConsumerCsrPtr, DConsumerCsrIdx,
                DConsumerCsrVal, DFusedMask, OutTensor->DACx,
                OutTensor->DXx);
    }
    checkCuda(cudaGetLastError(), "hybrid kernel launch");
    timer.stopGPU("CSRCSC_HybridColored");
    OutTensor->copyDeviceToHost();
    return timer;
  }

public:
  FusedSpMMSpMMCSRCSCHybridColored(CudaTensorInputs *inputs, Stats *stats,
                                   int threadsPerBlock, int tileSize,
                                   int maxRowNnz, bool rowParallel128 = false,
                                   bool cooperative = false)
      : FusedSpMMSpMMCSRCSCPrepared(inputs, stats, threadsPerBlock),
        TileSize(tileSize), MaxRowNnz(maxRowNnz),
        RowParallel128(rowParallel128), Cooperative(cooperative) {
    if (tileSize <= 0 || maxRowNnz <= 0)
      throw std::invalid_argument("tile size and row nnz threshold must be positive");
    if (rowParallel128 && threadsPerBlock != 128)
      throw std::invalid_argument("row-parallel hybrid requires 128 threads per block");
    if (cooperative && !rowParallel128)
      throw std::invalid_argument("cooperative hybrid requires row-parallel mode");
  }

  ~FusedSpMMSpMMCSRCSCHybridColored() {
    cudaFree(DFusedRows);
    cudaFree(DUnfusedRows);
    cudaFree(DFusedMask);
    cudaFree(DColorTileIds);
    cudaFree(DColorOffsets);
    cudaFree(DConsumerCsrPtr);
    cudaFree(DConsumerCsrIdx);
    cudaFree(DConsumerCsrVal);
  }
};

class FusedSpMMSpMMCSRCSCColoredGraph
    : public FusedSpMMSpMMCSRCSCColored {
protected:
  cudaGraph_t Graph = nullptr;
  cudaGraphExec_t GraphExec = nullptr;

  static void checkCuda(cudaError_t result, const char *operation) {
    if (result != cudaSuccess)
      throw std::runtime_error(std::string(operation) + ": " +
                               cudaGetErrorString(result));
  }

  Timer analysis() override {
    Timer timer;
    timer.start();
    prepareColored();
    if (ColorOffsets.size() > 1) {
      cudaGetLastError();
      // The legacy default stream cannot be captured. The resulting graph
      // can still be replayed on the default stream alongside the baselines.
      cudaStream_t captureStream = nullptr;
      checkCuda(cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags");
      checkCuda(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal),
                "cudaStreamBeginCapture");
      launchColors(captureStream);
      checkCuda(cudaGetLastError(), "colored kernel capture");
      checkCuda(cudaStreamEndCapture(captureStream, &Graph),
                "cudaStreamEndCapture");
      checkCuda(cudaStreamDestroy(captureStream), "cudaStreamDestroy");
      checkCuda(cudaGraphInstantiate(&GraphExec, Graph, 0),
                "cudaGraphInstantiate");
    }
    timer.stop("CSRCSC_ColorGraphSetup");
    return timer;
  }

  Timer execute() override {
    OutTensor->reset();
    cudaGetLastError();
    Timer timer;
    timer.startGPU();
    if (GraphExec)
      checkCuda(cudaGraphLaunch(GraphExec, 0), "cudaGraphLaunch");
    timer.stopGPU("CSRCSC_ColoredGraph");
    OutTensor->copyDeviceToHost();
    return timer;
  }

public:
  FusedSpMMSpMMCSRCSCColoredGraph(CudaTensorInputs *inputs, Stats *stats,
                                  int threadsPerBlock, int tileSize)
      : FusedSpMMSpMMCSRCSCColored(inputs, stats, threadsPerBlock, tileSize) {}

  ~FusedSpMMSpMMCSRCSCColoredGraph() {
    if (GraphExec) cudaGraphExecDestroy(GraphExec);
    if (Graph) cudaGraphDestroy(Graph);
  }
};

class FusedSpMMSpMMSeqReduceRowBalanceRedundant
    : public SpMMSpMMSeqReduceRowBalance {
protected:
    int *HL1Ptr;
    int *DL1Ptr;
    int *HL1Id;
    int *DL1Id;
  int RowTile;
  int RowsPerThread;

  Timer analysis() override {
    Timer t1;
    t1.start();
    MGridDim = CEIL(InTensor->M, RowTile);
    std::vector<std::set<int>> dependantL1Rows(MGridDim);
    int rowTile = RowTile;
    int *ap = InTensor->ACsr->p;
    int *ai = InTensor->ACsr->i;
    for (int i = 0; i < InTensor->M; i += rowTile) {
      int t = i / rowTile;
      int end = MIN(i + rowTile, InTensor->M);
      for (int ii = i; ii < end; ii++) {
        for (int j = ap[ii]; j < ap[ii + 1]; j++) {
          dependantL1Rows[t].insert(ai[j]);
        }
      }
    }
    int dependantCount = 0;
    int maxL1TileSize = 0;
    for (int i = 0; i < dependantL1Rows.size(); i++) {
      int tileDependantCount = dependantL1Rows[i].size();
      if (tileDependantCount > maxL1TileSize){
        maxL1TileSize = tileDependantCount;
      }
      dependantCount += tileDependantCount;
    }
    int ntiles = dependantL1Rows.size() + 1;
    HL1Ptr = new int[dependantL1Rows.size() + 1];
    HL1Id = new int[dependantCount];
    HL1Ptr[0] = 0;
    int ptr = 0;
    for (int i = 0; i < dependantL1Rows.size(); i++) {
      HL1Ptr[i + 1] = HL1Ptr[i] + dependantL1Rows[i].size();
      for (int dr : dependantL1Rows[i]) {
        HL1Id[ptr] = dr;
        ptr++;
      }
    }
    RowsPerThread = CEIL(maxL1TileSize,MBlockDim);
    //    this->St->OtherStats["Number of Fused Rows"] = {(double)fIdCount};
    //    this->St->OtherStats["Number of Fused Nnz"] = {(double)NnzCount};
    cudaMalloc(&DL1Ptr, ntiles * sizeof(int));
    cudaMalloc(&DL1Id, dependantCount * sizeof(int));
    cudaMemcpy(DL1Ptr, HL1Ptr, ntiles * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DL1Id, HL1Id, dependantCount * sizeof(int),
               cudaMemcpyHostToDevice);
    t1.stop();
    return t1;
  }

  Timer execute() override {
    Timer t1;
    dim3 rGridDim(MGridDim, NGridDim, 1);
    dim3 rBlockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csr_redundantFusedTile_multiplerow_seqreduce_rowbalance_kernel<<<rGridDim, rBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, RowTile, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DL1Ptr, DL1Id);
    t1.stopGPU("RedundantFusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }
public:
  FusedSpMMSpMMSeqReduceRowBalanceRedundant(CudaTensorInputs *In1, Stats *Stat1,
                                            int ThreadPerBlock,
                                            int RowTileIn)
      : SpMMSpMMSeqReduceRowBalance(In1, Stat1, ThreadPerBlock), RowTile(RowTileIn) {}

  ~FusedSpMMSpMMSeqReduceRowBalanceRedundant(){
    delete[] HL1Ptr;
    delete[] HL1Id;
    cudaFree(DL1Id);
    cudaFree(DL1Ptr);
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
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<ufGridDim,
                                                           ufBlockDim>>>(
        UFDim, InTensor->N, InTensor->K, InTensor->DACsrAp, InTensor->DACsrI,
        InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx, DUFPtr);
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

class FusedSpMMSpMMSeqReduceRowBalanceReordered
    : public FusedSpMMSpMMSeqReduceRowBalance {
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
    int *ap = InTensor->ACsr->p;
    int *ai = InTensor->ACsr->i;
    float *ax = InTensor->HACsrVal;
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
    HROAx = new float[uFNnzCount];
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
    cudaMalloc(&DROAx, uFNnzCount * sizeof(float));
    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (ufCount + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, uFNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, uFNnzCount * sizeof(float),
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
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    //    t1.stopGPU("ROFusedTileSpMMSpMM");
    //    std::cout << "ROFusedTileSpMMSpMM: " << t1.printTimeCsv(0) <<
    //    std::endl; t2.startGPU();
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    t1.stopGPU("ROUnFusedTileSpMMSpMM");
    //    std::cout << "ROUnFusedTileSpMMSpMM: " << t2.printTimeCsv(0) <<
    //    std::endl;
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
    : public FusedSpMMSpMMSeqReduceRowBalanceReordered {
protected:
  int ThreadWorkReps;
  int FusedThreadsPerBlock;
  int RowTile;
  int UFMBlockDim;
  int UFNBlockDim;
  int UFNGridDim;

  void setup() override {
    SeqSpMMSpMM::setup();
    //    if (InTensor->N == 32){
    //      RowTile = 256;
    //    } else if (InTensor ->N == 64){
    //      RowTile = 128;
    //    } else if (InTensor ->N == 128){
    //      RowTile = 64;
    //    }
    UFNGridDim = CEIL(InTensor->N, ThreadPerBlock);
    UFNBlockDim = MIN(InTensor->N, ThreadPerBlock);
    UFMBlockDim = CEIL(ThreadPerBlock, UFNBlockDim);
    // assert that bCols >= 32 and bCols is a product of 32.
    NGridDim = CEIL(InTensor->N, FusedThreadsPerBlock);
    NBlockDim = MIN(InTensor->N, FusedThreadsPerBlock);
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
    float *ax = InTensor->HACsrVal;
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
    HROAx = new float[uFNnzCount];
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
    cudaMalloc(&DROAx, uFNnzCount * sizeof(float));
    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (ufCount + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, uFNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, uFNnzCount * sizeof(float),
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
        InTensor->M, InTensor->N, InTensor->K, ThreadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }


public:
  FusedSpMMSpMMHighFusionRatio(CudaTensorInputs *In1, Stats *Stat1,
                               int FusedThreadsPerBlock, int ThreadPerBlock,
                               int RowTile)
      : FusedSpMMSpMMSeqReduceRowBalanceReordered(In1, Stat1, ThreadPerBlock),
        FusedThreadsPerBlock(FusedThreadsPerBlock), RowTile(RowTile) {}
};

class FusedSpMMSpMMHighFusionRatioProductOf4Fused
    : public FusedSpMMSpMMHighFusionRatio {
protected:

  Timer analysis() override{
    Timer t1;
    t1.start();
    std::vector<int> ufRows;
    std::vector<std::vector<int>> fRows(MGridDim);
    int rowTile = RowTile;
    int *ap = InTensor->ACsr->p;
    int *ai = InTensor->ACsr->i;
    float *ax = InTensor->HACsrVal;
    int nnzCount = 0;
    for (int i = 0; i < InTensor->M; i += rowTile) {
      std::vector<int> tufRows;
      int t = i / rowTile;
      int end = MIN(i + rowTile, InTensor->M);
      for (int ii = i; ii < end; ii++) {
        if (ai[ap[ii]] < i || ai[ap[ii+1]-1] >= end) {
          tufRows.push_back(ii);
        }
        else {
          fRows[t].push_back(ii);
          nnzCount += ap[ii + 1] - ap[ii];
        }
      }
      while(fRows[t].size() % 4 != 0){
        int fr = *(fRows[t].rbegin());
        tufRows.push_back(fr);
        fRows[t].pop_back();
        nnzCount -= ap[fr + 1] - ap[fr];
      }
      std::sort(tufRows.begin(), tufRows.end());
      std::copy(tufRows.begin(), tufRows.end(), std::back_inserter(ufRows));
    }
    int ufCount = ufRows.size();
    int uFNnzCount = InTensor->ACsr->nnz - nnzCount;
    int fIdCount = InTensor->M - ufCount;
    int fPtrCount = MGridDim + 1;
    HUFPtr = new int[ufCount];
    HROAp = new int[ufCount + 1];
    HROAi = new int[uFNnzCount];
    HROAx = new float[uFNnzCount];
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
    this->St->OtherStats["Number of Fused Nnz"] = {(double)nnzCount};
    UFDim = ufCount;
    UFMGridDim = CEIL(UFDim, UFMBlockDim);
    cudaMalloc(&DUFPtr, ufCount * sizeof(int));
    cudaMalloc(&DROAp, (ufCount + 1) * sizeof(int));
    cudaMalloc(&DROAi, uFNnzCount * sizeof(int));
    cudaMalloc(&DROAx, uFNnzCount * sizeof(float));
    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (ufCount + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, uFNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, uFNnzCount * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DFPtr, HFPtr, fPtrCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFId, HFId, fIdCount * sizeof(int), cudaMemcpyHostToDevice);
    t1.stop();
    return t1;
  }
public:
  FusedSpMMSpMMHighFusionRatioProductOf4Fused(CudaTensorInputs *In1, Stats *Stat1,
                                     int FusedThreadsPerBlock, int ThreadPerBlock,
                                     int RowTile)
      : FusedSpMMSpMMHighFusionRatio(In1, Stat1, FusedThreadsPerBlock, ThreadPerBlock, RowTile)
  {}

};

class FusedSpMMSpMMHighFusionRatioStride
    : public FusedSpMMSpMMHighFusionRatio {
protected:

  Timer execute() override{
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, UFNGridDim, 1);
    dim3 ufBlockDim(UFNBlockDim, UFMBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_multiplerow_seqreduce_striderowbalance_kernel<<<fGridDim,
                                                                  fBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, ThreadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMHighFusionRatioStride(CudaTensorInputs *In1, Stats *Stat1,
                               int FusedThreadsPerBlock, int ThreadPerBlock,
                               int RowTile)
      : FusedSpMMSpMMHighFusionRatio(In1, Stat1, FusedThreadsPerBlock, ThreadPerBlock, RowTile)
  {}

};


//TODO: fix the offset
class FusedSpMMSpMMHighFusionRatioResultPacked
    : public FusedSpMMSpMMHighFusionRatio {
protected:

  Timer execute() override{
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, UFNGridDim, 1);
    dim3 ufBlockDim(UFNBlockDim, UFMBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_multiplerow_seqreduce_resultpacking_rowbalance_kernel<<<fGridDim,
                                                                  fBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, ThreadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    csr_reorderedresultpacked_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N, InTensor->K, 0, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMHighFusionRatioResultPacked(CudaTensorInputs *In1, Stats *Stat1,
                                     int FusedThreadsPerBlock, int ThreadPerBlock,
                                     int RowTile)
      : FusedSpMMSpMMHighFusionRatio(In1, Stat1, FusedThreadsPerBlock, ThreadPerBlock, RowTile)
  {}

};


class FusedSpMMSpMMHighFusionRatioMultipleBCols
    : public FusedSpMMSpMMHighFusionRatio{
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
    UFNGridDim = CEIL(InTensor->N, ThreadPerBlock);
    UFNBlockDim = MIN(InTensor->N, ThreadPerBlock);
    UFMBlockDim = CEIL(ThreadPerBlock, UFNBlockDim);
    // assert that bCols >= 32 and bCols is a product of 32.
    //TODO: this needs generalization and factorization of bcol per thread and row per thread.
    MGridDim = CEIL(InTensor->M, RowTile);
    NGridDim = CEIL(InTensor->N,FusedThreadsPerBlock);
    MBlockDim = RowTile;
    NBlockDim = CEIL(FusedThreadsPerBlock, MBlockDim);
    ThreadWorkReps = CEIL(InTensor->N, NBlockDim); //used as colPerThread
  }

  Timer execute() override {
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, UFNGridDim, 1);
    dim3 ufBlockDim(UFNBlockDim, UFMBlockDim, 1);
    OutTensor->reset();
    t1.startGPU();
    csr_fusedTile_multiplecol_seqreduce_rowbalance_kernel<<<fGridDim,
                                                            fBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, ThreadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMHighFusionRatioMultipleBCols(CudaTensorInputs *In1, Stats *Stat1,
                                      int ThreadsPerBlock,
                                      int RowTile)
      : FusedSpMMSpMMHighFusionRatio(In1, Stat1, ThreadsPerBlock, ThreadsPerBlock, RowTile)
  {}
};

class FusedSpMMSpMMHighFusionRatioNoSynch
    : public FusedSpMMSpMMHighFusionRatio{
protected:
  Timer execute() override {
    Timer t1;
    int nBlockDimX = CEIL(32, MBlockDim);
    int nBlockDimZ = CEIL(NBlockDim, nBlockDimX);
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(nBlockDimX, MBlockDim, nBlockDimZ);
    dim3 ufGridDim(UFMGridDim, UFNGridDim, 1);
    dim3 ufBlockDim(UFNBlockDim, UFMBlockDim, 1);
    t1.startGPU();
    csr_fusedSynchTile_multiplerow_seqreduce_rowbalance_kernel<<<fGridDim,
                                                            fBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, ThreadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }
public:
  FusedSpMMSpMMHighFusionRatioNoSynch(CudaTensorInputs *In1, Stats *Stat1,
                                      int ThreadsPerBlock,
                                      int RowTile)
  : FusedSpMMSpMMHighFusionRatio(In1, Stat1, ThreadsPerBlock, ThreadsPerBlock, RowTile)
  {}
};

class FusedSpMMSpMMHighFusionRatio2Level: public FusedSpMMSpMMHighFusionRatio{
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
    float *ax = InTensor->HACsrVal;
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
    HROAx = new float[uFNnzCount];
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
    cudaMalloc(&DROAx, uFNnzCount * sizeof(float));
    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (ufCount + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, uFNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, uFNnzCount * sizeof(float),
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
        InTensor->M, InTensor->N, InTensor->K, ThreadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMHighFusionRatio2Level(CudaTensorInputs *In1, Stats *Stat1,
                                     int ThreadsPerBlock,
                                     int RowTile)
      : FusedSpMMSpMMHighFusionRatio(In1, Stat1, ThreadsPerBlock, ThreadsPerBlock, RowTile)
  {}

};

class FusedSpMMSpMMFusedParReduce: public FusedSpMMSpMMHighFusionRatio{
protected:
  int* HFAp;
  int* HFAi;
  float* HFAx;
  int* DFAp;
  int* DFAi;
  float* DFAx;


  //TODO: Complete the code and clean if necessary.
  Timer analysis() override {
    Timer t1;
    t1.start();
    std::vector<int> ufRows;
    std::vector<std::vector<int>> fRows(MGridDim);
    int rowTile = RowTile;
    int *ap = InTensor->ACsr->p;
    int *ai = InTensor->ACsr->i;
    float *ax = InTensor->HACsrVal;
    int NnzCount = 0;
    std::vector<int> fusedPtr;
    std::vector<int> fusedRowInd;
    std::vector<float> fusedRowVal;
    fusedPtr.push_back(0);
    int cntr = 0;
    for (int i = 0; i < InTensor->M; i += rowTile) {
      std::vector<std::vector<int>> colNnzRow(rowTile);
      std::vector<std::vector<float>> colNnzVal(rowTile);
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
    HFAx = new float[fNnzCount];
    HUFPtr = new int[ufCount];
    HROAp = new int[ufCount + 1];
    HROAi = new int[uFNnzCount];
    HROAx = new float[uFNnzCount];
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
    cudaMalloc(&DROAx, uFNnzCount * sizeof(float));
//    cudaMalloc(&DFPtr, fPtrCount * sizeof(int));
//    cudaMalloc(&DFId, fIdCount * sizeof(int));
    cudaMalloc(&DFAp, fApCount * sizeof(int));
    cudaMalloc(&DFAi, fNnzCount * sizeof(int));
    cudaMalloc(&DFAx, fNnzCount * sizeof(float));
    cudaMemcpy(DUFPtr, HUFPtr, ufCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAp, HROAp, (ufCount + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(DROAi, HROAi, uFNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DROAx, HROAx, uFNnzCount * sizeof(float),
               cudaMemcpyHostToDevice);
//    cudaMemcpy(DFPtr, HFPtr, fPtrCount * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(DFId, HFId, fIdCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFAp, HFAp, fApCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFAi, HFAi, fNnzCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(DFAx, HFAx, fNnzCount * sizeof(float), cudaMemcpyHostToDevice);
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
        InTensor->M, InTensor->N, InTensor->K, ThreadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFAp, DFAi, DFAx);
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:

  FusedSpMMSpMMFusedParReduce(CudaTensorInputs *In1, Stats *Stat1,
                              int ThreadsPerBlock,
                              int RowTile)
      : FusedSpMMSpMMHighFusionRatio(In1, Stat1, ThreadsPerBlock, ThreadsPerBlock, RowTile){}

  ~FusedSpMMSpMMFusedParReduce(){
    delete[] HFAp;
    delete[] HFAi;
    delete[] HFAx;
    cudaFree(DFAp);
    cudaFree(DFAi);
    cudaFree(DFAx);
  }
};

//TODO: Wrong
class FusedSpMMSpMMFusedParReduceNoAtomic: public FusedSpMMSpMMFusedParReduce{
protected:
  Timer execute() override {
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, UFNGridDim, 1);
    dim3 ufBlockDim(UFNBlockDim, UFMBlockDim, 1);
    OutTensor->reset();
    t1.startGPU();
    csr_fusedTile_multiplerow_fusedParReduceNoAtomic_rowbalance_kernel<<<fGridDim,
                                                                 fBlockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, ThreadWorkReps, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFAp, DFAi, DFAx);
    csr_reordered_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<
        ufGridDim, ufBlockDim>>>(UFDim, InTensor->N, InTensor->K, DROAp, DROAi,
                                 DROAx, OutTensor->DACx, OutTensor->DXx,
                                 DUFPtr);
    t1.stopGPU("UnFusedTileSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:

  FusedSpMMSpMMFusedParReduceNoAtomic(CudaTensorInputs *In1, Stats *Stat1,
                              int ThreadsPerBlock,
                              int RowTile)
      : FusedSpMMSpMMFusedParReduce(In1, Stat1, ThreadsPerBlock, RowTile){}
};

class FusedSpMMSpMMSeqReduceBColsBlocking
    : public FusedSpMMSpMMSeqReduceRowBalance {
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
      : FusedSpMMSpMMSeqReduceRowBalance(In1, Stat1, ThreadPerBlock),
        NTile(NTile) {}
};

class FusedSpMMSpMMSeqReduceBColsBlockingWithSharedMem
    : public FusedSpMMSpMMSeqReduceRowBalance {
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

  Timer execute() override {
    Timer t1;
    dim3 fGridDim(MGridDim, NGridDim, 1);
    dim3 fBlockDim(NBlockDim, MBlockDim, 1);
    dim3 ufGridDim(UFMGridDim, NGridDim, 1);
    dim3 ufBlockDim(NBlockDim, MBlockDim, 1);
    t1.startGPU();
    csr_fusedTile_spmmspmm_seqreduce_rowbalance_sm_kernel<<<fGridDim, fBlockDim,
                                                            SharedMemSize>>>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<ufGridDim,
                                                           ufBlockDim>>>(
        UFDim, InTensor->N, InTensor->K, InTensor->DACsrAp, InTensor->DACsrI,
        InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx, DUFPtr);
    t1.stopGPU("FusedSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t1;
  }

public:
  FusedSpMMSpMMSeqReduceBColsBlockingWithSharedMem(CudaTensorInputs *In1,
                                                   Stats *Stat1,
                                                   int ThreadPerBlock,
                                                   int NTile)
      : FusedSpMMSpMMSeqReduceRowBalance(In1, Stat1, ThreadPerBlock),
        NTile(NTile) {}
};
#endif // SPARSE_FUSION_CUDA_SPMM_SPMM_DEMO_UTILS_H

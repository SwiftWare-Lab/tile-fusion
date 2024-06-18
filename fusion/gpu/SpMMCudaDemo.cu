//
// Created by mehdi on 6/16/24.
//
#include "Cuda_SpMM_Demo_Utils.h"
#include "Stats.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Defs.h"
#include "sparse-fusion/Fusion_Utils.h"
#include <iostream>

#define WARMUP_NUM_CUDA 20
#define EXE_NUM_CUDA 200
using namespace sym_lib;

int main (const int argc, const char *argv[]) {
  TestParameters tp;
  tp._order_method = SYM_ORDERING::NONE;
  ScheduleParameters sp;
  Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  CSC *aCSC = get_matrix_from_parameter(&tp);
  if (aCSC->m != aCSC->n) {
    return -1;
  }
  CSC *aCSCFull = nullptr;
  if (aCSC->stype == -1 || aCSC->stype == 1) {
    aCSCFull = make_full(aCSC);
  } else {
    aCSCFull = copy_sparse(aCSC);
  }
  tp._dim1 = aCSCFull->m;
  tp._dim2 = aCSCFull->n;
  tp._nnz = aCSCFull->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);

  int numThread = sp._num_threads, numTrial = 7;
  std::string expName = "SpMM_SpMM_Demo";
  auto *inSpMM =
      new CudaTensorInputs(aCSCFull->m, tp._b_cols, aCSCFull->m, aCSCFull->m,
                               aCSCFull, aCSCFull, numThread, numTrial, expName);

  stats = new swiftware::benchmark::Stats("CPU_SpMM_Demo", "SpMM", numTrial, tp._matrix_name, numThread);
  auto *cpuSpMM = new CpuSpMM(inSpMM, stats);
  cpuSpMM->run();
//  cpuSpMM->OutTensor->printDx();
  std::copy(cpuSpMM->OutTensor->ACx,
            cpuSpMM->OutTensor->ACx +
                cpuSpMM->OutTensor->M * cpuSpMM->OutTensor->N,
            inSpMM->CorrectSol);
  inSpMM->IsSolProvided = true;
  auto headerStat = cpuSpMM->printStatsHeader();
  auto cpuSpMMStat = cpuSpMM->printStats();
  delete cpuSpMM;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_cuSparse_SpMM_CSR_Default_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuCuBlasSpMMDefault = new GpuSpMMCuSparse(inSpMM,stats, CUSPARSE_SPMM_ALG_DEFAULT);
  gpuCuBlasSpMMDefault->run();
//  gpuCuBlasSpMMDefault->OutTensor->printDx();
  auto gpuCuBlasSpMMDefaultStat = gpuCuBlasSpMMDefault->printStats();
  delete gpuCuBlasSpMMDefault;
  delete stats;


  // TODO: CuSparseWithPreprocessing has a bug and need to be fixed.
  //  Adding a check function to Benchmark may help. The bug occurs at StopGpu and cudaEventSynchronize function.
  stats = new swiftware::benchmark::Stats("GPU_cuSparse_SpMM_CSR_ALG3_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuCuBlasSpMMAlg3 = new GpuSpMMCuSparse(inSpMM,stats, CUSPARSE_SPMM_CSR_ALG3);
  gpuCuBlasSpMMAlg3->run();
  //  gpuCuBlasSpMMDefault->OutTensor->printDx();
  auto gpuCuBlasSpMMAlg3Stat = gpuCuBlasSpMMAlg3->printStats();
  delete gpuCuBlasSpMMAlg3;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_cuSparse_SpMM_CSR_Default_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuCuBlasSpMMAlg2 = new GpuSpMMCuSparse(inSpMM,stats, CUSPARSE_SPMM_CSR_ALG2);
  gpuCuBlasSpMMAlg2->run();
  //  gpuCuBlasSpMMDefault->OutTensor->printDx();
  auto gpuCuBlasSpMMAlg2Stat = gpuCuBlasSpMMAlg2->printStats();
  delete gpuCuBlasSpMMAlg2;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_GeSpMM_CSR_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuGeSpMM = new GpuGeSpMM(inSpMM,stats);
  gpuGeSpMM->run();
//  gpuGeSpMM->OutTensor->printDx();
  auto gpuGeSpMMStat = gpuGeSpMM->printStats();
  delete gpuGeSpMM;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_ParReduceRowBalance_CSR_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuParReduceRowBalanceSpMM = new GpuParReduceRowBalance(inSpMM,stats);
  gpuParReduceRowBalanceSpMM->run();
  //  gpuParReduceRowBalanceSpMM->OutTensor->printDx();
  auto gpuParReduceRowBalanceSpMMStat = gpuParReduceRowBalanceSpMM->printStats();
  delete gpuParReduceRowBalanceSpMM;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_ParReduceNNZBalance_CSR_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuParReduceNNZBalanceSpMM = new GpuParReduceNnzBalance(inSpMM,stats);
  gpuParReduceNNZBalanceSpMM->run();
  //  gpuParReduceNNZBalanceSpMM->OutTensor->printDx();
  auto gpuParReduceNNZBalanceSpMMStat = gpuParReduceNNZBalanceSpMM->printStats();
  delete gpuParReduceNNZBalanceSpMM;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_SeqReduceRowBalance_CSR_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuSeqReduceRowBalanceSpMM = new GpuSeqReduceRowBalance(inSpMM,stats);
  gpuSeqReduceRowBalanceSpMM->run();
  //  gpuSeqReduceRowBalanceSpMM->OutTensor->printDx();
  auto gpuSeqReduceRowBalanceSpMMStat = gpuSeqReduceRowBalanceSpMM->printStats();
  delete gpuSeqReduceRowBalanceSpMM;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_SeqReduceNNZBalance_CSR_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuSeqReduceNNZBalanceSpMM = new GpuSeqReduceNnzBalance(inSpMM,stats);
  gpuSeqReduceNNZBalanceSpMM->run();
  //  gpuSeqReduceNNZBalanceSpMM->OutTensor->printDx();
  auto gpuSeqReduceNNZBalanceSpMMStat = gpuSeqReduceNNZBalanceSpMM->printStats();
  delete gpuSeqReduceNNZBalanceSpMM;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_RowCachingRowBalance_CSR_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuRowCachingRowBalanceSpMM = new GpuRowCachingRowBalance(inSpMM,stats);
  gpuRowCachingRowBalanceSpMM->run();
  //  gpuRowCachingRowBalanceSpMM->OutTensor->printDx();
  auto gpuRowCachingRowBalanceSpMMStat = gpuRowCachingRowBalanceSpMM->printStats();
  delete gpuRowCachingRowBalanceSpMM;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_RowCachingNNZBalance_CSR_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuRowCachingNNZBalanceSpMM = new GpuRowCachingNnzBalance(inSpMM,stats);
  gpuRowCachingNNZBalanceSpMM->run();
  //  gpuRowCachingNNZBalanceSpMM->OutTensor->printDx();
  auto gpuRowCachingNNZBalanceSpMMStat = gpuRowCachingNNZBalanceSpMM->printStats();
  delete gpuRowCachingNNZBalanceSpMM;
  delete stats;



  std::string profHeader = "";
  std::string profStat = "";

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if (tp.print_header)
    std::cout << headerStat + spHeader + tpHeader + profHeader << std::endl;
  std::cout << cpuSpMMStat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuCuBlasSpMMDefaultStat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuCuBlasSpMMAlg3Stat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuCuBlasSpMMAlg2Stat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuGeSpMMStat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuParReduceRowBalanceSpMMStat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuParReduceNNZBalanceSpMMStat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuSeqReduceRowBalanceSpMMStat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuSeqReduceNNZBalanceSpMMStat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuRowCachingRowBalanceSpMMStat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuRowCachingNNZBalanceSpMMStat << spStat + tpStat + profStat << std::endl;



  delete aCSCFull;
  delete aCSC;

}
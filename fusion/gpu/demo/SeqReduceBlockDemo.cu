 //
// Created by salehm32 on 21/06/24.
//

#include "../Cuda_SpMM_Demo_Utils.h"
#include "Stats.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Defs.h"
#include "sparse-fusion/Fusion_Utils.h"
#include <iostream>

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

  stats = new swiftware::benchmark::Stats("CPU_SpMM", "SpMM", numTrial,
                                          tp._matrix_name, numThread);
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

  stats = new swiftware::benchmark::Stats("GPU_SeqReduceRowBalance_128","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuSeqReduceRowBalanceSpMM128 = new GpuSeqReduceRowBalanceVariableThreadPerBlock(inSpMM,stats,128);
  gpuSeqReduceRowBalanceSpMM128->run();
//  gpuSeqReduceRowBalanceSpMM128->OutTensor->printDx();
  auto gpuSeqReduceRowBalanceSpMM128Stat =
      gpuSeqReduceRowBalanceSpMM128->printStats();
  delete gpuSeqReduceRowBalanceSpMM128;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_SeqReduceRowBalance_256","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuSeqReduceRowBalanceSpMM256 = new GpuSeqReduceRowBalanceVariableThreadPerBlock(inSpMM,stats,256);
  gpuSeqReduceRowBalanceSpMM256->run();
//  gpuSeqReduceRowBalanceSpMM256->OutTensor->printDx();
  auto gpuSeqReduceRowBalanceSpMM256Stat =
      gpuSeqReduceRowBalanceSpMM256->printStats();
  delete gpuSeqReduceRowBalanceSpMM256;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_SeqReduceRowBalance_512","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuSeqReduceRowBalanceSpMM512 = new GpuSeqReduceRowBalanceVariableThreadPerBlock(inSpMM,stats,512);
  gpuSeqReduceRowBalanceSpMM512->run();
//  gpuSeqReduceRowBalanceSpMM512->OutTensor->printDx();
  auto gpuSeqReduceRowBalanceSpMM512Stat =
      gpuSeqReduceRowBalanceSpMM512->printStats();
  delete gpuSeqReduceRowBalanceSpMM512;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_SeqReduceRowBalance_1024","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuSeqReduceRowBalanceSpMM1024 = new GpuSeqReduceRowBalanceVariableThreadPerBlock(inSpMM,stats,1024);
  gpuSeqReduceRowBalanceSpMM1024->run();
//  gpuSeqReduceRowBalanceSpMM1024->OutTensor->printDx();
  auto gpuSeqReduceRowBalanceSpMM1024Stat =
      gpuSeqReduceRowBalanceSpMM1024->printStats();
  delete gpuSeqReduceRowBalanceSpMM1024;
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
  std::cout << gpuSeqReduceRowBalanceSpMM128Stat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuSeqReduceRowBalanceSpMM256Stat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuSeqReduceRowBalanceSpMM512Stat << spStat + tpStat + profStat << std::endl;
  std::cout << gpuSeqReduceRowBalanceSpMM1024Stat << spStat + tpStat + profStat << std::endl;

  delete inSpMM;
  delete aCSC;
  delete aCSCFull;

}
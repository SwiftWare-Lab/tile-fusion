//
// Created by salehm32 on 30/01/25.
//

#include "../Cuda_GeMM_Demo_Utils.h"
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
      new CudaGeMMSpMMTensorInputs(aCSCFull->m, tp._b_cols, aCSCFull->m, tp._b_cols,
                           aCSCFull, aCSCFull, numThread, numTrial, expName);

  stats = new swiftware::benchmark::Stats("CPU_GeMM_Demo", "SpMM", numTrial,
                                          tp._matrix_name, numThread);
  auto *cpuGeMM = new GeMMCPUParallel(inSpMM, stats);
  cpuGeMM->run();
//    cpuGeMM->OutTensor->printDx();
  std::copy(cpuGeMM->OutTensor->ACx,
            cpuGeMM->OutTensor->ACx +
                cpuGeMM->OutTensor->M * cpuGeMM->OutTensor->N,
            inSpMM->CorrectSol);
  inSpMM->IsSolProvided = true;
  auto headerStat = cpuGeMM->printStatsHeader();
  auto cpuGeMMStat = cpuGeMM->printStats();
  delete cpuGeMM;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_GeMM_2DBlocking_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuGeMM = new GeMMCuda2DBlocking(inSpMM,stats);
  gpuGeMM->run();
//    gpuGeMM->OutTensor->printDx();
  auto gpuGeMMStat = gpuGeMM->printStats();
  delete gpuGeMM;
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
  std::cout << cpuGeMMStat + spStat + tpStat + profStat << std::endl;
  std::cout << gpuGeMMStat + spStat + tpStat + profStat << std::endl;

  delete inSpMM;
  delete aCSCFull;
  delete aCSC;

}
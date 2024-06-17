//
// Created by mehdi on 6/16/24.
//
#include "Timer.h"
#include "Stats.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Defs.h"
#include "sparse-fusion/Fusion_Utils.h"
#include <cusparse.h>
#include <iostream>
#include "Cuda_SpMM_Demo_Utils.h"

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

  stats = new swiftware::benchmark::Stats("GPU_cuBlas_SpMM_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *gpuCuBlasSpMM = new GpuSpMMCuBlas(inSpMM,stats);
  gpuCuBlasSpMM->run();
//  gpuCuBlasSpMM->OutTensor->printDx();
  auto gpuCuBlasSpMMVTStat = gpuCuBlasSpMM->printStats();
  delete gpuCuBlasSpMM;
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
  std::cout << gpuCuBlasSpMMVTStat << spStat + tpStat + profStat << std::endl;

  std::cout << sizeof(int) << std::endl;

  delete aCSCFull;
  delete aCSC;

}
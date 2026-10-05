//
// Created by salehm32 on 09/01/25.
//

#include "../Benchmarks.h"
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
  stats = new swiftware::benchmark::Stats("Filter_K", "F", numTrial, tp._matrix_name, numThread);
  auto *filterK = new FilterKBench(inSpMM, stats, 1e6);
  filterK->run();
  auto headerStat = filterK->printStatsHeader();
  auto filterKStat = filterK->printStats();
  delete filterK;
  delete stats;

  stats = new swiftware::benchmark::Stats("Filter_K_No_Atomic", "F", numTrial, tp._matrix_name, numThread);
  auto *filterKNoAtomic = new FilterKNoAtomicBench(inSpMM, stats, 1e6);
  filterKNoAtomic->run();
  auto filterKNoAtomicStat = filterKNoAtomic->printStats();
  delete filterKNoAtomic;
  delete stats;

  std::cout << headerStat << std::endl;
  std::cout << filterKStat << std::endl;
  std::cout << filterKNoAtomicStat << std::endl;


  delete inSpMM;
  delete aCSC;
  delete aCSCFull;

  return 0;

}
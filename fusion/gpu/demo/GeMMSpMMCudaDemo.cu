//
// Created by salehm32 on 30/01/25.
//

#include "../Cuda_GeMM_SpMM_Demo_Utils.h"

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
      new CudaGeMMSpMMTensorInputs(aCSCFull->m, tp._b_cols, tp._b_cols, aCSCFull->m,
                                   aCSCFull, aCSCFull, numThread, numTrial, expName);

  stats = new swiftware::benchmark::Stats("CPU_GeMM_SpMM_Demo", "SpMM", numTrial,
                                          tp._matrix_name, numThread);
  auto *geMMSpMMCPU = new GeMMSpMMCPU(inSpMM, stats);
  geMMSpMMCPU->run();
//      geMMSpMMCPU->OutTensor->printDx();
  std::copy(geMMSpMMCPU->OutTensor->Xx,
            geMMSpMMCPU->OutTensor->Xx +
                geMMSpMMCPU->OutTensor->M * geMMSpMMCPU->OutTensor->N,
            inSpMM->CorrectSol);
  auto headerStat = geMMSpMMCPU->printStatsHeader();
  auto cpuGeMMSpMMStat = geMMSpMMCPU->printStats();
  delete geMMSpMMCPU;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Unfused_GeMMSpMM_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedGeMMSpMM = new UnfusedGeMMSpMMGPU(inSpMM,stats);
  unfusedGeMMSpMM->run();
  std::copy(unfusedGeMMSpMM->OutTensor->Xx,
            unfusedGeMMSpMM->OutTensor->Xx +
                unfusedGeMMSpMM->OutTensor->M * unfusedGeMMSpMM->OutTensor->N,
            inSpMM->CorrectSol);
  inSpMM->IsSolProvided = true;
//      unfusedGeMMSpMM->OutTensor->printDx();
  auto unfusedGeMMSpMMStat = unfusedGeMMSpMM->printStats();
  delete unfusedGeMMSpMM;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Fused_GeMMSpMM_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedGeMMSpMM = new FusedGeMMSpMMGPU(inSpMM,stats);
  fusedGeMMSpMM->run();
//      fusedGeMMSpMM->OutTensor->printDx();
  auto fusedGeMMSpMMStat = fusedGeMMSpMM->printStats();
  delete fusedGeMMSpMM;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Fused_SpMMGeMM_Demo","SpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedSpMMGeMM = new FusedSpMMGeMMGPU(inSpMM,stats);
  fusedSpMMGeMM->run();
//  fusedSpMMGeMM->OutTensor->printDx();
  auto fusedSpMMGeMMStat = fusedSpMMGeMM->printStats();
  delete fusedSpMMGeMM;
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
  std::cout << cpuGeMMSpMMStat + spStat + tpStat + profStat << std::endl;
  std::cout << unfusedGeMMSpMMStat + spStat + tpStat + profStat << std::endl;
  std::cout << fusedGeMMSpMMStat + spStat + tpStat + profStat << std::endl;
  std::cout << fusedSpMMGeMMStat + spStat + tpStat + profStat << std::endl;

  delete inSpMM;
  delete aCSCFull;
  delete aCSC;

}
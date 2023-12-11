//
// Created by salehm32 on 08/12/23.
//

#include "aggregation/def.h"
#include "SWTensorBench.h"
#include "SpMV_SpMV_Demo_Utils.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/SparseFusion.h"
#include <fstream>

using namespace sym_lib;

int main(const int argc, const char *argv[]) {
  TestParameters tp;
  tp._order_method = SYM_ORDERING::NONE;
  ScheduleParameters sp;
  Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  CSC *aLtCsc = NULLPNTR;
  CSC *aCSC = get_matrix_from_parameter(&tp);
  if (aCSC->m != aCSC->n) {
    return -1;
  }
  CSC *aCSCFull = nullptr;
  if (aCSC->stype == -1 || aCSC->stype == 1) {
    aCSCFull = sym_lib::make_full(aCSC);
  } else {
    aCSCFull = sym_lib::copy_sparse(aCSC);
  }
  tp._dim1 = aCSCFull->m;
  tp._dim2 = aCSCFull->n;
  tp._nnz = aCSCFull->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);

  CSC *bCSC = sym_lib::copy_sparse(aCSCFull);
  auto *alCSC = make_half(aCSC->n, aCSC->p, aCSC->i, aCSC->x);
  std::vector<CSC *> orderedVec;
  if (tp._order_method != SYM_ORDERING::NONE) {
    // applies ordering here
    get_reorderd_matrix(alCSC, orderedVec);
    delete alCSC;
    alCSC = orderedVec[0];
  }
  int numThread = sp._num_threads, numTrial = 7; std::string expName = "SpMV_SpMV_Demo";
  auto *inSpMM = new TensorInputs<double>(aCSCFull->m,  tp._b_cols, aCSCFull->n,
                                          bCSC->m, aCSCFull, bCSC,
                                          numThread, numTrial, expName);
  stats = new swiftware::benchmark::Stats("SpMV_SpMV_Sequential", "SpMV", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *unfused = new SpMVSpMVSequential(inSpMM, stats);
  unfused->run();
  //unfused->OutTensor->printDx();
  inSpMM->CorrectSol = std::copy(unfused->OutTensor->Dx, unfused->OutTensor->Dx + unfused->OutTensor->M, inSpMM->CorrectMul);
  inSpMM->IsSolProvided = true;
//  unfused->OutTensor->printDx();
  auto headerStat = unfused->printStatsHeader();
  auto baselineStat = unfused->printStats();
  delete unfused;
  delete stats;

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if(tp.print_header)
    std::cout<<headerStat+spHeader+tpHeader<<std::endl;
  std::cout<<baselineStat<<spStat+tpStat<<std::endl;

}
//
// Created by kazem on 1/19/24.
//

#include "Inspection/GraphColoring.h"
#include "Stats.h"
#include "aggregation/def.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "jacobi_demo_utils.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/SparseFusion.h"
#include <fstream>

using namespace sym_lib;

int main(const int argc, const char *argv[]) {
  TestParameters tp;
  tp._order_method = SYM_ORDERING::NONE;
  ScheduleParameters sp;
  swiftware::benchmark::Stats *stats;
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

  // print_csc(1,"",aCSC);
  int numThread = sp._num_threads, numTrial = 7;
  std::string expName = "Jacobi_CSR_Demo";
  auto *inJacobi =
      new TensorInputs<double>(aCSCFull->m, tp._b_cols, aCSCFull, numThread, 1,
                               expName);

  stats = new swiftware::benchmark::Stats("Jacobi_Demo", "Jacobi CSR", 1,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *unfused = new JacobiCSRUnfused(inJacobi, stats);
  unfused->run();
  //unfused->OutTensor->printDx();
  inJacobi->CorrectSol = std::copy(
      unfused->OutTensor->Xx2,
      unfused->OutTensor->Xx2 + unfused->OutTensor->M * unfused->OutTensor->K,
      inJacobi->CorrectSol);
  inJacobi->IsSolProvided = true;
//  unfused->OutTensor->printDx();
  auto headerStat = unfused->printStatsHeader();
  auto baselineStat = unfused->printStats();
  std::string unfusedRetValue = std::to_string(unfused->OutTensor->RetValue);
  delete unfused;
  delete stats;

  stats = new swiftware::benchmark::Stats("Jacobi_BiIterationFused_Demo", "Jacobi CSR", 1,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *fused = new JacobiCSRFused(inJacobi, stats, sp);
  fused->run();
//  fused->OutTensor->printDx();
  auto fusedStat = fused->printStats();
  std::string fusedRetValue = std::to_string(fused->OutTensor->RetValue);
  delete fused;
  delete stats;

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if(tp.print_header){
    std::cout << headerStat+spHeader+tpHeader+",RetValue," << std::endl;
  }
  std::cout << baselineStat+spStat+tpStat+unfusedRetValue+',' << std::endl;
  std::cout << fusedStat+spStat+tpStat+unfusedRetValue+',' << std::endl;

  delete aCSC;
  delete aCSCFull;
  delete bCSC;
  delete alCSC;
  //delete inJacobi;
  return 0;
}
//
// Created by kazem on 08/07/23.
//



#include "aggregation/def.h"
#include "SpMM_SpMM_Demo_Utils.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/SparseFusion.h"
#include <fstream>

using namespace sym_lib;
// A is MxK, C is KxN, B is LxM, and D is LxN; AC is MxN
int main(const int argc, const char *argv[]){
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
  //  delete aCSC;
  //  auto *alCSC = make_half(aCSC->n, aCSC->p, aCSC->i, aCSC->x);
  //  std::vector<CSC*> orderedVec;
  //  if(tp._order_method != SYM_ORDERING::NONE){
  //    // applies ordering here
  //    get_reorderd_matrix(alCSC, orderedVec);
  //    delete alCSC;
  //    alCSC = orderedVec[0];
  //  }

  // print_csc(1,"",aCSC);
  int numThread = sp._num_threads, numTrial = 1;
  std::string expName = "SpMM_SpMM_Demo";
  auto *inSpMM =
      new TensorInputs<double>(aCSCFull->m, tp._b_cols, aCSCFull->n, aCSCFull->m,
                               aCSCFull, aCSCFull, numThread, numTrial, expName);
  //  DsaturColoringForConflictGraph *dsaturColoring =
  //      new DsaturColoringForConflictGraph();
  //  std::map<int, std::vector<int>> colorToTiles =
  //      dsaturColoring->generateGraphColoringForConflictGraphOf(
  //          aCSCFull, sp.IterPerPartition, true);
  delete aCSCFull;
  delete aCSC;
  //print_csc(1,"",aCSC);

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo", "SpMM", numTrial, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *fusionProfiler = new SpMMSpMMFusionProfiler(inSpMM, stats, sp);
  fusionProfiler->run();
  //unfused->OutTensor->printDx();
  std::copy(fusionProfiler->OutTensor->Xx,
                fusionProfiler->OutTensor->Xx +
                    fusionProfiler->OutTensor->M * fusionProfiler->OutTensor->N, inSpMM->CorrectSol);
  inSpMM->IsSolProvided = true;
  auto headerStat = fusionProfiler->printStatsHeader();
  auto baselineStat = fusionProfiler->printStats();
  delete fusionProfiler;
  delete stats;

  if(tp.print_header){
    std::cout << headerStat << std::endl;
  }
  std::cout << baselineStat << std::endl;
  delete inSpMM;

  return 0;
}

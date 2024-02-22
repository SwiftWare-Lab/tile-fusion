//
// Created by kazem on 19/02/24.
//


#include "SpMM_SpMM_Demo_Utils.h"
#include "aggregation/def.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/SparseFusion.h"
#include <fstream>

using namespace sym_lib;

// A is MxK, C is KxN, B is LxM, and D is LxN; AC is MxN
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
  int numThread = sp._num_threads, numTrial = 7;
  std::string expName = "SpMM_SpMM_Demo";
  auto *inSpMM =
      new TensorInputs<double>(aCSCFull->m, tp._b_cols, aCSCFull->n, bCSC->m,
                               aCSCFull, bCSC, numThread, numTrial, expName);
  //  DsaturColoringForConflictGraph *dsaturColoring =
  //      new DsaturColoringForConflictGraph();
  //  std::map<int, std::vector<int>> colorToTiles =
  //      dsaturColoring->generateGraphColoringForConflictGraphOf(
  //          aCSCFull, sp.IterPerPartition, true);
  delete aCSCFull;
  delete bCSC;
  delete aCSC;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel","SpMM",
                                          7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] ={Interleaved};
  auto *fusedParallel = new SpMMSpMMFusedInterLayer(inSpMM,
                                                    stats, sp); fusedParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelStat = fusedParallel->printStats();
  auto headerStat = fusedParallel->printStatsHeader();
  delete fusedParallel;
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
  //  std::cout<<baselineStat<<spStat+tpStat+profStat<<std::endl;
  std::cout << fusedParallelStat << spStat + tpStat + profStat << std::endl;


#ifdef __AVX2__
  stats = new
      swiftware::benchmark::Stats("SpMM_SpMM_FusedParallelAvx256","SpMM",
                                  7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
      {Interleaved}; auto *fusedParallelVectorized256 = new
      SpMMSpMMFusedInterLayerVectorizedAvx256(inSpMM, stats, sp);
  fusedParallelVectorized256->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelVectorized256Stat =
      fusedParallelVectorized256->printStats(); delete
      fusedParallelVectorized256; delete stats;
  std::cout<<fusedParallelVectorized256Stat<<spStat+tpStat+profStat<<std::endl;

  stats = new
      swiftware::benchmark::Stats("SpMM_SpMM_FusedParallelKTiledAvx256","SpMM",
                                  7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
      {Interleaved}; auto *fusedParallelVectorizedKTiled256 = new
      SpMMSpMMFusedInterLayerKTiled8VectorizedAvx256(inSpMM, stats, sp);
  fusedParallelVectorizedKTiled256->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelVectorizedKTiled256Stat =
      fusedParallelVectorizedKTiled256->printStats(); delete fusedParallelVectorizedKTiled256; delete stats;
  std::cout<<fusedParallelVectorizedKTiled256Stat<<spStat+tpStat+profStat<<std::endl;

#endif

  delete inSpMM;
  //  delete dsaturColoring;
  //  delete dsaturColoringWithKTiling;

  return 0;
}

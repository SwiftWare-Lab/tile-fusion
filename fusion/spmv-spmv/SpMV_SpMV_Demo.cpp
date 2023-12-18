//
// Created by salehm32 on 08/12/23.
//

#include "../gcn/Inspection/GraphColoring.h"
#include "SWTensorBench.h"
#include "SpMV_SpMV_Demo_Utils.h"
#include "aggregation/def.h"
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
  stats = new swiftware::benchmark::Stats("SpMV_SpMV_UnFusedSequential", "SpMV", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *unfused = new SpMVSpMVUnFusedSequential(inSpMM, stats);
  unfused->run();
  inSpMM->CorrectSol = std::copy(unfused->OutTensor->Dx, unfused->OutTensor->Dx + unfused->OutTensor->M, inSpMM->CorrectMul);
  inSpMM->IsSolProvided = true;
  auto headerStat = unfused->printStatsHeader();
  auto baselineStat = unfused->printStats();
  delete unfused;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMV_SpMV_UnFusedParallel", "SpMV", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *spmvParallel = new SpMVSpMVUnFusedParallel(inSpMM, stats);
  spmvParallel->run();
  auto spMVParallelStat = spmvParallel->printStats();
  delete spmvParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMV_SpMV_FusedParallel", "SpMV", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *spmvFused = new SpMVSpMVFused(inSpMM, stats, sp);
  spmvFused->run();
  auto spMVFusedStat = spmvFused->printStats();
  delete spmvFused;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMV_SpMV_FusedParallel_RegisterReuse_BandedSpecific", "SpMV", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *spmvFusedRR = new SpMVSpMVFusedInterleaved(inSpMM, stats, sp);
  spmvFusedRR->run();
//  for (int i = 0; i < inSpMM->M; i++){
//    std::cout << spmvFusedRR->OutTensor->ACx[i]<< ", ";
//  }
//  std::cout << std::endl;
//  for (int i = 0; i < inSpMM->M; i++){
//    std::cout << spmvFusedRR->OutTensor->Dx[i] << ", ";
//  }
//  std::cout << std::endl;
  auto spMVFusedRRStat = spmvFusedRR->printStats();
  delete spmvFusedRR;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMV_SpMV_FusedParallel_Redundant_BandedSpecific", "SpMV", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *spmvFusedRB = new SpMVSpMVFusedTiledTri(inSpMM, stats, sp);
  spmvFusedRB->run();
  auto spMVFusedRBStat = spmvFusedRB->printStats();
  delete spmvFusedRB;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMV_SpMV_FusedParallel_Redundant_General", "SpMV", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *spmvFusedRG = new SpMVSpMVFusedTiledRedundant(inSpMM, stats, sp);
  spmvFusedRG->run();
  auto spMVFusedRGStat = spmvFusedRG->printStats();
  delete spmvFusedRG;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMV_SpMV_FusedParallel_Separated", "SpMV", 7, tp._matrix_name, numThread);
  auto *spmvFusedSeparated = new SpMVSpMVFusedParallelSeparated(inSpMM, stats, sp);
  spmvFusedSeparated->run();
  auto spMVFusedSeparatedStat = spmvFusedSeparated->printStats();
  delete spmvFusedSeparated;
  delete stats;


  int tileSize = sp.TileM;
  DsaturColoringForConflictGraph *dsaturColoring =
      new DsaturColoringForConflictGraph();


  std::map<int, std::vector<int>> colorToTiles =
      dsaturColoring->generateGraphColoringForConflictGraphOf(aCSCFull,
                                                              tileSize, true);

  stats = new swiftware::benchmark::Stats("SpMV_SpMV_CSC_Interleaved_Coloring_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  auto *fusedCSCInterleavedColoringParallel = new SpMVCSRSpMVCSCFusedColoring(inSpMM, stats, sp, tileSize,
                                                                              colorToTiles);
  fusedCSCInterleavedColoringParallel->run();
  auto fusedCSCInterleavedColoringParallelStat = fusedCSCInterleavedColoringParallel->printStats();
  delete fusedCSCInterleavedColoringParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMV_SpMV_CSC_Interleaved_ColoringReduction_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  auto *fusedCSCInterleavedColoringReductionParallel = new SpMVCSRSpMVCSCFusedColoringWithReduction(inSpMM, stats, sp, tileSize,
                                                                              colorToTiles);
  fusedCSCInterleavedColoringReductionParallel->run();
  auto fusedCSCInterleavedColoringReductionParallelStat = fusedCSCInterleavedColoringReductionParallel->printStats();
  delete fusedCSCInterleavedColoringReductionParallel;
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
  std::cout << spMVParallelStat<<spStat+tpStat<<std::endl;
  std::cout << spMVFusedStat<<spStat+tpStat<<std::endl;
  std::cout << spMVFusedRRStat<<spStat+tpStat<<std::endl;
  std::cout << spMVFusedRBStat<<spStat+tpStat<<std::endl;
  std::cout << spMVFusedRGStat<<spStat+tpStat<<std::endl;
  std::cout << spMVFusedSeparatedStat<<spStat+tpStat<<std::endl;
  std::cout << fusedCSCInterleavedColoringParallelStat<<spStat+tpStat<<std::endl;
  std::cout << fusedCSCInterleavedColoringReductionParallelStat<<spStat+tpStat<<std::endl;
#ifdef MKL
  stats = new swiftware::benchmark::Stats("SpMV_SpMV_UnFusedMKL", "SpMV", 7, tp._matrix_name, numThread);
  auto spmvMKL = new SpMVSpMVMkl(inSpMM, stats);
  spmvMKL->run();
  auto spMVKLStat = spmvMKL->printStats();
  delete spmvMKL;
  delete stats;

  std::cout << spMVKLStat<<spStat+tpStat<<std::endl;
#endif //MKL

  delete inSpMM;
  delete aCSC;
  delete aCSCFull;
  delete bCSC;
  delete dsaturColoring;
  return 0;
}
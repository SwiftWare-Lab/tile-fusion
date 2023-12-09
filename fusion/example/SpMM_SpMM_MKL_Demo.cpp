//
// Created by mehdi on 5/25/23.
//

#include "SpMM_SpMM_Demo_Utils.h"
#include "SpMM_SpMM_MKL_Demo_Utils.h"
#include "aggregation/def.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "Inspection/GraphColoring.h"
#include <fstream>

using namespace sym_lib;

// A is MxK, C is KxN, B is LxM, and D is LxN; AC is MxN
int main(const int argc, const char *argv[]){
  TestParameters tp;tp._order_method=SYM_ORDERING::NONE;
  ScheduleParameters sp;
  Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  CSC *aLtCsc=NULLPNTR;
  CSC *aCSC = get_matrix_from_parameter(&tp);
  if(aCSC->m != aCSC->n){
    return -1;
  }
  CSC *aCSCFull = nullptr;
  if(aCSC->stype == -1 || aCSC->stype == 1){
    aCSCFull = sym_lib::make_full(aCSC);
  } else{
    aCSCFull = sym_lib::copy_sparse(aCSC);
  }
  tp._dim1 = aCSCFull->m; tp._dim2 = aCSCFull->n; tp._nnz = aCSCFull->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);

  CSC *bCSC = sym_lib::copy_sparse(aCSCFull);
  auto *alCSC = make_half(aCSC->n, aCSC->p, aCSC->i, aCSC->x);
  std::vector<CSC*> orderedVec;
  if(tp._order_method != SYM_ORDERING::NONE){
    // applies ordering here
    get_reorderd_matrix(alCSC, orderedVec);
    delete alCSC;
    alCSC = orderedVec[0];
  }

  int numThread = sp._num_threads, numTrial = 7; std::string expName = "SpMM_SpMM_Demo";
  auto *inSpMM = new TensorInputs<double>(aCSCFull->m,  tp._b_cols, aCSC->n,
                                          bCSC->m, aCSCFull, bCSC,
                                          numThread, numTrial, expName);

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo", "SpMM", 7, tp._matrix_name, numThread);
  auto *unfused = new SpMMSpMMUnFused(inSpMM, stats);
  stats->OtherStats["PackingType"] = {Separated};
  unfused->run();
  inSpMM->CorrectSol = std::copy(unfused->OutTensor->Dx, unfused->OutTensor->Dx + unfused->OutTensor->M * unfused->OutTensor->N, inSpMM->CorrectMul);
  inSpMM->IsSolProvided = true;
  auto headerStat = unfused->printStatsHeader();
  auto baselineStat = unfused->printStats();
  delete unfused;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo_UnFusedParallel", "SpMM", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *unfusedParallel = new SpMMSpMMUnFusedParallel(inSpMM, stats);
  unfusedParallel->run();
  //unfusedParallel->OutTensor->printDx();
  auto unfusedParallelStat = unfusedParallel->printStats();
  delete unfusedParallel;
  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_MKL", "SpMM", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  auto *mklImpl = new SpMMSpMMMKL(inSpMM, stats);
  mklImpl->run();
  auto mklImplStat = mklImpl->printStats();
  delete mklImpl;
  delete stats;

  int tileSize = sp.TileM;
  DsaturColoringForConflictGraph *dsaturColoring =
      new DsaturColoringForConflictGraph();
  DsaturColoringForConflictGraphWithKTiling *dsaturColoringWithKTiling =
      new DsaturColoringForConflictGraphWithKTiling();


  std::map<int, std::vector<int>> colorToTiles =
      dsaturColoring->generateGraphColoringForConflictGraphOf(aCSCFull,
                                                              tileSize, true);
  //  for (auto ct: colorToTiles){
  //    std::cout << ct.first << std::endl;
  //  }

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_CSC_Interleaved_Coloring_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Separated};
  auto *fusedCSCInterleavedColoringParallel = new SpMMCSRSpMMCSCFusedColoring(inSpMM, stats, sp, tileSize,
                                                                              colorToTiles);
  fusedCSCInterleavedColoringParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedCSCInterleavedColoringParallelStat = fusedCSCInterleavedColoringParallel->printStats();
  delete fusedCSCInterleavedColoringParallel;
  delete stats;


  std::vector<std::string> scheduledKTilingStats;
  std::vector<std::string> replicatedKTilingStats;
  std::vector<std::string> fusedKTiledStats;
  for(int i = 2; pow(2,i) < inSpMM->N; i++){
    int kTileSize = pow(2,i);
    std::map<int, std::vector<int>> colorToTilesForKTiling =
        dsaturColoringWithKTiling->generateGraphColoringForConflictGraphOf(aCSCFull, tileSize, inSpMM->N, kTileSize, true);
    stats = new swiftware::benchmark::Stats("SpMM_SpMM_CSC_Interleaved_Coloring_FusedParallel_ScheduledKTiling","SpMM", 7,tp._matrix_name,numThread);
    stats->OtherStats["PackingType"] = {Separated};
    auto *fusedCSCInterleavedColoringParallelScheduledKTiling = new SpMMCSRSpMMCSCFusedColoringWithScheduledKTiling(inSpMM, stats, sp, tileSize,
                                                                                                                    colorToTilesForKTiling, kTileSize);
    stats->OtherStats["NTile"] = {(double)kTileSize};
    fusedCSCInterleavedColoringParallelScheduledKTiling->run();
    //fusedParallel->OutTensor->printDx();
    scheduledKTilingStats.push_back(
        fusedCSCInterleavedColoringParallelScheduledKTiling->printStats());
    delete fusedCSCInterleavedColoringParallelScheduledKTiling;
    delete stats;

    stats = new swiftware::benchmark::Stats("SpMM_SpMM_CSC_Interleaved_Coloring_FusedParallel_ReplicatedKTiling","SpMM", 7,tp._matrix_name,numThread);
    stats->OtherStats["PackingType"] = {Separated};
    auto *fusedCSCInterleavedColoringParallelKTiling = new SpMMCSRSpMMCSCFusedColoringWithReplicatedKTiling(inSpMM, stats, sp, tileSize,
                                                                                                            colorToTiles, kTileSize);
    fusedCSCInterleavedColoringParallelKTiling->run();
    stats->OtherStats["NTile"] = {(double)kTileSize};
    //fusedParallel->OutTensor->printDx();
    replicatedKTilingStats.push_back(fusedCSCInterleavedColoringParallelKTiling->printStats());
    delete fusedCSCInterleavedColoringParallelKTiling;
    delete stats;

    stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel_KTiled","SpMM", 7,tp._matrix_name,numThread);
    stats->OtherStats["PackingType"] = {Interleaved};
    auto *fusedParallelKTiled = new SpMMSpMMFusedInterLayerKTiled(inSpMM, stats, sp, kTileSize);
    stats->OtherStats["NTile"] = {(double)kTileSize};
    fusedParallelKTiled->run();
    //fusedParallel->OutTensor->printDx();
    fusedKTiledStats.push_back(fusedParallelKTiled->printStats());
    delete fusedParallelKTiled;
    delete stats;

  }


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel", "SpMM", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *fusedParallel = new SpMMSpMMFusedInterLayer(inSpMM, stats, sp);
  fusedParallel->run();
  auto fusedParallelStat = fusedParallel->printStats();
  delete fusedParallel;
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
  std::cout<<mklImplStat<<spStat+tpStat<<std::endl;
  std::cout<<fusedParallelStat<<spStat+tpStat<<std::endl;
  std::cout<<unfusedParallelStat<<spStat+tpStat<<std::endl;
  std::cout<<fusedCSCInterleavedColoringParallelStat << spStat+tpStat<<std::endl;
  for (auto stat: scheduledKTilingStats){
    std::cout<<stat<<spStat+tpStat<<std::endl;
  }
  for (auto stat: replicatedKTilingStats){
    std::cout<<stat<<spStat+tpStat<<std::endl;
  }
  for (auto stat: fusedKTiledStats){
    std::cout<<stat<<spStat+tpStat<<std::endl;
  }




  delete aCSC;
  delete bCSC;
  delete alCSC;
  delete inSpMM;

  return 0;
}

//
// Created by kazem on 02/05/23.
//

#include "Inspection/GraphColoring.h"
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
  //  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo", "SpMM", 7,
  //  tp._matrix_name, numThread); stats->OtherStats["PackingType"] =
  //  {Interleaved}; auto *unfused = new SpMMSpMMUnFused(inSpMM, stats);
  //  unfused->run();
  //  //unfused->OutTensor->printDx();
  //  std::copy(unfused->OutTensor->Dx, unfused->OutTensor->Dx +
  //  unfused->OutTensor->M * unfused->OutTensor->N, inSpMM->CorrectMul);
  //  inSpMM->IsSolProvided = true;
  //  auto headerStat = unfused->printStatsHeader();
  //  auto baselineStat = unfused->printStats();
  //  delete unfused;
  //  delete stats;

  stats = new swiftware::benchmark::Stats(
      "SpMM_SpMM_Demo_UnFusedParallel", "SpMM", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  stats->OtherStats["TilingMethod"] = {Fixed};
  auto *unfusedParallel = new SpMMSpMMUnFusedParallel(inSpMM, stats);
  unfusedParallel->run();
  //  unfusedParallel->OutTensor->printDx();
  std::copy(unfusedParallel->OutTensor->Xx,
            unfusedParallel->OutTensor->Xx +
                unfusedParallel->OutTensor->M * unfusedParallel->OutTensor->N,
            inSpMM->CorrectSol);
  inSpMM->IsSolProvided = true;
  auto headerStat = unfusedParallel->printStatsHeader();
  auto unfusedParallelStat = unfusedParallel->printStats();
  delete unfusedParallel;
  delete stats;

  //  stats = new
  //  swiftware::benchmark::Stats("SpMM_SpMM_Demo_InnerProduct_UnFusedParallel",
  //  "SpMM", 7, tp._matrix_name, numThread); stats->OtherStats["PackingType"] =
  //  {Interleaved}; auto *unfusedOutParallel = new
  //  SpMMSpMMUnFusedInnerParallel(inSpMM, stats); unfusedOutParallel->run();
  //  //unfusedParallel->OutTensor->printDx();
  //  auto unfusedOutParallelStat = unfusedOutParallel->printStats();
  //  delete unfusedOutParallel;
  //  delete stats;

  // sp.TileM = std::min(sp.IterPerPartition, inSpMM->M);

  //  stats = new
  //  swiftware::benchmark::Stats("SpMM_SpMM_Demo_CTiled_UnFusedParallel",
  //  "SpMM", 7, tp._matrix_name, numThread); stats->OtherStats["PackingType"] =
  //  {Interleaved}; auto *unfusedCTiledParallel = new
  //  SpMMSpMMUnFusedCTiledParallel(inSpMM, stats, sp);
  //  unfusedCTiledParallel->run();
  //  auto unfusedCTiledParallelStat = unfusedCTiledParallel->printStats();
  //  delete unfusedCTiledParallel;
  //  delete stats;

//    stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel","SpMM",
//    7,tp._matrix_name,numThread);
//    stats->OtherStats["PackingType"] ={Interleaved};
//    auto *fusedParallel = new SpMMSpMMFusedInterLayer(inSpMM,
//    stats, sp); fusedParallel->run();
//    //fusedParallel->OutTensor->printDx();
//    auto fusedParallelStat = fusedParallel->printStats();
//    delete fusedParallel;
//    delete stats;

    stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel_VariableTileSize","SpMM",
                                            7,tp._matrix_name,numThread);
    stats->OtherStats["PackingType"] ={Separated};
    stats->OtherStats["TilingMethod"] = {Variable};
    auto *fusedParallelVT = new SpMMSpMMFusedVariableTileSize(inSpMM,stats, sp);
    fusedParallelVT->run();
    //fusedParallel->OutTensor->printDx();
    auto fusedParallelVTStat = fusedParallelVT->printStats();
    delete fusedParallelVT;
    delete stats;
  ////
//    stats = new
//    swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel_BFS","SpMM",
//    7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
//    {Interleaved}; auto spBfs = sp; spBfs.SeedPartitioningParallelism = BFS;
//    auto *fusedParallelBfs = new SpMMSpMMFusedInterLayer(inSpMM, stats,
//    spBfs); fusedParallelBfs->run();
//    //fusedParallel->OutTensor->printDx();
//    auto fusedParallelStatBfs = fusedParallelBfs->printStats();
//    delete fusedParallelBfs;
//    delete stats;

  //  stats = new
  //  swiftware::benchmark::Stats("SpMM_SpMM_FusedTiledParallel","SpMM",
  //  7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
  //  {Interleaved}; auto *fusedTiledParallel = new SpMMSpMMFusedTiled(inSpMM,
  //  stats, sp); fusedTiledParallel->run();
  //  //fusedTiledParallel->OutTensor->printDx();
  //  auto fusedTiledParallelStat = fusedTiledParallel->printStats();
  //  delete fusedTiledParallel;
  //  delete stats;

//    stats = new
//    swiftware::benchmark::Stats("SpMM_SpMM_FusedTiledParallel_Redundant","SpMM",
//    7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
//    {Separated}; auto *fusedTiledParallel = new SpMMSpMMFusedTiledTri(inSpMM,
//    stats, sp); fusedTiledParallel->run();
//    //fusedTiledParallel->OutTensor->printDx();
//    auto fusedTiledParallelStat = fusedTiledParallel->printStats();
//    delete fusedTiledParallel;
//    delete stats;

//  stats =
//      new swiftware::benchmark::Stats("SpMM_SpMM_FusedTiledParallel_Redundant",
//                                      "SpMM", 7, tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  auto *fusedTiledParallelGen =
//      new SpMMSpMMFusedInterLayerRedundant(inSpMM, stats, sp);
//  fusedTiledParallelGen->run();
//  //  fusedTiledParallelGen->OutTensor->printDx();
//  auto fusedTiledParallelGenStat = fusedTiledParallelGen->printStats();
//  auto profileInfoRed = fusedTiledParallelGen->getSpInfo().printCSV(true);
//  std::string profHeaderRed = std::get<0>(profileInfoRed);
//  std::string profStatRed = std::get<1>(profileInfoRed);
//  delete fusedTiledParallelGen;
//  delete stats;

  //  stats = new
  //  swiftware::benchmark::Stats("SpMM_SpMM_FusedTiledParallel_Mixed_General","SpMM",
  //  7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
  //  {Separated}; auto *fusedTiledParallelGenMixed = new
  //  SpMMSpMMFusedInterLayerMixed(inSpMM, stats, sp);
  //  fusedTiledParallelGenMixed->run();
  //  //fusedTiledParallelGen->OutTensor->printDx();
  //  auto fusedTiledParallelMixedStat =
  //  fusedTiledParallelGenMixed->printStats(); auto profileInfoMixed =
  //  fusedTiledParallelGenMixed->getSpInfo().printCSV(true); std::string
  //  profHeaderMixed = std::get<0>(profileInfoMixed); std::string profStatMixed
  //  = std::get<1>(profileInfoMixed); delete fusedTiledParallelGenMixed; delete
  //  stats;

  //  stats = new
  //  swiftware::benchmark::Stats("SpMM_SpMM_OuterProduct_FusedParallel","SpMM",
  //  7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
  //  {Interleaved}; auto *fusedOuterParallel = new
  //  SpMMSpMMFusedInnerProdInterLayer(inSpMM, stats, sp);
  //  fusedOuterParallel->run();
  //  //fusedParallel->OutTensor->printDx();
  //  auto fusedParallelOutStat = fusedOuterParallel->printStats();
  //  delete fusedOuterParallel;
  //  delete stats;

  //  stats = new
  //  swiftware::benchmark::Stats("SpMM_SpMM_Mixed_FusedParallel","SpMM",
  //  7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
  //  {Interleaved}; auto *fusedMixedParallel = new
  //  SpMMSpMMFusedInnerProdInterLayer(inSpMM, stats, sp);
  //  fusedMixedParallel->run();
  //  //fusedParallel->OutTensor->printDx();
  //  auto fusedParallelMixedStat = fusedMixedParallel->printStats();
  //  delete fusedMixedParallel;
  //  delete stats;

//    stats = new
//    swiftware::benchmark::Stats("SpMM_SpMM_Separated_FusedParallel","SpMM",
//    7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
//    {Separated}; auto *fusedSepParallel = new
//    SpMMSpMMFusedSepInterLayer(inSpMM, stats, sp); fusedSepParallel->run();
//    //fusedParallel->OutTensor->printDx();
//    auto fusedParallelSepStat = fusedSepParallel->printStats();
//    delete fusedSepParallel;
//    delete stats;

  //  stats = new
  //  swiftware::benchmark::Stats("SpMM_SpMM_CSC_Separated_FusedParallel","SpMM",
  //  7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
  //  {Separated}; auto *fusedCSCSepParallel = new
  //  SpMMCSRSpMMCSCFusedAtomic(inSpMM, stats, sp); fusedCSCSepParallel->run();
  //  //fusedParallel->OutTensor->printDx();
  //  auto fusedCSCParallelSepStat = fusedCSCSepParallel->printStats();
  //  delete fusedCSCSepParallel;
  //  delete stats;
  //
  //
  //  stats = new
  //  swiftware::benchmark::Stats("SpMM_SpMM_CSC_Interleaved_Atomic_FusedParallel","SpMM",
  //  7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
  //  {Separated}; auto *fusedCSCInterleavedParallel = new
  //  SpMMCSRSpMMCSCFusedAtomicInterleaved(inSpMM, stats, sp);
  //  fusedCSCInterleavedParallel->run();
  //  //fusedParallel->OutTensor->printDx();
  //  auto fusedCSCInterleavedParallelStat =
  //  fusedCSCInterleavedParallel->printStats(); delete
  //  fusedCSCInterleavedParallel; delete stats;

  /// Coloring test
  //  int tileSize = sp.TileM;
//  DsaturColoringForConflictGraph *dsaturColoring =
//      new DsaturColoringForConflictGraph();
  //  DsaturColoringForConflictGraphWithKTiling *dsaturColoringWithKTiling =
  //      new DsaturColoringForConflictGraphWithKTiling();
  //
  //
//  std::map<int, std::vector<int>> colorToTiles =
//      dsaturColoring->generateGraphColoringForConflictGraphOf(
//          aCSCFull, sp.IterPerPartition, true);
  //  for (auto ct: colorToTiles){
  //    std::cout << ct.first << std::endl;
  //  }

//  stats = new swiftware::benchmark::Stats(
//      "SpMM_SpMM_CSC_Interleaved_Coloring_FusedParallel", "SpMM", 7,
//      tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  auto *fusedCSCInterleavedColoringParallel = new SpMMCSRSpMMCSCFusedColoring(
//      inSpMM, stats, sp, colorToTiles);
//  fusedCSCInterleavedColoringParallel->run();
//  // fusedParallel->OutTensor->printDx();
//  auto fusedCSCInterleavedColoringParallelStat =
//      fusedCSCInterleavedColoringParallel->printStats();
//  delete fusedCSCInterleavedColoringParallel;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats(
//      "SpMM_SpMM_CSC_Interleaved_Coloring_FusedParallel_Vectorized", "SpMM", 7,
//      tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  auto *fusedCSCInterleavedColoringParallelVectorized = new SpMMCSRSpMMCSCFusedColoringVectorized(
//      inSpMM, stats, sp, colorToTiles);
//  fusedCSCInterleavedColoringParallelVectorized->run();
//  // fusedParallel->OutTensor->printDx();
//  auto fusedCSCInterleavedColoringParallelVectorizedStat =
//      fusedCSCInterleavedColoringParallelVectorized->printStats();
//  delete fusedCSCInterleavedColoringParallelVectorized;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats(
//      "SpMM_SpMM_CSC_Interleaved_Coloring_FusedParallel_Vectorized_Packed", "SpMM", 7,
//      tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  auto *fusedCSCInterleavedColoringParallelVectorizedPacked = new SpMMCSRSpMMCSCFusedColoringVectorizedPacked(
//      inSpMM, stats, sp, colorToTiles);
//  fusedCSCInterleavedColoringParallelVectorizedPacked->run();
//  // fusedParallel->OutTensor->printDx();
//  auto fusedCSCInterleavedColoringParallelVectorizedPackedStat =
//      fusedCSCInterleavedColoringParallelVectorizedPacked->printStats();
//  delete fusedCSCInterleavedColoringParallelVectorizedPacked;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats(
//      "SpMM_SpMM_CSC_Interleaved_Coloring_NTiled_FusedParallel", "SpMM", 7,
//      tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  auto *fusedCSCInterleavedColoringNTParallel = new SpMMCSRSpMMCSCFusedColoringNTiling(
//      inSpMM, stats, sp, colorToTiles);
//  fusedCSCInterleavedColoringNTParallel->run();
//  // fusedParallel->OutTensor->printDx();
//  auto fusedCSCInterleavedColoringNTParallelStat =
//      fusedCSCInterleavedColoringNTParallel->printStats();
//  delete fusedCSCInterleavedColoringNTParallel;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats(
//      "SpMM_SpMM_CSC_Interleaved_Coloring_IterationTiled_FusedParallel", "SpMM", 7,
//      tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  auto *fusedCSCInterleavedColoringITParallel = new SpMMCSRSpMMCSCFusedColoringRowTiling(
//      inSpMM, stats, sp, colorToTiles);
//  fusedCSCInterleavedColoringITParallel->run();
//  // fusedParallel->OutTensor->printDx();
//  auto fusedCSCInterleavedColoringITParallelStat =
//      fusedCSCInterleavedColoringITParallel->printStats();
//  delete fusedCSCInterleavedColoringITParallel;
//  delete stats;

  //  std::vector<std::string> scheduledKTilingStats;
  //  std::vector<std::string> replicatedKTilingStats;
  //  std::vector<std::string> fusedKTiledStats;
  //  for(int i = 2; pow(2,i) < inSpMM->N; i++){
  //    int kTileSize = pow(2,i);
  //    std::map<int, std::vector<int>> colorToTilesForKTiling =
  //        dsaturColoringWithKTiling->generateGraphColoringForConflictGraphOf(aCSCFull,
  //        tileSize, inSpMM->N, kTileSize, true);
  //    stats = new
  //    swiftware::benchmark::Stats("SpMM_SpMM_CSC_Interleaved_Coloring_FusedParallel_ScheduledKTiling","SpMM",
  //    7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
  //    {Separated}; auto *fusedCSCInterleavedColoringParallelScheduledKTiling =
  //    new SpMMCSRSpMMCSCFusedColoringWithScheduledKTiling(inSpMM, stats, sp,
  //    tileSize,
  //                                                                                                  colorToTilesForKTiling, kTileSize);
  //    fusedCSCInterleavedColoringParallelScheduledKTiling->run();
  //    stats->OtherStats["NTile"] = {(double)kTileSize};
  //    //fusedParallel->OutTensor->printDx();
  //    scheduledKTilingStats.push_back(
  //        fusedCSCInterleavedColoringParallelScheduledKTiling->printStats());
  //    delete fusedCSCInterleavedColoringParallelScheduledKTiling;
  //    delete stats;
  //
  //    stats = new
  //    swiftware::benchmark::Stats("SpMM_SpMM_CSC_Interleaved_Coloring_FusedParallel_ReplicatedKTiling","SpMM",
  //    7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
  //    {Separated}; auto *fusedCSCInterleavedColoringParallelKTiling = new
  //    SpMMCSRSpMMCSCFusedColoringWithReplicatedKTiling(inSpMM, stats, sp,
  //    tileSize,
  //                                                                                                           colorToTiles, kTileSize);
  //    fusedCSCInterleavedColoringParallelKTiling->run();
  //    stats->OtherStats["NTile"] = {(double)kTileSize};
  //    //fusedParallel->OutTensor->printDx();
  //    replicatedKTilingStats.push_back(fusedCSCInterleavedColoringParallelKTiling->printStats());
  //    delete fusedCSCInterleavedColoringParallelKTiling;
  //    delete stats;
  //
  //    stats = new
  //    swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel_KTiled","SpMM",
  //    7,tp._matrix_name,numThread); stats->OtherStats["PackingType"] =
  //    {Interleaved}; auto *fusedParallelKTiled = new
  //    SpMMSpMMFusedInterLayerKTiled(inSpMM, stats, sp, kTileSize);
  //    fusedParallelKTiled->run();
  //    stats->OtherStats["NTile"] = {(double)kTileSize};
  //    //fusedParallel->OutTensor->printDx();
  //    fusedKTiledStats.push_back(fusedParallelKTiled->printStats());
  //    delete fusedParallelKTiled;
  //    delete stats;
  //
  //  }

//  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Profiler", "SpMM", 7,
//                                          tp._matrix_name, numThread);
//  auto *fusionProfiler = new SpMMSpMMFusionProfiler(inSpMM, stats, sp);
//  fusionProfiler->run();
//  // unfused->OutTensor->printDx();
//  inSpMM->IsSolProvided = true;
//  auto profileInfo = fusionProfiler->getSpInfo().printCSV(true);
  std::string profHeader = "";
  std::string profStat = "";
//  delete fusionProfiler;
//  delete stats;

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if (tp.print_header)
    std::cout << headerStat + spHeader + tpHeader + profHeader << std::endl;
  //  std::cout<<baselineStat<<spStat+tpStat+profStat<<std::endl;
  std::cout << unfusedParallelStat << spStat + tpStat + profStat << std::endl;
  //  std::cout<<unfusedOutParallelStat<<spStat+tpStat+profStat<<std::endl;
  //  std::cout<<unfusedCTiledParallelStat<<spStat+tpStat+profStat<<std::endl;
//  std::cout << fusedParallelStat <<spStat+tpStat+profStat<<std::endl;
  std::cout << fusedParallelVTStat << spStat + tpStat + profStat << std::endl;
//  std::cout<<fusedParallelStatBfs<<spStat+tpStat+profStat<<std::endl;
  // std::cout<<fusedTiledParallelStat<<spStat+tpStat+profStat<<std::endl;
  //  std::cout << fusedTiledParallelGenStat << spStat + tpStat + profStatRed
  //            << std::endl;
  // std::cout<<fusedTiledParallelMixedStat<<spStat+tpStat+profStatMixed<<std::endl;
  //  std::cout<<fusedParallelOutStat<<spStat+tpStat+profStat<<std::endl;
  //  std::cout<<fusedParallelMixedStat<<spStat+tpStat+profStat<<std::endl;
//  std::cout <<fusedParallelSepStat<<spStat+tpStat+profStat<<std::endl;
  //  std::cout<<fusedCSCParallelSepStat<<spStat+tpStat+profStat<<std::endl;
  //  std::cout<<fusedCSCInterleavedParallelStat<<spStat+tpStat+profStat<<std::endl;
//  std::cout<<fusedCSCInterleavedColoringParallelStat << spStat+tpStat+profStat<<std::endl;
//  std::cout<<fusedCSCInterleavedColoringParallelVectorizedStat << spStat+tpStat+profStat<<std::endl;
//  std::cout<<fusedCSCInterleavedColoringITParallelStat << spStat+tpStat+profStat<<std::endl;
//  std::cout<<fusedCSCInterleavedColoringNTParallelStat << spStat+tpStat+profStat<<std::endl;
//  std::cout<<fusedCSCInterleavedColoringParallelVectorizedPackedStat << spStat+tpStat+profStat<<std::endl;
  //  for (auto stat: scheduledKTilingStats){
  //    std::cout<<stat<<spStat+tpStat+profStat<<std::endl;
  //  }
  //  for (auto stat: replicatedKTilingStats){
  //    std::cout<<stat<<spStat+tpStat+profStat<<std::endl;
  //  }
  //  for (auto stat: fusedKTiledStats){
  //    std::cout<<stat<<spStat+tpStat+profStat<<std::endl;
  //  }
   #ifdef MKL

    stats = new swiftware::benchmark::Stats("SpMM_SpMM_MKL", "SpMM", 7,
    tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] ={Separated};
    stats->OtherStats["TilingMethod"] = {Fixed};
    auto *mklImpl = new SpMMSpMMMKL(inSpMM, stats);
    mklImpl->run();
    auto mklImplStat = mklImpl->printStats();
    delete mklImpl;
    delete stats;

    std::cout<<mklImplStat<<spStat+tpStat+profStat<<std::endl;
   #endif
   #ifdef __AVX2__
    stats = new
    swiftware::benchmark::Stats("SpMM_SpMM_FusedParallelAvx256","SpMM",
    7,tp._matrix_name,numThread);
    stats->OtherStats["PackingType"] ={Separated};
    stats->OtherStats["TilingMethod"] = {Variable};
    auto *fusedParallelVectorized256 = new
    SpMMSpMMFusedInterLayerVectorizedAvx256(inSpMM, stats, sp);
    fusedParallelVectorized256->run();
    //fusedParallel->OutTensor->printDx();
    auto fusedParallelVectorized256Stat =
    fusedParallelVectorized256->printStats();
    delete fusedParallelVectorized256;
    delete stats;
    std::cout<<fusedParallelVectorized256Stat<<spStat+tpStat+profStat<<std::endl;

    stats = new
        swiftware::benchmark::Stats("SpMM_SpMM_FusedParallelKTiled8Avx256","SpMM",
                                    7,tp._matrix_name,numThread);
    stats->OtherStats["PackingType"] = {Separated};
    stats->OtherStats["TilingMethod"] = {Variable};
    auto *fusedParallelVectorizedKTiled256 = new
        SpMMSpMMFusedInterLayerKTiled8VectorizedAvx256(inSpMM, stats, sp);
    fusedParallelVectorizedKTiled256->run();
    //fusedParallel->OutTensor->printDx();
    auto fusedParallelVectorizedKTiled256Stat =
        fusedParallelVectorizedKTiled256->printStats();
    delete fusedParallelVectorizedKTiled256;
    delete stats;
    std::cout<<fusedParallelVectorizedKTiled256Stat<<spStat+tpStat+profStat<<std::endl;

   #endif

   #ifdef __AVX512F__
     stats = new
     swiftware::benchmark::Stats("SpMM_SpMM_FusedParallelAvx512","SpMM",
     7,tp._matrix_name,numThread);
     stats->OtherStats["PackingType"] ={Separated};
     stats->OtherStats["TilingMethod"] = {Variable};
     auto *fusedParallelVectorized512 = new SpMMSpMMFusedInterLayerVectorizedAvx512(inSpMM, stats, sp);
     fusedParallelVectorized512->run();
     //fusedParallel->OutTensor->printDx();
     auto fusedParallelVectorized512Stat =
     fusedParallelVectorized512->printStats();
     delete fusedParallelVectorized512;
     delete stats;
     std::cout<<fusedParallelVectorized512Stat<<spStat+tpStat+profStat<<std::endl;

     stats = new
         swiftware::benchmark::Stats("SpMM_SpMM_FusedParallelKTiled8Avx512","SpMM",
                                     7,tp._matrix_name,numThread);
     stats->OtherStats["PackingType"] = {Separated};
     stats->OtherStats["TilingMethod"] = {Variable};
     auto *fusedParallelKTiledVectorized512 = new
         SpMMSpMMFusedInterLayerKTiled8VectorizedAvx512(inSpMM, stats, sp);
     fusedParallelKTiledVectorized512->run();
     auto fusedParallelKTVectorized512Stat =
         fusedParallelKTiledVectorized512->printStats();
     delete fusedParallelKTiledVectorized512;
     delete stats;
     std::cout<<fusedParallelKTVectorized512Stat<<spStat+tpStat+profStat<<std::endl;

//     stats = new swiftware::benchmark::Stats("SpMM_SpMM_CSC_Interleaved_Coloring_FusedParallel_Avx512","SpMM",7,tp._matrix_name,numThread);
//      stats->OtherStats["PackingType"] = {Separated};
//      auto *fusedCSCInterleavedColoringParallelVectorized512 = new SpMMCSRSpMMCSCFusedColoringAvx512(inSpMM, stats, sp, colorToTiles);
//      fusedCSCInterleavedColoringParallelVectorized512->run();
//      auto fusedCSCInterleavedColoringParallelVectorized512Stat = fusedCSCInterleavedColoringParallelVectorized512->printStats();
//      delete fusedCSCInterleavedColoringParallelVectorized512;
//      delete stats;
//      std::cout<<fusedCSCInterleavedColoringParallelVectorized512Stat<<spStat+tpStat+profStat<<std::endl;

   #endif
  //
  //
  //
  //
  //
  //
  //   auto tpCsv = tp.print_csv(tp.print_header);
  //   auto spCsv = sp.print_csv(tp.print_header);
  //   if(tp.print_header){
  //     std::cout<<std::get<0>(tpCsv)<<std::get<0>(spCsv)<<"\n";
  //   }
  //   std::cout<<std::get<1>(tpCsv)<<std::get<1>(spCsv);

  //  delete aCSC;
  //  delete aCSCFull;
  //  delete bCSC;
  //  delete alCSC;
  delete inSpMM;
//  delete dsaturColoring;
  //  delete dsaturColoringWithKTiling;

  return 0;
}

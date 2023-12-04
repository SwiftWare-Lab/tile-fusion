//
// Created by kazem on 02/05/23.
//

#include "aggregation/def.h"
#include "SpMM_SpMM_Demo_Utils.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/SparseFusion.h"
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


  //print_csc(1,"",aCSC);
  int numThread = sp._num_threads, numTrial = 7; std::string expName = "SpMM_SpMM_Demo";
  auto *inSpMM = new TensorInputs<double>(aCSCFull->m,  tp._b_cols, aCSCFull->n,
                                         bCSC->m, aCSCFull, bCSC,
                                          numThread, numTrial, expName);

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo", "SpMM", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *unfused = new SpMMSpMMUnFused(inSpMM, stats);
  unfused->run();
  //unfused->OutTensor->printDx();
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


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo_InnerProduct_UnFusedParallel", "SpMM", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *unfusedOutParallel = new SpMMSpMMUnFusedInnerParallel(inSpMM, stats);
  unfusedOutParallel->run();
  //unfusedParallel->OutTensor->printDx();
  auto unfusedOutParallelStat = unfusedOutParallel->printStats();
  delete unfusedOutParallel;
  delete stats;

  //sp.TileM = std::min(sp.IterPerPartition, inSpMM->M);

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo_CTiled_UnFusedParallel", "SpMM", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *unfusedCTiledParallel = new SpMMSpMMUnFusedCTiledParallel(inSpMM, stats, sp);
  unfusedCTiledParallel->run();
  auto unfusedCTiledParallelStat = unfusedCTiledParallel->printStats();
  delete unfusedCTiledParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *fusedParallel = new SpMMSpMMFusedInterLayer(inSpMM, stats, sp);
  fusedParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelStat = fusedParallel->printStats();
  delete fusedParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel_BFS","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto spBfs = sp; spBfs.SeedPartitioningParallelism = BFS;
  auto *fusedParallelBfs = new SpMMSpMMFusedInterLayer(inSpMM, stats, spBfs);
  fusedParallelBfs->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelStatBfs = fusedParallelBfs->printStats();
  delete fusedParallelBfs;
  delete stats;

//  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedTiledParallel","SpMM", 7,tp._matrix_name,numThread);
//  stats->OtherStats["PackingType"] = {Interleaved};
//  auto *fusedTiledParallel = new SpMMSpMMFusedTiled(inSpMM, stats, sp);
//  fusedTiledParallel->run();
//  //fusedTiledParallel->OutTensor->printDx();
//  auto fusedTiledParallelStat = fusedTiledParallel->printStats();
//  delete fusedTiledParallel;
//  delete stats;

//  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedTiledParallel_Redundant","SpMM", 7,tp._matrix_name,numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  auto *fusedTiledParallel = new SpMMSpMMFusedTiledTri(inSpMM, stats, sp);
//  fusedTiledParallel->run();
//  //fusedTiledParallel->OutTensor->printDx();
//  auto fusedTiledParallelStat = fusedTiledParallel->printStats();
//  delete fusedTiledParallel;
//  delete stats;

    stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedTiledParallel_Redundant_General","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Separated};
  auto *fusedTiledParallelGen = new SpMMSpMMFusedInterLayerRedundant(inSpMM, stats, sp);
  fusedTiledParallelGen->run();
  //fusedTiledParallelGen->OutTensor->printDx();
  auto fusedTiledParallelGenStat = fusedTiledParallelGen->printStats();
  auto profileInfoRed = fusedTiledParallelGen->getSpInfo().printCSV(true);
  std::string profHeaderRed = std::get<0>(profileInfoRed);
  std::string profStatRed = std::get<1>(profileInfoRed);
  delete fusedTiledParallelGen;
  delete stats;


//  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedTiledParallel_Mixed_General","SpMM", 7,tp._matrix_name,numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  auto *fusedTiledParallelGenMixed = new SpMMSpMMFusedInterLayerMixed(inSpMM, stats, sp);
//  fusedTiledParallelGenMixed->run();
//  //fusedTiledParallelGen->OutTensor->printDx();
//  auto fusedTiledParallelMixedStat = fusedTiledParallelGenMixed->printStats();
//  auto profileInfoMixed = fusedTiledParallelGenMixed->getSpInfo().printCSV(true);
//  std::string profHeaderMixed = std::get<0>(profileInfoMixed);
//  std::string profStatMixed = std::get<1>(profileInfoMixed);
//  delete fusedTiledParallelGenMixed;
//  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_OuterProduct_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *fusedOuterParallel = new SpMMSpMMFusedInnerProdInterLayer(inSpMM, stats, sp);
  fusedOuterParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelOutStat = fusedOuterParallel->printStats();
  delete fusedOuterParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Mixed_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *fusedMixedParallel = new SpMMSpMMFusedInnerProdInterLayer(inSpMM, stats, sp);
  fusedMixedParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelMixedStat = fusedMixedParallel->printStats();
  delete fusedMixedParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Separated_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Separated};
  auto *fusedSepParallel = new SpMMSpMMFusedSepInterLayer(inSpMM, stats, sp);
  fusedSepParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelSepStat = fusedSepParallel->printStats();
  delete fusedSepParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_CSC_Separated_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Separated};
  auto *fusedCSCSepParallel = new SpMMCSRSpMMCSCFusedAtomic(inSpMM, stats, sp);
  fusedCSCSepParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedCSCParallelSepStat = fusedCSCSepParallel->printStats();
  delete fusedCSCSepParallel;
  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_CSC_Interleaved_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Separated};
  auto *fusedCSCInterleavedParallel = new SpMMCSRSpMMCSCFusedAtomicInterleaved(inSpMM, stats, sp);
  fusedCSCInterleavedParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedCSCInterleavedParallelStat = fusedCSCInterleavedParallel->printStats();
  delete fusedCSCInterleavedParallel;
  delete stats;

  /// Coloring test
  int tileSize = sp.TileM;
  DsaturColoringForConflictGraph *dsaturColoring =
      new DsaturColoringForConflictGraph();
  DsaturColoringForConflictGraphWithKTiling *dsaturColoringWithKTiling =
      new DsaturColoringForConflictGraphWithKTiling();
  std::map<int, std::vector<int>> colorToTiles =
      dsaturColoring->generateGraphColoringForConflictGraphOf(aCSCFull,
                                                              tileSize);

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_CSC_Interleaved_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Separated};
  auto *fusedCSCInterleavedColoringParallel = new SpMMCSRSpMMCSCFusedColoring(inSpMM, stats, sp, tileSize,
                                                                               colorToTiles);
  fusedCSCInterleavedColoringParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedCSCInterleavedColoringParallelStat = fusedCSCInterleavedColoringParallel->printStats();
  delete fusedCSCInterleavedColoringParallel;
  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Profiler","SpMM", 7,tp._matrix_name,numThread);
  auto *fusionProfiler = new SpMMSpMMFusionProfiler(inSpMM, stats, sp);
  fusionProfiler->run();
  //unfused->OutTensor->printDx();
  inSpMM->IsSolProvided = true;
  auto profileInfo = fusionProfiler->getSpInfo().printCSV(true);
  std::string profHeader = std::get<0>(profileInfo);
  std::string profStat = std::get<1>(profileInfo);
  //delete fusionProfiler;
  delete stats;



  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if(tp.print_header)
    std::cout<<headerStat+spHeader+tpHeader+profHeader<<std::endl;
  std::cout<<baselineStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<unfusedParallelStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<unfusedOutParallelStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<unfusedCTiledParallelStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<fusedParallelStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<fusedParallelStatBfs<<spStat+tpStat+profStat<<std::endl;
  //std::cout<<fusedTiledParallelStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<fusedTiledParallelGenStat<<spStat+tpStat+profStatRed<<std::endl;
  //std::cout<<fusedTiledParallelMixedStat<<spStat+tpStat+profStatMixed<<std::endl;
  std::cout<<fusedParallelOutStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<fusedParallelMixedStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<fusedParallelSepStat<<spStat+tpStat+profStat;
  std::cout<<fusedCSCParallelSepStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<fusedCSCInterleavedParallelStat<<spStat+tpStat+profStat<<std::endl;

//  sp._num_w_partition = 2;
//  //print_csc(1,"",A_csc);
//  auto *sf01 = new SparseFusion(&sp, 2);
//  auto *mvDAG =  diagonal(alCSC->n, 1.0);
//  sf01->fuse(0, mvDAG, NULLPNTR);
//  //sf01->print_final_list();
//  sf01->fuse(1, mvDAG, alCSC);
//  sf01->print_final_list();
//
//
//  auto *sf02 = new SparseFusion(&sp, 2);
//  sf02->fuse(0, mvDAG, NULLPNTR);
//  //sf01->print_final_list();
//  sf02->fuse(1, mvDAG, alCSC);
//  auto *fusedCompSet = sf02->getFusedCompressed();
//  fusedCompSet->print_3d();
//
//
//
//
//  auto tpCsv = tp.print_csv(tp.print_header);
//  auto spCsv = sp.print_csv(tp.print_header);
//  if(tp.print_header){
//    std::cout<<std::get<0>(tpCsv)<<std::get<0>(spCsv)<<"\n";
//  }
//  std::cout<<std::get<1>(tpCsv)<<std::get<1>(spCsv);

  delete aCSC;
  delete aCSCFull;
  delete bCSC;
  delete alCSC;
  delete inSpMM;

  return 0;
}

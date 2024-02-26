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
#include <set>

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


  stats = new swiftware::benchmark::Stats(
      "SpMM_SpMM_Demo_UnFusedParallel", "SpMM", numTrial, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  stats->OtherStats["TilingMethod"] = {Single};
  auto *unfusedParallel = new SpMMSpMMUnFusedParallel(inSpMM, stats);
  unfusedParallel->run();
  //  unfusedParallel->OutTensor->printDx();
//  std::copy(unfusedParallel->OutTensor->Xx,
//            unfusedParallel->OutTensor->Xx +
//                unfusedParallel->OutTensor->M * unfusedParallel->OutTensor->N,
//            inSpMM->CorrectSol);
//  inSpMM->IsSolProvided = true;
  auto headerStat = unfusedParallel->printStatsHeader();
  auto unfusedParallelStat = unfusedParallel->printStats();
  delete unfusedParallel;
  delete stats;

  

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if (tp.print_header)
    std::cout << headerStat + spHeader + tpHeader << std::endl;
  std::cout<<unfusedParallelStat << spStat + tpStat <<std::endl;
  //5 samples
  for (int i = 1; i < 6; i++){

    std::vector<int> mTileParameters = {16,32,64,128,256,512,1024,2048};
    std::string newMatName = tp._matrix_name+ '_' +std::to_string(i);

    for (auto param: mTileParameters){
      ScheduleParameters spTemp(sp);
      spTemp.IterPerPartition = param;
      spTemp.TileM = param;
      auto csvTempInfo = spTemp.print_csv(true);
      std::string spTempStat = std::get<1>(csvTempInfo);
      stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel_FixedTile","SpMM",
                                              numTrial,newMatName,numThread);
      stats->OtherStats["PackingType"] ={Separated};
      stats->OtherStats["TilingMethod"] = {Fixed};
      auto *fusedParallel = new SpMMSpMMFusedVariableTileSize(inSpMM,
                                                              stats, spTemp, i);
      fusedParallel->run();
      //fusedParallel->OutTensor->printDx();
      auto fusedParallelStats = fusedParallel->printStats();
      std::cout << fusedParallelStats << spTempStat + tpStat<< std::endl;

      delete fusedParallel;
      delete stats;
#ifdef __AVX2__
      stats = new
          swiftware::benchmark::Stats("SpMM_SpMM_FusedParallelAvx256_FixedTile","SpMM",
                                      numTrial,newMatName,numThread);
      stats->OtherStats["PackingType"] ={Separated};
      stats->OtherStats["TilingMethod"] = {Fixed};
      auto *fusedParallelVectorized256 = new
          SpMMSpMMFusedInterLayerVectorizedAvx256(inSpMM, stats, spTemp, i);
      fusedParallelVectorized256->run();
      //fusedParallel->OutTensor->printDx();
      auto fusedParallelVectorized256Stats = fusedParallelVectorized256->printStats();
      std::cout<<fusedParallelVectorized256Stats<<spTempStat+tpStat<<std::endl;
      delete fusedParallelVectorized256;
      delete stats;
      stats = new
          swiftware::benchmark::Stats("SpMM_SpMM_UnFusedParallelAvx256_FixedTile","SpMM",
                                      numTrial,newMatName,numThread);
      stats->OtherStats["PackingType"] ={Separated};
      stats->OtherStats["TilingMethod"] = {Fixed};
      auto *unfusedParallelVectorized256 = new SpMMSpMMUnFusedParallelVectorized16(inSpMM, stats, spTemp, i);
      unfusedParallelVectorized256->run();
      //fusedParallel->OutTensor->printDx();
      auto unfusedParallelVectorized256Stats = unfusedParallelVectorized256->printStats();
      std::cout<<unfusedParallelVectorized256Stats<<spTempStat+tpStat<<std::endl;
      delete unfusedParallelVectorized256;
      delete stats;
#endif

#ifdef __AVX512F__

#endif
    }

    std::vector<int> wsParameters = {10000,15000,20000,32000,50000,100000,500000,1000000};
    for (auto param: wsParameters){
      ScheduleParameters spTemp(sp);
      spTemp.IterPerPartition = param;
      spTemp.TileM = param;
      auto csvTempInfo = spTemp.print_csv(true);
      std::string spTempStat = std::get<1>(csvTempInfo);
      stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel_VariableTileSize","SpMM",
                                              numTrial,newMatName,numThread);
      stats->OtherStats["PackingType"] ={Separated};
      stats->OtherStats["TilingMethod"] = {Variable};
      auto *fusedParallelVT = new SpMMSpMMFusedVariableTileSize(inSpMM,stats, spTemp, i);
      fusedParallelVT->run();
      //fusedParallel->OutTensor->printDx();
      auto fusedParallelVTStat = fusedParallelVT->printStats();
      std::cout << fusedParallelVTStat << spTempStat + tpStat<< std::endl;
      delete fusedParallelVT;
      delete stats;
#ifdef __AVX2__
      stats = new
          swiftware::benchmark::Stats("SpMM_SpMM_FusedParallelAvx256_VariableTile","SpMM",
                                      numTrial,newMatName,numThread);
      stats->OtherStats["PackingType"] = {Separated};
      stats->OtherStats["TilingMethod"] = {Variable};
      auto *fusedParallelVectorized256 = new
          SpMMSpMMFusedInterLayerVectorizedAvx256(inSpMM, stats, spTemp, i);
      fusedParallelVectorized256->run();
      //fusedParallel->OutTensor->printDx();
      auto fusedParallelVectorized256Stat = fusedParallelVectorized256->printStats();
      delete fusedParallelVectorized256;
      delete stats;
      std::cout<<fusedParallelVectorized256Stat<<spTempStat+tpStat<<std::endl;

      stats = new
          swiftware::benchmark::Stats("SpMM_SpMM_FusedParallelKTiledAvx256_VariableTile","SpMM",
                                      numTrial,newMatName,numThread);
      stats->OtherStats["PackingType"] = {Separated};
      stats->OtherStats["TilingMethod"] = {Variable};
      auto *fusedParallelVectorizedKTiled256 = new
          SpMMSpMMFusedInterLayerKTiled8VectorizedAvx256(inSpMM, stats, spTemp, i);
      fusedParallelVectorizedKTiled256->run();
      //fusedParallel->OutTensor->printDx();
      auto fusedParallelVectorizedKTiled256Stat =
          fusedParallelVectorizedKTiled256->printStats();
      delete fusedParallelVectorizedKTiled256;
      delete stats;
      std::cout<<fusedParallelVectorizedKTiled256Stat<<spTempStat+tpStat<<std::endl;

#endif
    }
  }

  delete inSpMM;
  //  delete dsaturColoring;
  //  delete dsaturColoringWithKTiling;

  return 0;
}

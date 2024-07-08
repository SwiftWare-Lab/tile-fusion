//
// Created by salehm32 on 05/03/24.
//

#include "SpMM_SpMM_SP_Demo_Utils.h"
#include "aggregation/def.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/GraphColoring.h"
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
      new TensorInputs<float>(aCSCFull->m, tp._b_cols, aCSCFull->n, bCSC->m,
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
      "SpMM_SpMM_Demo_UnFusedParallel", "SpMM", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  stats->OtherStats["TilingMethod"] = {Fixed};
  auto *unfusedParallel = new SpMMSpMMUnFusedParallelSP(inSpMM, stats);
  unfusedParallel->run();
  //    unfusedParallel->OutTensor->printDx();
  std::copy(unfusedParallel->OutTensor->Xx,
            unfusedParallel->OutTensor->Xx +
                unfusedParallel->OutTensor->M * unfusedParallel->OutTensor->N,
            inSpMM->CorrectSol);
  inSpMM->IsSolProvided = true;
  auto headerStat = unfusedParallel->printStatsHeader();
  auto unfusedParallelStat = unfusedParallel->printStats();
  delete unfusedParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel","SpMM",
                                          7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] ={Separated};
  stats->OtherStats["TilingMethod"] = {Fixed};
  auto *fusedParallelVT = new SpMMSpMMFusedVariableTileSizeSP(inSpMM,stats, sp);
  fusedParallelVT->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelVTStat = fusedParallelVT->printStats();
  delete fusedParallelVT;
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
  std::cout << unfusedParallelStat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedParallelVTStat << spStat + tpStat + profStat << std::endl;

#ifdef MKL

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_MKL", "SpMM", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] ={Separated};
  stats->OtherStats["TilingMethod"] = {Fixed};
  auto *mklImpl = new SpMMSpMMMKL_SP(inSpMM, stats);
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
  stats->OtherStats["TilingMethod"] = {Fixed};
  auto *fusedParallelVectorized256 = new
      SpMMSpMMFusedInterLayerVectorizedAvx256SP(inSpMM, stats, sp);
  fusedParallelVectorized256->run();
  //    fusedParallelVectorized256->OutTensor->printDx();
  auto fusedParallelVectorized256Stat =
      fusedParallelVectorized256->printStats();
  delete fusedParallelVectorized256;
  delete stats;
  std::cout<<fusedParallelVectorized256Stat<<spStat+tpStat+profStat<<std::endl;

  stats = new
      swiftware::benchmark::Stats("SpMM_SpMM_UnFusedParallelAvx256","SpMM",
                                  7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] ={Separated};
  stats->OtherStats["TilingMethod"] = {Fixed};
  auto *unfusedParallelVectorized256 = new
      SpMMSpMMUnFusedParallelVectorizedAVX2SP(inSpMM, stats, sp);
  unfusedParallelVectorized256->run();
  //    unfusedParallelVectorized256->OutTensor->printDx();
  auto unfusedParallelVectorized256Stat =
      unfusedParallelVectorized256->printStats();
  delete unfusedParallelVectorized256;
  delete stats;
  std::cout<<unfusedParallelVectorized256Stat<<spStat+tpStat+profStat<<std::endl;


  stats = new
      swiftware::benchmark::Stats("SpMM_SpMM_SMatReuse_FusedParallelAvx256","SpMM",
                                  7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] ={Separated};
  stats->OtherStats["TilingMethod"] = {Fixed};
  auto *fusedParallelVectorized256SMatReuse = new
      SpMMSpMMFusedOneSparseMatInterLayerVectorizedAvx256SP(inSpMM, stats, sp);
  fusedParallelVectorized256SMatReuse->run();
  //    fusedParallelVectorized256SMatReuse->OutTensor->printDx();
  auto fusedParallelVectorized256SMatReuseStat =
      fusedParallelVectorized256SMatReuse->printStats();
  delete fusedParallelVectorized256SMatReuse;
  delete stats;
  std::cout<<fusedParallelVectorized256SMatReuseStat<<spStat+tpStat+profStat<<std::endl;

  stats = new
      swiftware::benchmark::Stats("SpMM_SpMM_SMatReuseVT_FusedParallelAvx256","SpMM",
                                  7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] ={Separated};
  stats->OtherStats["TilingMethod"] = {Variable};
  auto *fusedParallelVectorized256SMatReuseVT = new
      SpMMSpMMFusedInterLayerVectorizedAvx256SP(inSpMM, stats, sp);
  fusedParallelVectorized256SMatReuseVT->run();
  //    fusedParallelVectorized256SMatReuse->OutTensor->printDx();
  auto fusedParallelVectorized256SMatReuseVTStat =
      fusedParallelVectorized256SMatReuseVT->printStats();
  delete fusedParallelVectorized256SMatReuseVT;
  delete stats;
  std::cout<<fusedParallelVectorized256SMatReuseVTStat<<spStat+tpStat+profStat<<std::endl;


//  stats = new
//      swiftware::benchmark::Stats("SpMM_SpMM_SMatReuse_P2P_FusedParallelAvx256","SpMM",
//                                  7,tp._matrix_name,numThread);
//  stats->OtherStats["PackingType"] ={Separated};
//  stats->OtherStats["TilingMethod"] = {Fixed};
//  auto *fusedParallelVectorized256P2PSMatReuse = new
//      SpMMSpMMFusedInterLayerVectorizedAvx256P2PThreadingSP(inSpMM, stats, sp);
//  fusedParallelVectorized256P2PSMatReuse->run();
//  //    fusedParallelVectorized256P2PSMatReuse->OutTensor->printDx();
//  auto fusedParallelVectorized256P2PSMatReuseStat =
//      fusedParallelVectorized256P2PSMatReuse->printStats();
//  delete fusedParallelVectorized256P2PSMatReuse;
//  delete stats;
//  std::cout<<fusedParallelVectorized256P2PSMatReuseStat<<spStat+tpStat+profStat<<std::endl;

  stats = new
      swiftware::benchmark::Stats("SpMM_SpMM_SMatReuse_RO_FusedParallelAvx256","SpMM",
                                  7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] ={Separated};
  stats->OtherStats["TilingMethod"] = {Fixed};
  auto *fusedParallelVectorized256ROSMatReuse = new
      SpMMSpMMFusedReorderedUnFusedMatInterLayerVectorizedAvx256SP(inSpMM, stats, sp);
  fusedParallelVectorized256ROSMatReuse->run();
  //    fusedParallelVectorized256ROSMatReuse->OutTensor->printDx();
  auto fusedParallelVectorized256ROSMatReuseStat =
      fusedParallelVectorized256ROSMatReuse->printStats();
  delete fusedParallelVectorized256ROSMatReuse;
  delete stats;
  std::cout<<fusedParallelVectorized256ROSMatReuseStat<<spStat+tpStat+profStat<<std::endl;
#endif

#ifdef __AVX512F__
//  stats = new
//      swiftware::benchmark::Stats("SpMM_SpMM_FusedParallelAvx512","SpMM",
//                                  7,tp._matrix_name,numThread);
//  stats->OtherStats["PackingType"] ={Separated};
//  stats->OtherStats["TilingMethod"] = {Fixed};
//  auto *fusedParallelVectorized512 = new SpMMSpMMFusedInterLayerVectorizedAvx512SP(inSpMM, stats, sp);
//  fusedParallelVectorized512->run();
//  //fusedParallel->OutTensor->printDx();
//  auto fusedParallelVectorized512Stat =
//      fusedParallelVectorized512->printStats();
//  delete fusedParallelVectorized512;
//  delete stats;
//  std::cout<<fusedParallelVectorized512Stat<<spStat+tpStat+profStat<<std::endl;
//
//  stats = new
//      swiftware::benchmark::Stats("SpMM_SpMM_UnFusedParallelAvx512","SpMM",
//                                  7,tp._matrix_name,numThread);
//  stats->OtherStats["PackingType"] ={Separated};
//  stats->OtherStats["TilingMethod"] = {Fixed};
//  auto *unfusedParallelVectorized512 = new SpMMSpMMFusedInterLayerVectorizedAvx512SP(inSpMM, stats, sp);
//  unfusedParallelVectorized512->run();
//  //fusedParallel->OutTensor->printDx();
//  auto unfusedParallelVectorized512Stat =
//      unfusedParallelVectorized512->printStats();
//  delete unfusedParallelVectorized512;
//  delete stats;
//  std::cout<<unfusedParallelVectorized512Stat<<spStat+tpStat+profStat<<std::endl;

  //     stats = new
  //         swiftware::benchmark::Stats("SpMM_SpMM_FusedParallelKTiled8Avx512","SpMM",
  //                                     7,tp._matrix_name,numThread);
  //     stats->OtherStats["PackingType"] = {Separated};
  //     stats->OtherStats["TilingMethod"] = {Variable};
  //     auto *fusedParallelKTiledVectorized512 = new
  //         SpMMSpMMFusedInterLayerKTiled8VectorizedAvx512(inSpMM, stats, sp);
  //     fusedParallelKTiledVectorized512->run();
  //     auto fusedParallelKTVectorized512Stat =
  //         fusedParallelKTiledVectorized512->printStats();
  //     delete fusedParallelKTiledVectorized512;
  //     delete stats;
  //     std::cout<<fusedParallelKTVectorized512Stat<<spStat+tpStat+profStat<<std::endl;

  //     stats = new swiftware::benchmark::Stats("SpMM_SpMM_CSC_Interleaved_Coloring_FusedParallel_Avx512","SpMM",7,tp._matrix_name,numThread);
  //      stats->OtherStats["PackingType"] = {Separated};
  //      auto *fusedCSCInterleavedColoringParallelVectorized512 = new SpMMCSRSpMMCSCFusedColoringAvx512(inSpMM, stats, sp, colorToTiles);
  //      fusedCSCInterleavedColoringParallelVectorized512->run();
  //      auto fusedCSCInterleavedColoringParallelVectorized512Stat = fusedCSCInterleavedColoringParallelVectorized512->printStats();
  //      delete fusedCSCInterleavedColoringParallelVectorized512;
  //      delete stats;
  //      std::cout<<fusedCSCInterleavedColoringParallelVectorized512Stat<<spStat+tpStat+profStat<<std::endl;

#endif

  delete inSpMM;


  return 0;
}

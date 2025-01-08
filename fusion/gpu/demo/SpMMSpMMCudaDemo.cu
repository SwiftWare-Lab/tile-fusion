//
// Created by salehm32 on 20/06/24.
//
#include "../Cuda_SpMM_SpMM_Demo_Utils.h"

#include <iostream>

using namespace sym_lib;
int main (const int argc, const char *argv[]) {
  TestParameters tp;
  tp._order_method = SYM_ORDERING::NONE;
  ScheduleParameters sp;
  Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  CSC *aCSC = get_matrix_from_parameter(&tp);
  if (aCSC->m != aCSC->n) {
    return -1;
  }
  CSC *aCSCFull = nullptr;
  if (aCSC->stype == -1 || aCSC->stype == 1) {
    aCSCFull = make_full(aCSC);
  } else {
    aCSCFull = copy_sparse(aCSC);
  }
  tp._dim1 = aCSCFull->m;
  tp._dim2 = aCSCFull->n;
  tp._nnz = aCSCFull->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);

  int numThread = sp._num_threads, numTrial = 7;
  int ufThreadsPerBlock = 128;
  int fThreadsPerBlock=32;
  std::string expName = "SpMM_SpMM_Demo";
  auto *inSpMM =
      new CudaTensorInputs(aCSCFull->m, tp._b_cols, aCSCFull->m, aCSCFull->m,
                           aCSCFull, aCSCFull, numThread, numTrial, expName);



  stats = new swiftware::benchmark::Stats("CPU_Unfused_Seq", "SpMMSpMM", numTrial,
                                          tp._matrix_name, numThread);
  auto *cpuSpMMSpMM = new SeqSpMMSpMM(inSpMM, stats);
  cpuSpMMSpMM->run();
//  std::cout << "CPU: " << std::endl;
//  cpuSpMMSpMM->OutTensor->printDx();
  std::copy(cpuSpMMSpMM->OutTensor->Xx,
            cpuSpMMSpMM->OutTensor->Xx +
                cpuSpMMSpMM->OutTensor->M * cpuSpMMSpMM->OutTensor->N,
            inSpMM->CorrectSol);
  inSpMM->IsSolProvided = true;
  auto headerStat = cpuSpMMSpMM->printStatsHeader();
  auto cpuSpMMSpMMStat = cpuSpMMSpMM->printStats();
  delete cpuSpMMSpMM;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Unfused_CuSparse_ALG2","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedCuSparseAlg2 = new SpMMSpMMCuSparse(inSpMM,stats, CUSPARSE_SPMM_CSR_ALG2);
  unfusedCuSparseAlg2->run();
  //  std::cout << "UNFUSED: " << std::endl;
  //  unfusedCuSparseAlg2->OutTensor->printDx();
  auto unfusedCuSparseAlg2Stat = unfusedCuSparseAlg2->printStats();
  delete unfusedCuSparseAlg2;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Unfused_CuSparse_ALG3","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedCuSparseAlg3 = new SpMMSpMMCuSparse(inSpMM,stats, CUSPARSE_SPMM_CSR_ALG3);
  unfusedCuSparseAlg3->run();
  //  std::cout << "UNFUSED: " << std::endl;
  //  unfusedCuSparseAlg3->OutTensor->printDx();
  auto unfusedCuSparseAlg3Stat = unfusedCuSparseAlg3->printStats();
  delete unfusedCuSparseAlg3;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Unfused_SeqReduceRowBalance","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedSeqReduceRowBalance = new SpMMSpMMSeqReduceRowBalance(inSpMM,stats, ufThreadsPerBlock);
  unfusedSeqReduceRowBalance->run();
  //  std::cout << "UNFUSED: " << std::endl;
  //  unfusedSeqReduceRowBalance->OutTensor->printDx();
  auto unfusedSeqReduceRowBalanceStat = unfusedSeqReduceRowBalance->printStats();
  delete unfusedSeqReduceRowBalance;
  delete stats;

  //  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceRowBalance","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  //  auto *fusedSeqReduceRowBalance = new FusedSpMMSpMMSeqReduceRowBalance(inSpMM,stats, ThreadsPerBlock);
  //  fusedSeqReduceRowBalance->run();
  ////  std::cout << "FUSED: " << std::endl;
  ////  fusedSeqReduceRowBalance->OutTensor->printDx();
  //  auto fusedSeqReduceRowBalanceStat = fusedSeqReduceRowBalance->printStats();
  //  delete fusedSeqReduceRowBalance;
  //  delete stats;
  //
  //  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_SeqReduceRowBalance","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  //  auto *fusedSeqReduceRowBalanceReordered = new FusedSpMMSpMMSeqReduceRowBalanceReordered(inSpMM,stats, ThreadsPerBlock);
  //  fusedSeqReduceRowBalanceReordered->run();
  //  //  std::cout << "FUSED: " << std::endl;
  //  //  fusedSeqReduceRowBalanceReordered->OutTensor->printDx();
  //  auto fusedSeqReduceRowBalanceReorderedStat = fusedSeqReduceRowBalanceReordered->printStats();
  //  delete fusedSeqReduceRowBalanceReordered;
  //  delete stats;

  auto tileSizes = {8, 16, 32, 64};
  std::vector <std::string> statList;
  for (auto ts: tileSizes){
    std::string unfusedSeqReduceRowCoarsenedName = "GPU_Unfused_SeqReduceRowCoarsened_" + std::to_string(ts);
    stats = new swiftware::benchmark::Stats(unfusedSeqReduceRowCoarsenedName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
    auto *unfusedSeqReduceRowCoarsened = new SpMMSpMMSeqReduceRowBalanceCoarsenedRow(inSpMM,stats, ufThreadsPerBlock, ts);
    unfusedSeqReduceRowCoarsened->run();
    statList.push_back(unfusedSeqReduceRowCoarsened->printStats());
    delete stats;

//    std::string unfusedSeqReduceRowCoarsenedStrideName = "GPU_Unfused_SeqReduceRowCoarsenedStride_" + std::to_string(ts);
//    stats = new swiftware::benchmark::Stats(unfusedSeqReduceRowCoarsenedStrideName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
//    auto *unfusedSeqReduceRowCoarsenedStride = new SpMMSpMMSeqReduceRowBalanceCoarsenedRowStride(inSpMM,stats, ufThreadsPerBlock, ts);
//    unfusedSeqReduceRowCoarsenedStride->run();
//    statList.push_back(unfusedSeqReduceRowCoarsenedStride->printStats());
//    delete stats;

    std::string fusedHighFusionRatioName = "GPU_Fused_Reordered_HighFusionRatio_" + std::to_string(ts);
    stats = new swiftware::benchmark::Stats(fusedHighFusionRatioName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
    auto *fusedHighFusionRatio = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, ts);
    fusedHighFusionRatio->run();
    statList.push_back(fusedHighFusionRatio->printStats());
    delete fusedHighFusionRatio;
    delete stats;

//    std::string fusedHighFusionRatioStrideName = "GPU_Fused_Reordered_HighFusionRatioStride_" + std::to_string(ts);
//    stats = new swiftware::benchmark::Stats(fusedHighFusionRatioStrideName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
//    auto *fusedHighFusionRatioStride = new FusedSpMMSpMMHighFusionRatioStride(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, ts);
//    fusedHighFusionRatioStride->run();
//    statList.push_back(fusedHighFusionRatioStride->printStats());
//    delete fusedHighFusionRatioStride;
//    delete stats;

//    std::string fusedHighFusionRatioResultPackedName = "GPU_Fused_ReorderedResultPacked_HighFusionRatio_" + std::to_string(ts);
//    stats = new swiftware::benchmark::Stats(fusedHighFusionRatioResultPackedName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
//    auto *fusedHighFusionRatioResultPacked = new FusedSpMMSpMMHighFusionRatioResultPacked(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, ts);
//    fusedHighFusionRatioResultPacked->run();
//    statList.push_back(fusedHighFusionRatioResultPacked->printStats());
//    delete fusedHighFusionRatioResultPacked;
//    delete stats;

//    std::string fusedHighFusionRatioP4Name = "GPU_Fused_Reordered_HighFusionRatio_P4_" + std::to_string(ts);
//    stats = new swiftware::benchmark::Stats(fusedHighFusionRatioP4Name,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
//    auto *fusedHighFusionRatioP4 = new FusedSpMMSpMMHighFusionRatioProductOf4Fused(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, ts);
//    fusedHighFusionRatioP4->run();
//    statList.push_back(fusedHighFusionRatioP4->printStats());
//    delete fusedHighFusionRatioP4;
//    delete stats;

//    std::string fusedHighFusionRatioMBCName = "GPU_Fused_Reordered_HighFusionRatio_MBC_" + std::to_string(ts);
//    stats = new swiftware::benchmark::Stats(fusedHighFusionRatioMBCName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
//    auto *fusedHighFusionRatioMBC = new FusedSpMMSpMMHighFusionRatioMultipleBCols(inSpMM,stats, ufThreadsPerBlock, ts);
//    fusedHighFusionRatioMBC->run();
//    statList.push_back(fusedHighFusionRatioMBC->printStats());
//    delete fusedHighFusionRatioMBC;
//    delete stats;

    std::string fusedHighFusionRatio2LNName = "GPU_Fused_Reordered_HighFusionRatio_2LN_" + std::to_string(ts);
    stats = new swiftware::benchmark::Stats(fusedHighFusionRatio2LNName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
    auto *fusedHighFusionRatio2LN = new FusedSpMMSpMMHighFusionRatio2Level(inSpMM,stats, ufThreadsPerBlock, ts);
    fusedHighFusionRatio2LN->run();
    statList.push_back(fusedHighFusionRatio2LN->printStats());
    delete fusedHighFusionRatio2LN;
    delete stats;

    std::string fusedHighFusionRatioAtmoicFusedName = "GPU_AtomicFused_Reordered_HighFusionRatio_" + std::to_string(ts);
    stats = new swiftware::benchmark::Stats(fusedHighFusionRatioAtmoicFusedName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
    auto *fusedHighFusionRatioAtmoicFused = new FusedSpMMSpMMFusedParReduce(inSpMM,stats, ufThreadsPerBlock, ts);
    fusedHighFusionRatioAtmoicFused->run();
    statList.push_back(fusedHighFusionRatioAtmoicFused->printStats());
    delete fusedHighFusionRatioAtmoicFused;
    delete stats;

    std::string fusedHighFusionRatioNoAtmoicFusedName = "GPU_NoAtomicFused_Reordered_HighFusionRatio_" + std::to_string(ts);
    stats = new swiftware::benchmark::Stats(fusedHighFusionRatioNoAtmoicFusedName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
    auto *fusedHighFusionRatioNoAtmoicFused = new FusedSpMMSpMMFusedParReduceNoAtomic(inSpMM,stats, ufThreadsPerBlock, ts);
    fusedHighFusionRatioNoAtmoicFused->run();
    statList.push_back(fusedHighFusionRatioNoAtmoicFused->printStats());
    delete fusedHighFusionRatioNoAtmoicFused;
    delete stats;

    std::string fusedHighFusionRatioCSRCSCNoAtmoicFusedName = "GPU_CSRCSCNoAtomicFused_Reordered_HighFusionRatio_" + std::to_string(ts);
    stats = new swiftware::benchmark::Stats(fusedHighFusionRatioCSRCSCNoAtmoicFusedName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
    auto *fusedHighFusionRatioCSRCSCNoAtmoicFused = new FusedSpMMSpMMCSRCSCNoAtomic(inSpMM,stats, ufThreadsPerBlock);
    fusedHighFusionRatioCSRCSCNoAtmoicFused->run();
    statList.push_back(fusedHighFusionRatioCSRCSCNoAtmoicFused->printStats());
    delete fusedHighFusionRatioCSRCSCNoAtmoicFused;
    delete stats;

    std::string fusedHighFusionRatioCSRCSCAtmoicFusedName = "GPU_CSRCSCAtomicFused_Reordered_HighFusionRatio_" + std::to_string(ts);
    stats = new swiftware::benchmark::Stats(fusedHighFusionRatioCSRCSCAtmoicFusedName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
    auto *fusedHighFusionRatioCSRCSCAtmoicFused = new FusedSpMMSpMMCSRCSC(inSpMM,stats, ufThreadsPerBlock);
    fusedHighFusionRatioCSRCSCAtmoicFused->run();
    statList.push_back(fusedHighFusionRatioCSRCSCAtmoicFused->printStats());
    delete fusedHighFusionRatioCSRCSCNoAtmoicFused;
    delete stats;

//    std::string fusedHighFusionRatioNoSynchName = "GPU_NoSynch_Reordered_HighFusionRatio_" + std::to_string(ts);
//    stats = new swiftware::benchmark::Stats(fusedHighFusionRatioNoSynchName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
//    auto *fusedHighFusionRatioNoSynch = new FusedSpMMSpMMHighFusionRatioNoSynch(inSpMM,stats, ufThreadsPerBlock, ts);
//    fusedHighFusionRatioNoSynch->run();
//    statList.push_back(fusedHighFusionRatioNoSynch->printStats());
//    delete fusedHighFusionRatioNoSynch;
//    delete stats;

//    std::string fusedHighFusionRatio_32Name = "GPU_Fused_Reordered_32_HighFusionRatio_" + std::to_string(ts);
//    stats = new swiftware::benchmark::Stats(fusedHighFusionRatio_32Name,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
//    auto *fusedHighFusionRatio_32 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, 32, ufThreadsPerBlock, ts);
//    fusedHighFusionRatio_32->run();
//    statList.push_back(fusedHighFusionRatio_32->printStats());
//    delete fusedHighFusionRatio_32;
//    delete stats;

//    std::string fusedRedHighFusionRatioName = "GPU_Fused_Redundant_HighFusionRatio_" + std::to_string(ts);
//    stats = new swiftware::benchmark::Stats(fusedRedHighFusionRatioName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
//    auto *fusedRedHighFusionRatio = new FusedSpMMSpMMSeqReduceRowBalanceRedundant(inSpMM,stats, ufThreadsPerBlock, ts);
//    fusedRedHighFusionRatio->run();
//    statList.push_back(fusedRedHighFusionRatio->printStats());
//    delete fusedRedHighFusionRatio;
//    delete stats;

//    std::string fusedSeqReduceWSMName = "GPU_Fused_SeqReduceBColsBlocking_WSM_" + std::to_string(ts);
//    stats = new swiftware::benchmark::Stats(fusedSeqReduceWSMName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
//    auto *fusedSeqReduceWSM = new FusedSpMMSpMMSeqReduceBColsBlockingWithSharedMem(inSpMM,stats, fThreadsPerBlock, ts);
//    fusedSeqReduceWSM->run();
//    statList.push_back(fusedSeqReduceWSM->printStats());
//    delete fusedSeqReduceWSM;
//    delete stats;
//
//    std::string fusedSeqReduceBColsBlockingName = "GPU_Fused_SeqReduceBColsBlocking_" + std::to_string(ts);
//    stats = new swiftware::benchmark::Stats(fusedSeqReduceBColsBlockingName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
//    auto *fusedSeqReduceBColsBlocking = new FusedSpMMSpMMSeqReduceBColsBlocking(inSpMM,stats, fThreadsPerBlock, ts);
//    fusedSeqReduceBColsBlocking->run();
//    statList.push_back(fusedSeqReduceBColsBlocking->printStats());
//    delete fusedSeqReduceBColsBlocking;
//    delete stats;
//
//    std::string fusedSeqReduceBColsBlockingWSMName = "GPU_Fused_SeqReduceBColsBlocking_WSM_" + std::to_string(ts);
//    stats = new swiftware::benchmark::Stats(fusedSeqReduceBColsBlockingWSMName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
//    auto *fusedSeqReduceBColsBlockingWSM = new FusedSpMMSpMMSeqReduceBColsBlockingWithSharedMem(inSpMM,stats, fThreadsPerBlock, ts);
//    fusedSeqReduceBColsBlockingWSM->run();
//    statList.push_back(fusedSeqReduceBColsBlockingWSM->printStats());
//    delete fusedSeqReduceBColsBlockingWSM;
//    delete stats;



  }

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
  std::cout << cpuSpMMSpMMStat << spStat + tpStat + profStat << std::endl;
  std::cout << unfusedSeqReduceRowBalanceStat << spStat + tpStat + profStat << std::endl;
  for (auto implStat : statList){
    std::cout << implStat << spStat + tpStat + profStat << std::endl;
  }

  delete inSpMM;

}
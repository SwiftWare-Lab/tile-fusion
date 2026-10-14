//
// Created by salehm32 on 20/06/24.
//
#include "../Cuda_SpMM_SpMM_Demo_Utils.h"

#include <cstdlib>
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
  const char *atomicRowsEnv = std::getenv("SPMM_SPMM_CSRCSC_ROWS_PER_BLOCK");
  const int csrCscRowsPerBlock = atomicRowsEnv ? std::atoi(atomicRowsEnv) : 1;
  if (csrCscRowsPerBlock < 1 || csrCscRowsPerBlock > 8) {
    std::cerr << "SPMM_SPMM_CSRCSC_ROWS_PER_BLOCK must be 1..8" << std::endl;
    return 2;
  }
  const char *conflictRowsEnv =
      std::getenv("SPMM_SPMM_CONFLICT_ROWS_PER_BLOCK");
  const int conflictRowsPerBlock =
      conflictRowsEnv ? std::atoi(conflictRowsEnv) : 4;
  const char *conflictSavingsEnv =
      std::getenv("SPMM_SPMM_CONFLICT_MIN_SAVINGS_PCT");
  const int conflictMinSavings =
      conflictSavingsEnv ? std::atoi(conflictSavingsEnv) : 40;
  if (conflictRowsPerBlock < 2 || conflictRowsPerBlock > 8) {
    std::cerr << "SPMM_SPMM_CONFLICT_ROWS_PER_BLOCK must be 2..8"
              << std::endl;
    return 2;
  }
  if (conflictMinSavings < 0 || conflictMinSavings > 100) {
    std::cerr << "SPMM_SPMM_CONFLICT_MIN_SAVINGS_PCT must be 0..100"
              << std::endl;
    return 2;
  }
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

  // A focused comparison of unfused row balance, cuSPARSE ALG3, and
  // CSR-CSC atomic. The CPU result is still used for correctness checks.
  if (std::getenv("SPMM_SPMM_ATOMIC_ONLY")) {
    stats = new swiftware::benchmark::Stats(
        "GPU_Unfused_SeqReduceRowBalance", "SpMMSpMM", numTrial,
        tp._matrix_name, numThread);
    auto *unfused = new SpMMSpMMSeqReduceRowBalance(
        inSpMM, stats, ufThreadsPerBlock);
    unfused->run();
    const auto unfusedStat = unfused->printStats();
    delete unfused;
    delete stats;

    stats = new swiftware::benchmark::Stats(
        "GPU_Unfused_CuSparse_ALG3", "SpMMSpMM", numTrial,
        tp._matrix_name, numThread);
    auto *cuSparse = new SpMMSpMMCuSparse(
        inSpMM, stats, CUSPARSE_SPMM_CSR_ALG3);
    cuSparse->run();
    const auto cuSparseStat = cuSparse->printStats();
    delete cuSparse;
    delete stats;

    stats = new swiftware::benchmark::Stats(
        "GPU_CSRCSC_Fused_Atomic", "SpMMSpMM", numTrial,
        tp._matrix_name, numThread);
    auto *atomic = new FusedSpMMSpMMCSRCSCAtomic(
        inSpMM, stats, ufThreadsPerBlock, csrCscRowsPerBlock);
    atomic->run();
    const auto atomicStat = atomic->printStats();
    delete atomic;
    delete stats;

    stats = new swiftware::benchmark::Stats(
        "GPU_CSRCSC_Fused_ConflictReduce", "SpMMSpMM", numTrial,
        tp._matrix_name, numThread);
    auto *conflictReduce = new FusedSpMMSpMMCSRCSCConflictReduce(
        inSpMM, stats, ufThreadsPerBlock, conflictRowsPerBlock,
        conflictMinSavings);
    conflictReduce->run();
    const auto conflictReduceStat = conflictReduce->printStats();
    delete conflictReduce;
    delete stats;

    const auto spCsv = sp.print_csv(true);
    const auto tpCsv = tp.print_csv(true);
    const std::string suffix = std::get<1>(spCsv) + std::get<1>(tpCsv);
    if (tp.print_header)
      std::cout << headerStat + std::get<0>(spCsv) + std::get<0>(tpCsv)
                << std::endl;
    std::cout << cpuSpMMSpMMStat << suffix << std::endl;
    std::cout << unfusedStat << suffix << std::endl;
    std::cout << cuSparseStat << suffix << std::endl;
    std::cout << atomicStat << suffix << std::endl;
    std::cout << conflictReduceStat << suffix << std::endl;
    delete inSpMM;
    return 0;
  }

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
  const bool hybridCooperative =
      std::getenv("SPMM_SPMM_HYBRID_COOPERATIVE") != nullptr;
  const bool hybridRows4 = hybridCooperative ||
      std::getenv("SPMM_SPMM_HYBRID_128") != nullptr;
  const bool hybridOnly = hybridRows4 ||
      std::getenv("SPMM_SPMM_HYBRID_ONLY") != nullptr;
  std::vector <std::string> statList;

  stats = new swiftware::benchmark::Stats(
      "GPU_CSRCSC_Fused_Atomic", "SpMMSpMM", numTrial, tp._matrix_name,
      numThread);
  auto *csrCscAtomic = new FusedSpMMSpMMCSRCSCAtomic(
      inSpMM, stats, ufThreadsPerBlock, csrCscRowsPerBlock);
  csrCscAtomic->run();
  statList.push_back(csrCscAtomic->printStats());
  delete csrCscAtomic;
  delete stats;

  stats = new swiftware::benchmark::Stats(
      "GPU_CSRCSC_Fused_ConflictReduce", "SpMMSpMM", numTrial,
      tp._matrix_name, numThread);
  auto *csrCscConflictReduce = new FusedSpMMSpMMCSRCSCConflictReduce(
      inSpMM, stats, ufThreadsPerBlock, conflictRowsPerBlock,
      conflictMinSavings);
  csrCscConflictReduce->run();
  statList.push_back(csrCscConflictReduce->printStats());
  delete csrCscConflictReduce;
  delete stats;

  // The focused mode sweeps both parameters. Preserve the smaller original
  // sweep when running the full collection of legacy implementations.
  const std::vector<int> hybridTileSizes = hybridRows4
      ? std::vector<int>{4, 8, 16, 32, 64, 128, 256}
      : (hybridOnly ? std::vector<int>{4, 8, 16, 32}
                    : std::vector<int>{8});
  const std::vector<int> hybridThresholds = hybridOnly
      ? std::vector<int>{4, 8, 16, 32, 64}
      : std::vector<int>{16, 32, 64, 128};
  if (hybridOnly) {
    for (int tileSize : hybridTileSizes) {
      for (int maxRowNnz : hybridThresholds) {
        stats = new swiftware::benchmark::Stats(
            std::string(hybridCooperative
                            ? "GPU_CSRCSC_HybridCooperative"
                            : (hybridRows4 ? "GPU_CSRCSC_HybridColored128"
                                           : "GPU_CSRCSC_HybridColored")) +
                "_T" + std::to_string(tileSize) + "_MaxNNZ_" +
                std::to_string(maxRowNnz),
            "SpMMSpMM", numTrial, tp._matrix_name, numThread);
        auto *hybrid = new FusedSpMMSpMMCSRCSCHybridColored(
            inSpMM, stats, ufThreadsPerBlock, tileSize, maxRowNnz,
            hybridRows4, hybridCooperative);
        hybrid->run();
        statList.push_back(hybrid->printStats());
        delete hybrid;
        delete stats;
      }
    }
  }

  // The full demo also runs legacy kernels. This mode isolates the hybrid
  // comparison on matrices where an unrelated legacy variant cannot run.
  if (hybridOnly) {
    stats = new swiftware::benchmark::Stats(
        "GPU_CSRCSR_Fused_Atomic_8", "SpMMSpMM", numTrial,
        tp._matrix_name, numThread);
    auto *csrCsrAtomic = new FusedSpMMSpMMFusedParReduce(
        inSpMM, stats, ufThreadsPerBlock, 8);
    csrCsrAtomic->run();
    statList.push_back(csrCsrAtomic->printStats());
    delete csrCsrAtomic;
    delete stats;

    stats = new swiftware::benchmark::Stats(
        "GPU_CSRCSC_Fused_Colored_8", "SpMMSpMM", numTrial,
        tp._matrix_name, numThread);
    auto *colored = new FusedSpMMSpMMCSRCSCColored(
        inSpMM, stats, ufThreadsPerBlock, 8);
    colored->run();
    statList.push_back(colored->printStats());
    delete colored;
    delete stats;

    auto csvInfo = sp.print_csv(true);
    auto tpCsv = tp.print_csv(true);
    const std::string suffix = std::get<1>(csvInfo) + std::get<1>(tpCsv);
    if (tp.print_header)
      std::cout << headerStat + std::get<0>(csvInfo) + std::get<0>(tpCsv)
                << std::endl;
    std::cout << cpuSpMMSpMMStat << suffix << std::endl;
    std::cout << unfusedCuSparseAlg2Stat << suffix << std::endl;
    std::cout << unfusedCuSparseAlg3Stat << suffix << std::endl;
    std::cout << unfusedSeqReduceRowBalanceStat << suffix << std::endl;
    for (const auto &implStat : statList)
      std::cout << implStat << suffix << std::endl;
    delete inSpMM;
    return 0;
  }

  for (auto ts: tileSizes){
    std::string unfusedSeqReduceRowCoarsenedName = "GPU_Unfused_SeqReduceRowCoarsened_" + std::to_string(ts);
    stats = new swiftware::benchmark::Stats(unfusedSeqReduceRowCoarsenedName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
    auto *unfusedSeqReduceRowCoarsened = new SpMMSpMMSeqReduceRowBalanceCoarsenedRow(inSpMM,stats, ufThreadsPerBlock, ts);
    unfusedSeqReduceRowCoarsened->run();
    statList.push_back(unfusedSeqReduceRowCoarsened->printStats());
    delete unfusedSeqReduceRowCoarsened;
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

    // Permit the remaining variants to be measured when this legacy analysis
    // crashes on a particular matrix; the default still runs every variant.
    if (!std::getenv("SPMM_SPMM_SKIP_2LN")) {
      std::string fusedHighFusionRatio2LNName = "GPU_Fused_Reordered_HighFusionRatio_2LN_" + std::to_string(ts);
      stats = new swiftware::benchmark::Stats(fusedHighFusionRatio2LNName,"SpMMSpMM", numTrial,tp._matrix_name,numThread);
      auto *fusedHighFusionRatio2LN = new FusedSpMMSpMMHighFusionRatio2Level(inSpMM,stats, ufThreadsPerBlock, ts);
      fusedHighFusionRatio2LN->run();
      statList.push_back(fusedHighFusionRatio2LN->printStats());
      delete fusedHighFusionRatio2LN;
      delete stats;
    }

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
  std::cout << unfusedCuSparseAlg2Stat << spStat + tpStat + profStat << std::endl;
  std::cout << unfusedCuSparseAlg3Stat << spStat + tpStat + profStat << std::endl;
  std::cout << unfusedSeqReduceRowBalanceStat << spStat + tpStat + profStat << std::endl;
  for (auto implStat : statList){
    std::cout << implStat << spStat + tpStat + profStat << std::endl;
  }

  delete inSpMM;

}

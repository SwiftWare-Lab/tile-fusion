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

  stats = new swiftware::benchmark::Stats("GPU_Unfused_SeqReduceRowCoarsened_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedSeqReduceRowCoarsened8 = new SpMMSpMMSeqReduceRowBalanceCoarsenedRow(inSpMM,stats, ufThreadsPerBlock, 8);
  unfusedSeqReduceRowCoarsened8->run();
  //  std::cout << "UNFUSED: " << std::endl;
//    unfusedSeqReduceRowCoarsened8->OutTensor->printDx();
  auto unfusedSeqReduceRowCoarsened8Stat = unfusedSeqReduceRowCoarsened8->printStats();
  delete unfusedSeqReduceRowCoarsened8;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Unfused_SeqReduceRowCoarsened_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedSeqReduceRowCoarsened16 = new SpMMSpMMSeqReduceRowBalanceCoarsenedRow(inSpMM,stats, ufThreadsPerBlock, 16);
  unfusedSeqReduceRowCoarsened16->run();
  //  std::cout << "UNFUSED: " << std::endl;
  //    unfusedSeqReduceRowCoarsened16->OutTensor->printDx();
  auto unfusedSeqReduceRowCoarsened16Stat = unfusedSeqReduceRowCoarsened16->printStats();
  delete unfusedSeqReduceRowCoarsened16;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Unfused_SeqReduceRowCoarsened_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedSeqReduceRowCoarsened32 = new SpMMSpMMSeqReduceRowBalanceCoarsenedRow(inSpMM,stats, ufThreadsPerBlock, 32);
  unfusedSeqReduceRowCoarsened32->run();
  //  std::cout << "UNFUSED: " << std::endl;
  //    unfusedSeqReduceRowCoarsened32->OutTensor->printDx();
  auto unfusedSeqReduceRowCoarsened32Stat = unfusedSeqReduceRowCoarsened32->printStats();
  delete unfusedSeqReduceRowCoarsened32;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Unfused_SeqReduceRowCoarsened_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedSeqReduceRowCoarsened64 = new SpMMSpMMSeqReduceRowBalanceCoarsenedRow(inSpMM,stats, ufThreadsPerBlock, 64);
  unfusedSeqReduceRowCoarsened64->run();
  //  std::cout << "UNFUSED: " << std::endl;
//      unfusedSeqReduceRowCoarsened64->OutTensor->printDx();
  auto unfusedSeqReduceRowCoarsened64Stat = unfusedSeqReduceRowCoarsened64->printStats();
  delete unfusedSeqReduceRowCoarsened64;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Unfused_SeqReduceRowCoarsenedStride_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedSeqReduceRowCoarsenedStride8 = new SpMMSpMMSeqReduceRowBalanceCoarsenedRowStride(inSpMM,stats, ufThreadsPerBlock, 8);
  unfusedSeqReduceRowCoarsenedStride8->run();
  //  std::cout << "UNFUSED: " << std::endl;
//      unfusedSeqReduceRowCoarsenedStride8->OutTensor->printDx();
  auto unfusedSeqReduceRowCoarsenedStride8Stat = unfusedSeqReduceRowCoarsenedStride8->printStats();
  delete unfusedSeqReduceRowCoarsenedStride8;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Unfused_SeqReduceRowCoarsenedStride_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedSeqReduceRowCoarsenedStride16 = new SpMMSpMMSeqReduceRowBalanceCoarsenedRowStride(inSpMM,stats, ufThreadsPerBlock, 16);
  unfusedSeqReduceRowCoarsenedStride16->run();
  //  std::cout << "UNFUSED: " << std::endl;
  //      unfusedSeqReduceRowCoarsenedStride16->OutTensor->printDx();
  auto unfusedSeqReduceRowCoarsenedStride16Stat = unfusedSeqReduceRowCoarsenedStride16->printStats();
  delete unfusedSeqReduceRowCoarsenedStride16;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Unfused_SeqReduceRowCoarsenedStride_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedSeqReduceRowCoarsenedStride32 = new SpMMSpMMSeqReduceRowBalanceCoarsenedRowStride(inSpMM,stats, ufThreadsPerBlock, 32);
  unfusedSeqReduceRowCoarsenedStride32->run();
  //  std::cout << "UNFUSED: " << std::endl;
  //      unfusedSeqReduceRowCoarsenedStride32->OutTensor->printDx();
  auto unfusedSeqReduceRowCoarsenedStride32Stat = unfusedSeqReduceRowCoarsenedStride32->printStats();
  delete unfusedSeqReduceRowCoarsenedStride32;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Unfused_SeqReduceRowCoarsenedStride_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *unfusedSeqReduceRowCoarsenedStride64 = new SpMMSpMMSeqReduceRowBalanceCoarsenedRowStride(inSpMM,stats, ufThreadsPerBlock, 64);
  unfusedSeqReduceRowCoarsenedStride64->run();
  //  std::cout << "UNFUSED: " << std::endl;
  //      unfusedSeqReduceRowCoarsenedStride64->OutTensor->printDx();
  auto unfusedSeqReduceRowCoarsenedStride64Stat = unfusedSeqReduceRowCoarsenedStride64->printStats();
  delete unfusedSeqReduceRowCoarsenedStride64;
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

  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatio8 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 8);
  fusedHighFusionRatio8->run();
  //  std::cout << "FUSED: " << std::endl;
  //  fusedHighFusionRatio8->OutTensor->printDx();
  auto fusedHighFusionRatio8Stat = fusedHighFusionRatio8->printStats();
  delete fusedHighFusionRatio8;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatio16 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 16);
  fusedHighFusionRatio16->run();
  //  std::cout << "FUSED: " << std::endl;
  //  fusedHighFusionRatio16->OutTensor->printDx();
  auto fusedHighFusionRatio16Stat = fusedHighFusionRatio16->printStats();
  delete fusedHighFusionRatio16;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatio32 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 32);
  fusedHighFusionRatio32->run();
  //  std::cout << "FUSED: " << std::endl;
  //  fusedHighFusionRatio32->OutTensor->printDx();
  auto fusedHighFusionRatio32Stat = fusedHighFusionRatio32->printStats();
  delete fusedHighFusionRatio32;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatio64 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 64);
  fusedHighFusionRatio64->run();
  //  std::cout << "FUSED: " << std::endl;
//    fusedHighFusionRatio64->OutTensor->printDx();
  auto fusedHighFusionRatio64Stat = fusedHighFusionRatio64->printStats();
  delete fusedHighFusionRatio64;
  delete stats;

//  stats = new swiftware::benchmark::Stats("GPU_Fused_ReorderedResultPacked_HighFusionRatio_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioResultPacked8 = new FusedSpMMSpMMHighFusionRatioResultPacked(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 8);
//  fusedHighFusionRatioResultPacked8->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatioResultPacked8->OutTensor->printDx();
//  auto fusedHighFusionRatioResultPacked8Stat = fusedHighFusionRatioResultPacked8->printStats();
//  delete fusedHighFusionRatioResultPacked8;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_ReorderedResultPacked_HighFusionRatio_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioResultPacked32 = new FusedSpMMSpMMHighFusionRatioResultPacked(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 32);
//  fusedHighFusionRatioResultPacked32->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatioResultPacked32->OutTensor->printDx();
//  auto fusedHighFusionRatioResultPacked32Stat = fusedHighFusionRatioResultPacked32->printStats();
//  delete fusedHighFusionRatioResultPacked32;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_ReorderedResultPacked_HighFusionRatio_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioResultPacked16 = new FusedSpMMSpMMHighFusionRatioResultPacked(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 16);
//  fusedHighFusionRatioResultPacked16->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatioResultPacked16->OutTensor->printDx();
//  auto fusedHighFusionRatioResultPacked16Stat = fusedHighFusionRatioResultPacked16->printStats();
//  delete fusedHighFusionRatioResultPacked16;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_ReorderedResultPacked_HighFusionRatio_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioResultPacked64 = new FusedSpMMSpMMHighFusionRatioResultPacked(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 64);
//  fusedHighFusionRatioResultPacked64->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatioResultPacked64->OutTensor->printDx();
//  auto fusedHighFusionRatioResultPacked64Stat = fusedHighFusionRatioResultPacked64->printStats();
//  delete fusedHighFusionRatioResultPacked64;
//  delete stats;

//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_P4_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioP48 = new FusedSpMMSpMMHighFusionRatioProductOf4Fused(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 8);
//  fusedHighFusionRatioP48->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatio64->OutTensor->printDx();
//  auto fusedHighFusionRatioP48Stat = fusedHighFusionRatioP48->printStats();
//  delete fusedHighFusionRatioP48;
//  delete stats;

//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_P4_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioP416 = new FusedSpMMSpMMHighFusionRatioProductOf4Fused(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 16);
//  fusedHighFusionRatioP416->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatio64->OutTensor->printDx();
//  auto fusedHighFusionRatioP416Stat = fusedHighFusionRatioP416->printStats();
//  delete fusedHighFusionRatioP416;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_P4_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioP432 = new FusedSpMMSpMMHighFusionRatioProductOf4Fused(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 32);
//  fusedHighFusionRatioP432->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatio64->OutTensor->printDx();
//  auto fusedHighFusionRatioP432Stat = fusedHighFusionRatioP432->printStats();
//  delete fusedHighFusionRatioP432;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_P4_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioP464 = new FusedSpMMSpMMHighFusionRatioProductOf4Fused(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 64);
//  fusedHighFusionRatioP464->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatio64->OutTensor->printDx();
//  auto fusedHighFusionRatioP464Stat = fusedHighFusionRatioP464->printStats();
//  delete fusedHighFusionRatioP464;
//  delete stats;

//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatioStride_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioStride8 = new FusedSpMMSpMMHighFusionRatioStride(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 8);
//  fusedHighFusionRatioStride8->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatioStride8->OutTensor->printDx();
//  auto fusedHighFusionRatioStride8Stat = fusedHighFusionRatioStride8->printStats();
//  delete fusedHighFusionRatioStride8;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatioStride_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioStride16 = new FusedSpMMSpMMHighFusionRatioStride(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 16);
//  fusedHighFusionRatioStride16->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatioStride16->OutTensor->printDx();
//  auto fusedHighFusionRatioStride16Stat = fusedHighFusionRatioStride16->printStats();
//  delete fusedHighFusionRatioStride16;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatioStride_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioStride32 = new FusedSpMMSpMMHighFusionRatioStride(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 32);
//  fusedHighFusionRatioStride32->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatioStride32->OutTensor->printDx();
//  auto fusedHighFusionRatioStride32Stat = fusedHighFusionRatioStride32->printStats();
//  delete fusedHighFusionRatioStride32;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatioStride_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioStride64 = new FusedSpMMSpMMHighFusionRatioStride(inSpMM,stats, ufThreadsPerBlock, ufThreadsPerBlock, 64);
//  fusedHighFusionRatioStride64->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatioStride64->OutTensor->printDx();
//  auto fusedHighFusionRatioStride64Stat = fusedHighFusionRatioStride64->printStats();
//  delete fusedHighFusionRatioStride64;
//  delete stats;



//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_MBC_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioMBC8 = new FusedSpMMSpMMHighFusionRatioMultipleBCols(inSpMM,stats, ufThreadsPerBlock, 8);
//  fusedHighFusionRatioMBC8->run();
//  //  std::cout << "FUSED: " << std::endl;
////    fusedHighFusionRatioMBC8->OutTensor->printDx();
//  auto fusedHighFusionRatioMBC8Stat = fusedHighFusionRatioMBC8->printStats();
//  delete fusedHighFusionRatioMBC8;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_MBC_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioMBC16 = new FusedSpMMSpMMHighFusionRatioMultipleBCols(inSpMM,stats, ufThreadsPerBlock, 16);
//  fusedHighFusionRatioMBC16->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatioMBC16->OutTensor->printDx();
//  auto fusedHighFusionRatioMBC16Stat = fusedHighFusionRatioMBC16->printStats();
//  delete fusedHighFusionRatioMBC16;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_MBC_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioMBC32 = new FusedSpMMSpMMHighFusionRatioMultipleBCols(inSpMM,stats, ufThreadsPerBlock, 32);
//  fusedHighFusionRatioMBC32->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatioMBC32->OutTensor->printDx();
//  auto fusedHighFusionRatioMBC32Stat = fusedHighFusionRatioMBC32->printStats();
//  delete fusedHighFusionRatioMBC32;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_HighFusionRatio_MBC_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioMBC64 = new FusedSpMMSpMMHighFusionRatioMultipleBCols(inSpMM,stats, ufThreadsPerBlock, 64);
//  fusedHighFusionRatioMBC64->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatioMBC64->OutTensor->printDx();
//  auto fusedHighFusionRatioMBC64Stat = fusedHighFusionRatioMBC64->printStats();
//  delete fusedHighFusionRatioMBC64;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_2LFused_Reordered_HighFusionRatio_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatio2L8 = new FusedSpMMSpMMHighFusionRatio2Level(inSpMM,stats, ufThreadsPerBlock, 8);
//  fusedHighFusionRatio2L8->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatio2L64->OutTensor->printDx();
//  auto fusedHighFusionRatio2L8Stat = fusedHighFusionRatio2L8->printStats();
//  delete fusedHighFusionRatio2L8;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_2LFused_Reordered_HighFusionRatio_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatio2L16 = new FusedSpMMSpMMHighFusionRatio2Level(inSpMM,stats, ufThreadsPerBlock, 16);
//  fusedHighFusionRatio2L16->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatio2L64->OutTensor->printDx();
//  auto fusedHighFusionRatio2L16Stat = fusedHighFusionRatio2L16->printStats();
//  delete fusedHighFusionRatio2L16;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_2LFused_Reordered_HighFusionRatio_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatio2L32 = new FusedSpMMSpMMHighFusionRatio2Level(inSpMM,stats, ufThreadsPerBlock, 32);
//  fusedHighFusionRatio2L32->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //    fusedHighFusionRatio2L64->OutTensor->printDx();
//  auto fusedHighFusionRatio2L32Stat = fusedHighFusionRatio2L32->printStats();
//  delete fusedHighFusionRatio2L32;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_2LFused_Reordered_HighFusionRatio_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatio2L64 = new FusedSpMMSpMMHighFusionRatio2Level(inSpMM,stats, ufThreadsPerBlock, 64);
//  fusedHighFusionRatio2L64->run();
//  //  std::cout << "FUSED: " << std::endl;
////    fusedHighFusionRatio2L64->OutTensor->printDx();
//  auto fusedHighFusionRatio2L64Stat = fusedHighFusionRatio2L64->printStats();
//  delete fusedHighFusionRatio2L64;
//  delete stats;
//
  stats = new swiftware::benchmark::Stats("GPU_AtomicFused_Reordered_HighFusionRatio_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatioAtmoicFused8 = new FusedSpMMSpMMFusedParReduce(inSpMM,stats, ufThreadsPerBlock, 8);
  fusedHighFusionRatioAtmoicFused8->run();
  //  std::cout << "FUSED: " << std::endl;
//    fusedHighFusionRatioAtmoicFused64->OutTensor->printDx();
  auto fusedHighFusionRatioAtmoicFused8Stat = fusedHighFusionRatioAtmoicFused8->printStats();
  delete fusedHighFusionRatioAtmoicFused8;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_AtomicFused_Reordered_HighFusionRatio_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatioAtmoicFused16 = new FusedSpMMSpMMFusedParReduce(inSpMM,stats, ufThreadsPerBlock, 26);
  fusedHighFusionRatioAtmoicFused16->run();
  //  std::cout << "FUSED: " << std::endl;
  //    fusedHighFusionRatioAtmoicFused64->OutTensor->printDx();
  auto fusedHighFusionRatioAtmoicFused16Stat = fusedHighFusionRatioAtmoicFused16->printStats();
  delete fusedHighFusionRatioAtmoicFused16;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_AtomicFused_Reordered_HighFusionRatio_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatioAtmoicFused32 = new FusedSpMMSpMMFusedParReduce(inSpMM,stats, ufThreadsPerBlock, 32);
  fusedHighFusionRatioAtmoicFused32->run();
  //  std::cout << "FUSED: " << std::endl;
  //    fusedHighFusionRatioAtmoicFused64->OutTensor->printDx();
  auto fusedHighFusionRatioAtmoicFused32Stat = fusedHighFusionRatioAtmoicFused32->printStats();
  delete fusedHighFusionRatioAtmoicFused32;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_AtomicFused_Reordered_HighFusionRatio_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatioAtmoicFused64 = new FusedSpMMSpMMFusedParReduce(inSpMM,stats, ufThreadsPerBlock, 64);
  fusedHighFusionRatioAtmoicFused64->run();
  //  std::cout << "FUSED: " << std::endl;
  //    fusedHighFusionRatioAtmoicFused64->OutTensor->printDx();
  auto fusedHighFusionRatioAtmoicFused64Stat = fusedHighFusionRatioAtmoicFused64->printStats();
  delete fusedHighFusionRatioAtmoicFused64;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_NoAtomicFused_Reordered_HighFusionRatio_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatioNoAtmoicFused8 = new FusedSpMMSpMMFusedParReduceNoAtomic(inSpMM,stats, ufThreadsPerBlock, 8);
  fusedHighFusionRatioNoAtmoicFused8->run();
  //  std::cout << "FUSED: " << std::endl;
  //    fusedHighFusionRatioNoAtmoicFused64->OutTensor->printDx();
  auto fusedHighFusionRatioNoAtmoicFused8Stat = fusedHighFusionRatioNoAtmoicFused8->printStats();
  delete fusedHighFusionRatioNoAtmoicFused8;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_NoAtomicFused_Reordered_HighFusionRatio_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatioNoAtmoicFused16 = new FusedSpMMSpMMFusedParReduceNoAtomic(inSpMM,stats, ufThreadsPerBlock, 26);
  fusedHighFusionRatioNoAtmoicFused16->run();
  //  std::cout << "FUSED: " << std::endl;
  //    fusedHighFusionRatioNoAtmoicFused64->OutTensor->printDx();
  auto fusedHighFusionRatioNoAtmoicFused16Stat = fusedHighFusionRatioNoAtmoicFused16->printStats();
  delete fusedHighFusionRatioNoAtmoicFused16;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_NoAtomicFused_Reordered_HighFusionRatio_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatioNoAtmoicFused32 = new FusedSpMMSpMMFusedParReduceNoAtomic(inSpMM,stats, ufThreadsPerBlock, 32);
  fusedHighFusionRatioNoAtmoicFused32->run();
  //  std::cout << "FUSED: " << std::endl;
  //    fusedHighFusionRatioNoAtmoicFused64->OutTensor->printDx();
  auto fusedHighFusionRatioNoAtmoicFused32Stat = fusedHighFusionRatioNoAtmoicFused32->printStats();
  delete fusedHighFusionRatioNoAtmoicFused32;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_NoAtomicFused_Reordered_HighFusionRatio_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatioNoAtmoicFused64 = new FusedSpMMSpMMFusedParReduceNoAtomic(inSpMM,stats, ufThreadsPerBlock, 64);
  fusedHighFusionRatioNoAtmoicFused64->run();
  //  std::cout << "FUSED: " << std::endl;
  //    fusedHighFusionRatioNoAtmoicFused64->OutTensor->printDx();
  auto fusedHighFusionRatioNoAtmoicFused64Stat = fusedHighFusionRatioNoAtmoicFused64->printStats();
  delete fusedHighFusionRatioNoAtmoicFused64;
  delete stats;

//  stats = new swiftware::benchmark::Stats("GPU_FusedNoSynch_Reordered_HighFusionRatio_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioNoSynch8 = new FusedSpMMSpMMHighFusionRatioNoSynch(inSpMM,stats, ufThreadsPerBlock, 8);
//  fusedHighFusionRatioNoSynch8->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatioNoSynch8->OutTensor->printDx();
//  auto fusedHighFusionRatioNoSynch8Stat = fusedHighFusionRatioNoSynch8->printStats();
//  delete fusedHighFusionRatioNoSynch8;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_FusedNoSynch_Reordered_HighFusionRatio_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioNoSynch16 = new FusedSpMMSpMMHighFusionRatioNoSynch(inSpMM,stats, ufThreadsPerBlock, 16);
//  fusedHighFusionRatioNoSynch16->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatioNoSynch16->OutTensor->printDx();
//  auto fusedHighFusionRatioNoSynch16Stat = fusedHighFusionRatioNoSynch16->printStats();
//  delete fusedHighFusionRatioNoSynch16;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_FusedNoSynch_Reordered_HighFusionRatio_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioNoSynch32 = new FusedSpMMSpMMHighFusionRatioNoSynch(inSpMM,stats, ufThreadsPerBlock, 32);
//  fusedHighFusionRatioNoSynch32->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatioNoSynch32->OutTensor->printDx();
//  auto fusedHighFusionRatioNoSynch32Stat = fusedHighFusionRatioNoSynch32->printStats();
//  delete fusedHighFusionRatioNoSynch32;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_FusedNoSynch_Reordered_HighFusionRatio_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatioNoSynch64 = new FusedSpMMSpMMHighFusionRatioNoSynch(inSpMM,stats, ufThreadsPerBlock, 64);
//  fusedHighFusionRatioNoSynch64->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatioNoSynch64->OutTensor->printDx();
//  auto fusedHighFusionRatioNoSynch64Stat = fusedHighFusionRatioNoSynch64->printStats();
//  delete fusedHighFusionRatioNoSynch64;
//  delete stats;


//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_32_HighFusionRatio_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatio_32_8 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, 32, ufThreadsPerBlock, 8);
//  fusedHighFusionRatio_32_8->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatio_32_8->OutTensor->printDx();
//  auto fusedHighFusionRatio_32_8Stat = fusedHighFusionRatio_32_8->printStats();
//  delete fusedHighFusionRatio_32_8;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_32_HighFusionRatio_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatio_32_16 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, 32, ufThreadsPerBlock, 16);
//  fusedHighFusionRatio_32_16->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatio_32_16->OutTensor->printDx();
//  auto fusedHighFusionRatio_32_16Stat = fusedHighFusionRatio_32_16->printStats();
//  delete fusedHighFusionRatio_32_16;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_32_HighFusionRatio_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatio_32_32 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, 32, ufThreadsPerBlock, 32);
//  fusedHighFusionRatio_32_32->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatio_32_32->OutTensor->printDx();
//  auto fusedHighFusionRatio_32_32Stat = fusedHighFusionRatio_32_32->printStats();
//  delete fusedHighFusionRatio_32_32;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_32_HighFusionRatio_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedHighFusionRatio_32_64 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, 32, ufThreadsPerBlock, 64);
//  fusedHighFusionRatio_32_64->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatio_32_64->OutTensor->printDx();
//  auto fusedHighFusionRatio_32_64Stat = fusedHighFusionRatio_32_64->printStats();
//  delete fusedHighFusionRatio_32_64;
//  delete stats;

//  stats = new swiftware::benchmark::Stats("GPU_Fused_Redundant_HighFusionRatio_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedRedHighFusionRatio_8 = new FusedSpMMSpMMSeqReduceRowBalanceRedundant(inSpMM,stats, ufThreadsPerBlock, 8);
//  fusedRedHighFusionRatio_8->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatio_32_64->OutTensor->printDx();
//  auto fusedRedHighFusionRatio_8Stat = fusedRedHighFusionRatio_8->printStats();
//  delete fusedRedHighFusionRatio_8;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Redundant_HighFusionRatio_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedRedHighFusionRatio_16 = new FusedSpMMSpMMSeqReduceRowBalanceRedundant(inSpMM,stats, ufThreadsPerBlock, 16);
//  fusedRedHighFusionRatio_16->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatio_32_64->OutTensor->printDx();
//  auto fusedRedHighFusionRatio_16Stat = fusedRedHighFusionRatio_16->printStats();
//  delete fusedRedHighFusionRatio_16;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Redundant_HighFusionRatio_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedRedHighFusionRatio_32 = new FusedSpMMSpMMSeqReduceRowBalanceRedundant(inSpMM,stats, ufThreadsPerBlock, 32);
//  fusedRedHighFusionRatio_32->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatio_32_64->OutTensor->printDx();
//  auto fusedRedHighFusionRatio_32Stat = fusedRedHighFusionRatio_32->printStats();
//  delete fusedRedHighFusionRatio_32;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_Redundant_HighFusionRatio_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedRedHighFusionRatio_64 = new FusedSpMMSpMMSeqReduceRowBalanceRedundant(inSpMM,stats, ufThreadsPerBlock, 64);
//  fusedRedHighFusionRatio_64->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedHighFusionRatio_32_64->OutTensor->printDx();
//  auto fusedRedHighFusionRatio_64Stat = fusedRedHighFusionRatio_64->printStats();
//  delete fusedRedHighFusionRatio_64;
//  delete stats;



//  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceBColsBlocking_WSM","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedSeqReduceWSM = new FusedSpMMSpMMSeqReduceBColsBlockingWithSharedMem(inSpMM, stats, ThreadsPerBlock, tp._b_cols);
//  fusedSeqReduceWSM->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedSeqReduceBColsBlockinWSMg->OutTensor->printDx();
//  auto fusedSeqReduceWSMStat = fusedSeqReduceWSM->printStats();
//  delete fusedSeqReduceWSM;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceBColsBlocking_4","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedSeqReduceBColsBlocking4 = new FusedSpMMSpMMSeqReduceBColsBlocking(inSpMM, stats, ThreadsPerBlock, 4);
//  fusedSeqReduceBColsBlocking4->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedSeqReduceBColsBlocking->OutTensor->printDx();
//  auto fusedSeqReduceBColsBlocking4Stat = fusedSeqReduceBColsBlocking4->printStats();
//  delete fusedSeqReduceBColsBlocking4;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceBColsBlocking_WSM_4","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedSeqReduceBColsBlockingWSM4 = new FusedSpMMSpMMSeqReduceBColsBlockingWithSharedMem(inSpMM, stats, ThreadsPerBlock, 4);
//  fusedSeqReduceBColsBlockingWSM4->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedSeqReduceBColsBlockinWSMg->OutTensor->printDx();
//  auto fusedSeqReduceBColsBlockingWSM4Stat = fusedSeqReduceBColsBlockingWSM4->printStats();
//  delete fusedSeqReduceBColsBlockingWSM4;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceBColsBlocking_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedSeqReduceBColsBlocking8 = new FusedSpMMSpMMSeqReduceBColsBlocking(inSpMM, stats, ThreadsPerBlock, 8);
//  fusedSeqReduceBColsBlocking8->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedSeqReduceBColsBlocking->OutTensor->printDx();
//  auto fusedSeqReduceBColsBlocking8Stat = fusedSeqReduceBColsBlocking8->printStats();
//  delete fusedSeqReduceBColsBlocking8;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceBColsBlocking_WSM_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedSeqReduceBColsBlockingWSM8 = new FusedSpMMSpMMSeqReduceBColsBlockingWithSharedMem(inSpMM, stats, ThreadsPerBlock, 8);
//  fusedSeqReduceBColsBlockingWSM8->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedSeqReduceBColsBlockinWSMg->OutTensor->printDx();
//  auto fusedSeqReduceBColsBlockingWSM8Stat = fusedSeqReduceBColsBlockingWSM8->printStats();
//  delete fusedSeqReduceBColsBlockingWSM8;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceBColsBlocking_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedSeqReduceBColsBlocking16 = new FusedSpMMSpMMSeqReduceBColsBlocking(inSpMM, stats, ThreadsPerBlock, 16);
//  fusedSeqReduceBColsBlocking16->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedSeqReduceBColsBlocking->OutTensor->printDx();
//  auto fusedSeqReduceBColsBlocking16Stat = fusedSeqReduceBColsBlocking16->printStats();
//  delete fusedSeqReduceBColsBlocking16;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceBColsBlocking_WSM_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedSeqReduceBColsBlockingWSM16 = new FusedSpMMSpMMSeqReduceBColsBlockingWithSharedMem(inSpMM, stats, ThreadsPerBlock, 16);
//  fusedSeqReduceBColsBlockingWSM16->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedSeqReduceBColsBlockinWSMg->OutTensor->printDx();
//  auto fusedSeqReduceBColsBlockingWSM16Stat = fusedSeqReduceBColsBlockingWSM16->printStats();
//  delete fusedSeqReduceBColsBlockingWSM16;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceBColsBlocking_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedSeqReduceBColsBlocking32 = new FusedSpMMSpMMSeqReduceBColsBlocking(inSpMM, stats, ThreadsPerBlock, 32);
//  fusedSeqReduceBColsBlocking32->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedSeqReduceBColsBlocking->OutTensor->printDx();
//  auto fusedSeqReduceBColsBlocking32Stat = fusedSeqReduceBColsBlocking32->printStats();
//  delete fusedSeqReduceBColsBlocking32;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceBColsBlocking_WSM_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedSeqReduceBColsBlockingWSM32 = new FusedSpMMSpMMSeqReduceBColsBlockingWithSharedMem(inSpMM, stats, ThreadsPerBlock, 32);
//  fusedSeqReduceBColsBlockingWSM32->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedSeqReduceBColsBlockinWSMg->OutTensor->printDx();
//  auto fusedSeqReduceBColsBlockingWSM32Stat = fusedSeqReduceBColsBlockingWSM32->printStats();
//  delete fusedSeqReduceBColsBlockingWSM32;
//  delete stats;

//  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceBColsBlocking_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedSeqReduceBColsBlocking64 = new FusedSpMMSpMMSeqReduceBColsBlocking(inSpMM, stats, ThreadsPerBlock, 64);
//  fusedSeqReduceBColsBlocking64->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedSeqReduceBColsBlocking->OutTensor->printDx();
//  auto fusedSeqReduceBColsBlocking64Stat = fusedSeqReduceBColsBlocking64->printStats();
//  delete fusedSeqReduceBColsBlocking64;
//  delete stats;
//
//  stats = new swiftware::benchmark::Stats("GPU_Fused_SeqReduceBColsBlocking_128","SpMMSpMM", numTrial,tp._matrix_name,numThread);
//  auto *fusedSeqReduceBColsBlocking128 = new FusedSpMMSpMMSeqReduceBColsBlocking(inSpMM,stats, ThreadsPerBlock, 128);
//  fusedSeqReduceBColsBlocking128->run();
//  //  std::cout << "FUSED: " << std::endl;
//  //  fusedSeqReduceBColsBlocking->OutTensor->printDx();
//  auto fusedSeqReduceBColsBlocking128Stat = fusedSeqReduceBColsBlocking128->printStats();
//  delete fusedSeqReduceBColsBlocking128;
//  delete stats;

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
  std::cout << unfusedSeqReduceRowCoarsened8Stat << spStat + tpStat + profStat << std::endl;
  std::cout << unfusedSeqReduceRowCoarsened16Stat << spStat + tpStat + profStat << std::endl;
  std::cout << unfusedSeqReduceRowCoarsened32Stat << spStat + tpStat + profStat << std::endl;
  std::cout << unfusedSeqReduceRowCoarsened64Stat << spStat + tpStat + profStat << std::endl;
  std::cout << unfusedSeqReduceRowCoarsenedStride8Stat << spStat + tpStat + profStat << std::endl;
  std::cout << unfusedSeqReduceRowCoarsenedStride16Stat << spStat + tpStat + profStat << std::endl;
  std::cout << unfusedSeqReduceRowCoarsenedStride32Stat << spStat + tpStat + profStat << std::endl;
  std::cout << unfusedSeqReduceRowCoarsenedStride64Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << unfusedCuSparseAlg2Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << unfusedCuSparseAlg3Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceRowBalanceStat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceRowBalanceReorderedStat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio8Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio16Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio32Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio64Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioResultPacked8Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioResultPacked16Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioResultPacked32Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioResultPacked64Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioP48Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioP416Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioP432Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioP464Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioStride8Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioStride16Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioStride32Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioStride64Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioMBC8Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioMBC16Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioMBC32Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioMBC64Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatio2L8Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatio2L16Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatio2L32Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatio2L64Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatioAtmoicFused8Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatioAtmoicFused16Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatioAtmoicFused32Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatioAtmoicFused64Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatioNoAtmoicFused8Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatioNoAtmoicFused16Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatioNoAtmoicFused32Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatioNoAtmoicFused64Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioNoSynch8Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioNoSynch16Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioNoSynch32Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatioNoSynch64Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatio_32_8Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatio_32_16Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatio_32_32Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedHighFusionRatio_32_64Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedRedHighFusionRatio_8Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedRedHighFusionRatio_16Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedRedHighFusionRatio_32Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedRedHighFusionRatio_64Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceWSMStat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceBColsBlocking4Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceBColsBlockingWSM4Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceBColsBlocking8Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceBColsBlockingWSM8Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceBColsBlocking16Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceBColsBlockingWSM16Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceBColsBlocking32Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceBColsBlockingWSM32Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceBColsBlocking64Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceBColsBlocking128Stat << spStat + tpStat + profStat << std::endl;


  delete inSpMM;

}
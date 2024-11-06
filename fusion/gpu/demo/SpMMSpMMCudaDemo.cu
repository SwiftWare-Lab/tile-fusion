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

  int numThread = sp._num_threads, numTrial = 5;
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
  //  fusedHighFusionRatio64->OutTensor->printDx();
  auto fusedHighFusionRatio64Stat = fusedHighFusionRatio64->printStats();
  delete fusedHighFusionRatio64;
  delete stats;


  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_32_HighFusionRatio_8","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatio_32_8 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, 32, ufThreadsPerBlock, 8);
  fusedHighFusionRatio_32_8->run();
  //  std::cout << "FUSED: " << std::endl;
  //  fusedHighFusionRatio_32_8->OutTensor->printDx();
  auto fusedHighFusionRatio_32_8Stat = fusedHighFusionRatio_32_8->printStats();
  delete fusedHighFusionRatio_32_8;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_32_HighFusionRatio_16","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatio_32_16 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, 32, ufThreadsPerBlock, 16);
  fusedHighFusionRatio_32_16->run();
  //  std::cout << "FUSED: " << std::endl;
  //  fusedHighFusionRatio_32_16->OutTensor->printDx();
  auto fusedHighFusionRatio_32_16Stat = fusedHighFusionRatio_32_16->printStats();
  delete fusedHighFusionRatio_32_16;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_32_HighFusionRatio_32","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatio_32_32 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, 32, ufThreadsPerBlock, 32);
  fusedHighFusionRatio_32_32->run();
  //  std::cout << "FUSED: " << std::endl;
  //  fusedHighFusionRatio_32_32->OutTensor->printDx();
  auto fusedHighFusionRatio_32_32Stat = fusedHighFusionRatio_32_32->printStats();
  delete fusedHighFusionRatio_32_32;
  delete stats;

  stats = new swiftware::benchmark::Stats("GPU_Fused_Reordered_32_HighFusionRatio_64","SpMMSpMM", numTrial,tp._matrix_name,numThread);
  auto *fusedHighFusionRatio_32_64 = new FusedSpMMSpMMHighFusionRatio(inSpMM,stats, 32, ufThreadsPerBlock, 64);
  fusedHighFusionRatio_32_64->run();
  //  std::cout << "FUSED: " << std::endl;
  //  fusedHighFusionRatio_32_64->OutTensor->printDx();
  auto fusedHighFusionRatio_32_64Stat = fusedHighFusionRatio_32_64->printStats();
  delete fusedHighFusionRatio_32_64;
  delete stats;

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
//  std::cout << unfusedCuSparseAlg2Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << unfusedCuSparseAlg3Stat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceRowBalanceStat << spStat + tpStat + profStat << std::endl;
//  std::cout << fusedSeqReduceRowBalanceReorderedStat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio8Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio16Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio32Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio64Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio_32_8Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio_32_16Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio_32_32Stat << spStat + tpStat + profStat << std::endl;
  std::cout << fusedHighFusionRatio_32_64Stat << spStat + tpStat + profStat << std::endl;
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
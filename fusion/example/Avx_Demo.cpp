//
// Created by mehdi on 6/16/23.
//

#include "SpMM_SpMM_Demo_Utils.h"
// #include "SpMM_SpMM_MKL_Demo_Utils.h"
#include "SpMM_SpMM_Avx_Demo_Utils.h"
#include "aggregation/def.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include <fstream>

using namespace sym_lib;

// A is MxK, C is KxN, B is LxM, and D is LxN; AC is MxN
int main(const int argc, const char *argv[]) {
  TestParameters tp;
  tp._order_method = SYM_ORDERING::NONE;
  ScheduleParameters sp;
  Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  CSC *aCSC = get_matrix_from_parameter(&tp);
  tp._dim1 = aCSC->m;
  tp._dim2 = aCSC->n;
  tp._nnz = aCSC->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);
  CSC *bCSC = sym_lib::copy_sparse(aCSC);
  auto *alCSC = make_half(aCSC->n, aCSC->p, aCSC->i, aCSC->x);
  std::vector<CSC *> orderedVec;
  if (tp._order_method != SYM_ORDERING::NONE) {
    // applies ordering here
    get_reorderd_matrix(alCSC, orderedVec);
    delete alCSC;
    alCSC = orderedVec[0];
  }

  int numThread = sp._num_threads, numTrial = 7;
  std::string expName = "SpMM_SpMM_Demo";
  auto *inSpMM =
      new TensorInputs<double>(aCSC->m, tp._b_cols, aCSC->n, bCSC->m, aCSC,
                               bCSC, numThread, numTrial, expName);

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo", "SpMM", 7,
                                          tp._matrix_name, numThread);
  auto *unfused = new SpMMSpMMUnFused(inSpMM, stats);
  unfused->run();
  inSpMM->CorrectSol = std::copy(
      unfused->OutTensor->Dx,
      unfused->OutTensor->Dx + unfused->OutTensor->M * unfused->OutTensor->N,
      inSpMM->CorrectMul);
  inSpMM->IsSolProvided = true;
  auto headerStat = unfused->printStatsHeader();
  auto baselineStat = unfused->printStats();
  delete unfused;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Unfused_Parallel", "SpMM",
                                          7, tp._matrix_name, numThread);
  auto *unfusedParallelImpl = new SpMMSpMMUnFusedParallel(inSpMM, stats);
  unfusedParallelImpl->run();
  auto unfusedParallelImplStat = unfusedParallelImpl->printStats();
  delete unfusedParallelImpl;
  delete stats;

  stats =
      new swiftware::benchmark::Stats("SpMM_SpMM_Avx_Sparse_Row_Vectorized",
                                      "SpMM", 7, tp._matrix_name, numThread);
  auto *onlySparseVectorizedImpl = new SpmmSpmmAvxFirstSparseRow(inSpMM, stats);
  onlySparseVectorizedImpl->run();
  auto onlySparseVectorizedImplStat = onlySparseVectorizedImpl->printStats();
  delete onlySparseVectorizedImpl;
  delete stats;

  stats =
      new swiftware::benchmark::Stats("SpMM_SpMM_Avx_Dense_Row_Vectorized",
                                      "SpMM", 7, tp._matrix_name, numThread);
  auto *onlyDenseVectorizedImpl = new SpmmSpmmAvxFirstDenseRow(inSpMM, stats);
  onlyDenseVectorizedImpl->run();
  auto onlyDenseVectorizedImplStat = onlyDenseVectorizedImpl->printStats();
  delete onlyDenseVectorizedImpl;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Avx_Combinational", "SpMM",
                                          7, tp._matrix_name, numThread);
  auto *majorlyDenseVectorizedImpl =
      new SpmmSpmmFirstDenseRowSecondSparseRow(inSpMM, stats);
  majorlyDenseVectorizedImpl->run();
  auto majorlyDenseVectorizedImplStat =
      majorlyDenseVectorizedImpl->printStats();
  delete majorlyDenseVectorizedImpl;
  delete stats;

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if (tp.print_header)
    std::cout << headerStat + spHeader + tpHeader << std::endl;
  std::cout << baselineStat << spStat + tpStat << std::endl;
  std::cout << unfusedParallelImplStat << spStat + tpStat << std::endl;
  std::cout << onlySparseVectorizedImplStat << spStat + tpStat << std::endl;
  std::cout << onlyDenseVectorizedImplStat << spStat + tpStat << std::endl;
  std::cout << majorlyDenseVectorizedImplStat << spStat + tpStat;

  delete aCSC;
  delete bCSC;
  delete alCSC;
  delete inSpMM;

  return 0;
}
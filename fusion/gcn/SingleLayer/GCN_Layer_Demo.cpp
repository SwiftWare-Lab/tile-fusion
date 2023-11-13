//
// Created by salehm32 on 12/10/23.
//
#include "GCN_Single_Layer_Demo_Utils.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
using namespace sym_lib;

/*
 * this demo is for comparing different implementations for one layer of GCN
 */
int main(const int argc, const char *argv[]) {
  TestParameters tp;
  tp._order_method = SYM_ORDERING::NONE;
  ScheduleParameters sp;
  Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  CSC *aCSC = get_matrix_from_parameter(&tp);
  Dense *features = get_feature_matrix_from_parameter(&tp, aCSC->m);
  CSC *aCSCFull = nullptr;
  if (aCSC->stype == -1 || aCSC->stype == 1) {
    aCSCFull = sym_lib::make_full(aCSC);
  } else {
    aCSCFull = sym_lib::copy_sparse(aCSC);
  }
  if (aCSC->m != aCSC->n) {
    return -1;
  }
  tp._dim1 = aCSCFull->m;
  tp._dim2 = aCSCFull->n;
  tp._nnz = aCSCFull->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);
  int embedDim = tp._embed_dim;
  int numClasses = 3;
  int numThread = sp._num_threads;
  int tileSize = sp.TileN;
  double *layer1Weight = generateRandomDenseMatrix(features->col, embedDim);
  double *layer2Weight = generateRandomDenseMatrix(embedDim, numClasses);

  int numOfSamples = std::ceil(tp._sampling_ratio * tp._dim1);
  GnnTensorInputs *inputs = new GnnTensorInputs(
      layer1Weight, layer2Weight, features, aCSCFull, aCSCFull->m, embedDim,
      numOfSamples, numThread, 7, "GCN_Demo");

  /*
   * The method that iterate over rows of the adjacency matrix and by doing the
   * corresponding GeMV to each nonzero, calculates the output
   */
  stats = new swiftware::benchmark::Stats("GCN_SingleLayerFused", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerFused *gcnSingleLayerFused =
      new GCNSingleLayerFused(inputs, stats);
  gcnSingleLayerFused->run();
  inputs->CorrectSol =
      new double[inputs->AdjacencyMatrix->m * inputs->EmbedDim];
  std::copy(gcnSingleLayerFused->OutTensor->FirstLayerOutput,
            gcnSingleLayerFused->OutTensor->FirstLayerOutput +
                inputs->AdjacencyMatrix->m * inputs->EmbedDim,
            inputs->CorrectSol);
  auto headerStat = gcnSingleLayerFused->printStatsHeader();
  auto gcnSequentialFusedLayerStat = gcnSingleLayerFused->printStats();
  delete stats;
  delete gcnSingleLayerFused;

  /*
   * The method that iterate over rows of the adjacency matrix in parallel and
   * by doing the corresponding GeMV to each nonzero, calculates the output
   */
  stats = new swiftware::benchmark::Stats("GCN_SingleLayerFusedParallel", "GCN",
                                          7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerFusedParallel *gcnSingleLayerFusedParallel =
      new GCNSingleLayerFusedParallel(inputs, stats);
  gcnSingleLayerFusedParallel->run();
  auto gcnSingleLayerFusedParallelStat =
      gcnSingleLayerFusedParallel->printStats();
  delete stats;
  delete gcnSingleLayerFusedParallel;

  /*
   * Method iterate over tile of rows, and by doing the corresponding GeMM to
   * each tile, then doing SpMM for midway result, calculates the output.
   */
  stats = new swiftware::benchmark::Stats("GCN_SingleLayerTiledFused", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerTiledFused *gcnSingleLayerTiledFused =
      new GCNSingleLayerTiledFused(inputs, stats, tileSize);
  gcnSingleLayerTiledFused->run();
  auto gcnTiledFusedSingleLayerStat = gcnSingleLayerTiledFused->printStats();
  delete stats;
  delete gcnSingleLayerTiledFused;

  stats = new swiftware::benchmark::Stats("GCN_SingleLayerTiledFusedParallel",
                                          "GCN", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerTiledFusedParallel *gcnSingleLayerTiledFusedParallel =
      new GCNSingleLayerTiledFusedParallel(inputs, stats, tileSize);
  gcnSingleLayerTiledFusedParallel->run();
  auto gcnSingleLayerTiledFusedParallelStat =
      gcnSingleLayerTiledFusedParallel->printStats();
  delete stats;
  delete gcnSingleLayerTiledFusedParallel;

  /*
   * Method that calculates the output by doing a GeMM and then an SpMM to the
   * input.
   */
  stats = new swiftware::benchmark::Stats("GCN_SingleLayerMKL", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerMKL *gcnSingleLayerMkl = new GCNSingleLayerMKL(inputs, stats);
  gcnSingleLayerMkl->run();
  auto gcnOneLayerMKLStat = gcnSingleLayerMkl->printStats();
  delete stats;
  delete gcnSingleLayerMkl;

  /*
   * Method that iterates over columns of Adjacency matrix and by doing the
   * corresponding GeMV to each nonzero, calculates the partial products of that
   * column's non-zeros.
   */
  stats = new swiftware::benchmark::Stats("GCN_SingleLayerFusedCSC", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerFusedCSC *gcnSingleLayerFusedCsc =
      new GCNSingleLayerFusedCSC(inputs, stats);
  gcnSingleLayerFusedCsc->run();
  auto gcnSingleLayerFusedCscStat = gcnSingleLayerFusedCsc->printStats();
  delete stats;
  delete gcnSingleLayerFusedCsc;

  /* Method that iterates over tiles of columns of Adjacency matrix in
   * parallel(First the odd tiles and after that the even tiles) and by doing
   * the corresponding GEMM to each tile, then doing SpMM for midway result,
   * calculates the output
   * BANDED-SPECIFIC FOR NOW
   */
  stats = new swiftware::benchmark::Stats("GCN_SingleLayerTiledFusedCSCParallel",
                                          "GCN", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerTiledFusedCSCParallel *gcnSingleLayerFusedCscParallel =
      new GCNSingleLayerTiledFusedCSCParallel(inputs, stats, tileSize);
  gcnSingleLayerFusedCscParallel->run();
  auto gcnSingleLayerFusedCscParallelStat =
      gcnSingleLayerFusedCscParallel->printStats();
  delete stats;
  delete gcnSingleLayerFusedCscParallel;

  stats = new swiftware::benchmark::Stats("GCN_SingleLayerTiledFusedCSCCombined", "GCN",
                                          7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerTiledFusedCSCCombined *gcnSingleLayerTiledFusedCscCombined =
      new GCNSingleLayerTiledFusedCSCCombined(inputs, stats, tileSize);
  gcnSingleLayerTiledFusedCscCombined->run();
  auto gcnSingleLayerTiledFusedCSCCombinedStat =
      gcnSingleLayerTiledFusedCscCombined->printStats();
  delete stats;
  delete gcnSingleLayerTiledFusedCscCombined;

  /*
   * Method that iterates over tiles of columns of Adjacency matrix and by doing
   * corresponding GeMM to each tile, then doing SpMM for midway result,
   * calculates the output.
   */
  stats = new swiftware::benchmark::Stats("GCN_SingleLayerTiledFusedCSC", "GCN",
                                          7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerTiledFusedCSC *gcnSingleLayerTiledFusedCsc =
      new GCNSingleLayerTiledFusedCSC(inputs, stats, tileSize);
  gcnSingleLayerTiledFusedCsc->run();
  auto gcnSingleLayerTiledFusedCscStat =
      gcnSingleLayerTiledFusedCsc->printStats();
  delete stats;
  delete gcnSingleLayerTiledFusedCsc;

#ifdef __AVX2__
  /*
   * Method that works like `GCN_SingleLayerFusedCSC` but use vectorization for
   * computing partial products
   */
//  stats = new swiftware::benchmark::Stats("GCN_SingleLayerFusedCSCVectorized",
//                                          "GCN", 7, tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  GCNSingleLayerFusedCSCParallelVectorized *gcnSingleLayerFusedCscVectorized =
//      new GCNSingleLayerFusedCSCParallelVectorized(inputs, stats);
//  gcnSingleLayerFusedCscVectorized->run();
//  auto gcnSingleLayerFusedCscVectorizedStat =
//      gcnSingleLayerFusedCscVectorized->printStats();
//  delete stats;
//  delete gcnSingleLayerFusedCscVectorized;

  /*
   * Method that works like `GCN_SingleLayerTiledFusedCSC` but use vectorization
   * for computing partial products
   */
//  stats =
//      new swiftware::benchmark::Stats("GCN_SingleLayerTiledFusedCSCVectorized",
//                                      "GCN", 7, tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  GCNSingleLayerTiledFusedCSCVectorized *gcnSingleLayerTiledFusedCscVectorized =
//      new GCNSingleLayerTiledFusedCSCVectorized(inputs, stats, tileSize);
//  gcnSingleLayerTiledFusedCscVectorized->run();
//  auto gcnSingleLayerTiledFusedCscVectorizedStat =
//      gcnSingleLayerTiledFusedCscVectorized->printStats();
//  delete stats;
//  delete gcnSingleLayerTiledFusedCscVectorized;
#endif

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if (tp.print_header)
    std::cout << headerStat + spHeader + tpHeader << std::endl;
  std::cout << gcnSequentialFusedLayerStat << spStat + tpStat << std::endl;
  std::cout << gcnOneLayerMKLStat << spStat + tpStat << std::endl;
  std::cout << gcnTiledFusedSingleLayerStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerTiledFusedParallelStat << spStat + tpStat
            << std::endl;
  std::cout << gcnSingleLayerFusedCscStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerFusedCscParallelStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerTiledFusedCscStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerFusedParallelStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerTiledFusedCSCCombinedStat << spStat + tpStat
            << std::endl;
#ifdef __AVX2__
//  std::cout << gcnSingleLayerFusedCscVectorizedStat << spStat + tpStat
//            << std::endl;
//  std::cout << gcnSingleLayerTiledFusedCscVectorizedStat << spStat + tpStat
//            << std::endl;
#endif

  delete[] inputs->CorrectSol;
  delete inputs;
  delete aCSC;
  delete aCSCFull;
}
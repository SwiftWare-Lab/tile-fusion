//
// Created by salehm32 on 12/10/23.
//
#include "../Inspection/GraphColoring.h"
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
  Dense *features = get_dense_matrix_from_parameter(&tp, aCSC->m, tp._b_cols,
                                                    tp._feature_matrix_path);
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
  int kTileSize = 8;
  Dense *layer1Weight = get_dense_matrix_from_parameter(
      &tp, tp._embed_dim, tp._b_cols, tp._weight1_matrix_path);
  Dense *layer2Weight = get_dense_matrix_from_parameter(
      &tp, tp._embed_dim, tp._embed_dim, tp._weight2_matrix_path);

  int numOfSamples = std::ceil(tp._sampling_ratio * tp._dim1);
  GnnTensorInputs *inputs = new GnnTensorInputs(
      layer1Weight, layer2Weight, features, aCSCFull, aCSCFull->m, embedDim,
      numOfSamples, numThread, 7, "GCN_Demo");

  /*
   * The method that iterate over rows of the adjacency matrix and by doing the
   * corresponding GeMV to each nonzero, calculates the output
   */
  //  stats = new swiftware::benchmark::Stats("GCN_SingleLayerFused", "GCN", 7,
  //                                          tp._matrix_name, numThread);
  //  stats->OtherStats["PackingType"] = {Separated};
  //  GCNSingleLayerFused *gcnSingleLayerFused =
  //      new GCNSingleLayerFused(inputs, stats);
  //  gcnSingleLayerFused->run();
  //  delete stats;
  //  delete gcnSingleLayerFused;

  /*
   * The method that iterate over rows of the adjacency matrix in parallel and
   * by doing the corresponding GeMV to each nonzero, calculates the output
   */
  //  stats = new swiftware::benchmark::Stats("GCN_SingleLayerFusedParallel",
  //  "GCN",
  //                                          7, tp._matrix_name, numThread);
  //  stats->OtherStats["PackingType"] = {Separated};
  //  GCNSingleLayerFusedParallel *gcnSingleLayerFusedParallel =
  //      new GCNSingleLayerFusedParallel(inputs, stats);
  //  gcnSingleLayerFusedParallel->run();
  //  auto gcnSingleLayerFusedParallelStat =
  //      gcnSingleLayerFusedParallel->printStats();
  //  delete stats;
  //  delete gcnSingleLayerFusedParallel;

  /*
   * Method iterate over tile of rows, and by doing the corresponding GeMM to
   * each tile, then doing SpMM for midway result, calculates the output.
   */
  //  stats = new swiftware::benchmark::Stats("GCN_SingleLayerTiledFused",
  //  "GCN", 7,
  //                                          tp._matrix_name, numThread);
  //  stats->OtherStats["PackingType"] = {Separated};
  //  GCNSingleLayerTiledFused *gcnSingleLayerTiledFused =
  //      new GCNSingleLayerTiledFused(inputs, stats, tileSize);
  //  gcnSingleLayerTiledFused->run();
  //  auto gcnTiledFusedSingleLayerStat =
  //  gcnSingleLayerTiledFused->printStats(); delete stats; delete
  //  gcnSingleLayerTiledFused;

  //  stats = new
  //  swiftware::benchmark::Stats("GCN_SingleLayerTiledFusedParallel",
  //                                          "GCN", 7, tp._matrix_name,
  //                                          numThread);
  //  stats->OtherStats["PackingType"] = {Separated};
  //  GCNSingleLayerTiledFusedParallel *gcnSingleLayerTiledFusedParallel =
  //      new GCNSingleLayerTiledFusedParallel(inputs, stats, tileSize);
  //  gcnSingleLayerTiledFusedParallel->run();
  //  auto gcnSingleLayerTiledFusedParallelStat =
  //      gcnSingleLayerTiledFusedParallel->printStats();
  //  delete stats;
  //  delete gcnSingleLayerTiledFusedParallel;

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
  inputs->CorrectSol =
      new double[inputs->AdjacencyMatrix->m * inputs->Weight1->col];
  std::copy(gcnSingleLayerMkl->OutTensor->FirstLayerOutput,
            gcnSingleLayerMkl->OutTensor->FirstLayerOutput +
                inputs->AdjacencyMatrix->m * inputs->Weight1->col,
            inputs->CorrectSol);
  auto headerStat = gcnSingleLayerMkl->printStatsHeader();
  delete stats;
  delete gcnSingleLayerMkl;

  stats = new swiftware::benchmark::Stats("GCN_SingleLayerUnfusedCSC", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerUnfusedCSC *gcnSingleLayerUnfusedCsc =
      new GCNSingleLayerUnfusedCSC(inputs, stats);
  gcnSingleLayerUnfusedCsc->run();
  auto gcnSingleLayerUnfusedCscStat = gcnSingleLayerUnfusedCsc->printStats();
  delete stats;
  delete gcnSingleLayerUnfusedCsc;

  /*
   * Method that iterates over columns of Adjacency matrix and by doing the
   * corresponding GeMV to each nonzero, calculates the partial products of that
   * column's non-zeros.
   */
  //  stats = new swiftware::benchmark::Stats("GCN_SingleLayerFusedCSC", "GCN",
  //  7,
  //                                          tp._matrix_name, numThread);
  //  stats->OtherStats["PackingType"] = {Separated};
  //  GCNSingleLayerFusedCSC *gcnSingleLayerFusedCsc =
  //      new GCNSingleLayerFusedCSC(inputs, stats);
  //  gcnSingleLayerFusedCsc->run();
  //  auto gcnSingleLayerFusedCscStat = gcnSingleLayerFusedCsc->printStats();
  //  delete stats;
  //  delete gcnSingleLayerFusedCsc;

  /* generating conflict graph once per tile size
   * this is not calculated in inspection time for now.
   */
  DsaturColoringForConflictGraph *dsaturColoring =
      new DsaturColoringForConflictGraph();
  DsaturColoringForConflictGraphWithKTiling *dsaturColoringWithKTiling =
      new DsaturColoringForConflictGraphWithKTiling();
  std::map<int, std::vector<int>> colorToTiles =
      dsaturColoring->generateGraphColoringForConflictGraphOf(aCSCFull,
                                                              tileSize, false);
  std::map<int, std::vector<int>> colorToTilesForKTiling =
      dsaturColoringWithKTiling->generateGraphColoringForConflictGraphOf(
          aCSCFull, tileSize, inputs->Weight1->row, kTileSize, false);

  /*
   * Method that iterates over tiles of columns of Adjacency matrix in
   * parallel(using scheduling based on conflict graph coloring) and by doing
   * the corresponding GeMM to each tile, then doing SpMM on midway result,
   * calculates the output
   */
  stats =
      new swiftware::benchmark::Stats("GCN_SingleLayerTiledFusedCSCParallel",
                                      "GCN", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerTiledFusedCSCParallel *gcnSingleLayerFusedCscParallel =
      new GCNSingleLayerTiledFusedCSCParallel(inputs, stats, tileSize,
                                              colorToTiles);
  gcnSingleLayerFusedCscParallel->run();
  auto gcnSingleLayerFusedCscParallelStat =
      gcnSingleLayerFusedCscParallel->printStats();
  delete stats;
  delete gcnSingleLayerFusedCscParallel;

  /*
   * Method that iterates over tiles of columns of Adjacency matrix in
   * parallel(using scheduling based on conflict graph coloring) and by doing
   * the corresponding GeMMs to each tile, then doing SpMM on midway result,
   * calculates the output
   * This method also divides weight matrix to kTiles. kTiles in this method
   * are considered in the conflict graph.
   */
  stats = new swiftware::benchmark::Stats(
      "GCN_SingleLayerTiledFusedCSCParallelWithSchedulingKTiling", "GCN", 7,
      tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerTiledFusedCSCParallelWithSchedulingKTiling
      *gcnSingleLayerFusedCscParallelWithSchedulingKTiling =
          new GCNSingleLayerTiledFusedCSCParallelWithSchedulingKTiling(
              inputs, stats, tileSize, colorToTilesForKTiling, kTileSize);
  gcnSingleLayerFusedCscParallelWithSchedulingKTiling->run();
  auto gcnSingleLayerFusedCscParallelWithSchedulingKTilingStat =
      gcnSingleLayerFusedCscParallelWithSchedulingKTiling->printStats();
  delete stats;
  delete gcnSingleLayerFusedCscParallelWithSchedulingKTiling;

  /*
   * Method that iterates over tiles of columns of Adjacency matrix in
   * parallel(using scheduling based on conflict graph coloring) and by doing
   * the corresponding GeMMs to each tile, then doing SpMM on midway result,
   * calculates the output
   * This method also divides weight matrix to kTiles. kTiles in this method
   * are considered in executer only.
   */
  stats = new swiftware::benchmark::Stats(
      "GCN_SingleLayerTiledFusedCSCParallelWithKTiling", "GCN", 7,
      tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerTiledFusedCSCParallelWithKTiling
      *gcnSingleLayerFusedCscParallelWithKTiling =
          new GCNSingleLayerTiledFusedCSCParallelWithKTiling(
              inputs, stats, tileSize, colorToTiles, kTileSize);
  gcnSingleLayerFusedCscParallelWithKTiling->run();
  auto gcnSingleLayerFusedCscParallelWithKTilingStat =
      gcnSingleLayerFusedCscParallelWithKTiling->printStats();
  delete stats;
  delete gcnSingleLayerFusedCscParallelWithKTiling;

  int minWorkloads[8] = {4, 6, 8, 10, 12, 14, 16, 18};

  // tuning of min workload size is done here so that conflict graph is once
  // computed per tile size
  std::vector<std::string> combinedStats;
  for (int minWorkload : minWorkloads) {
    /*
     * Method that iterates over tiles of columns of Adjacency matrix in
     * parallel(using scheduling based on conflict graph coloring) and by doing
     * the corresponding GeMM to each tile, then doing SpMM on midway result,
     * calculates the output.
     * some tiles are ran in parallel region and some are ran in sequential region,
     * The inspector prunes tiles that are in smaller workloads and merges them
     * so that new big tiles are ran in sequential region but with parallelized GeMMs.
     */
    stats =
        new swiftware::benchmark::Stats("GCN_SingleLayerTiledFusedCSCCombined",
                                        "GCN", 7, tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Separated};
    GCNSingleLayerTiledFusedCSCCombined *gcnSingleLayerTiledFusedCscCombined =
        new GCNSingleLayerTiledFusedCSCCombined(inputs, stats, tileSize,
                                                minWorkload, colorToTiles);
    gcnSingleLayerTiledFusedCscCombined->run();
    stats->OtherStats["Min Workload Size"] = {double(minWorkload)};
    combinedStats.push_back(gcnSingleLayerTiledFusedCscCombined->printStats());
    delete stats;
    delete gcnSingleLayerTiledFusedCscCombined;

#ifdef __AVX2__
    /*
     * Method that iterates over tiles of columns of Adjacency matrix in
     * parallel(using scheduling based on conflict graph coloring) and by doing
     * the corresponding GeMM to each tile, then doing SpMM on midway result,
     * calculates the output.
     * some tiles are ran in parallel region and some are ran in sequential region,
     * The inspector prunes tiles that are in smaller workloads and merges them
     * so that new big tiles are ran in sequential region but with parallelized GeMMs.
     * This implementation also includes unrolling and vectorization.
     */
    stats = new swiftware::benchmark::Stats(
        "GCN_SingleLayerTiledFusedCSCCombinedVectorized", "GCN", 7,
        tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Separated};
    GCNSingleLayerTiledFusedCSCCombinedVectorized
        *gcnSingleLayerTiledFusedCscCombinedVectorized =
            new GCNSingleLayerTiledFusedCSCCombinedVectorized(
                inputs, stats, tileSize, minWorkload, colorToTiles);
    gcnSingleLayerTiledFusedCscCombinedVectorized->run();
    stats->OtherStats["Min Workload Size"] = {double(minWorkload)};
    combinedStats.push_back(
        gcnSingleLayerTiledFusedCscCombinedVectorized->printStats());
    delete stats;
    delete gcnSingleLayerTiledFusedCscCombinedVectorized;

#endif
    stats = new swiftware::benchmark::Stats(
        "GCN_SingleLayerTiledFusedCSCCombinedWithKTiling", "GCN", 7,
        tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Separated};
    GCNSingleLayerTiledFusedCSCCombinedWithKTiling
        *gcnSingleLayerTiledFusedCscCombinedWithKTiling =
            new GCNSingleLayerTiledFusedCSCCombinedWithKTiling(
                inputs, stats, tileSize, minWorkload, colorToTiles, kTileSize);
    gcnSingleLayerTiledFusedCscCombinedWithKTiling->run();
    stats->OtherStats["Min Workload Size"] = {double(minWorkload)};
    combinedStats.push_back(
        gcnSingleLayerTiledFusedCscCombinedWithKTiling->printStats());
    delete stats;
    delete gcnSingleLayerTiledFusedCscCombinedWithKTiling;
  }

  /*
   * Method that iterates over tiles of columns of Adjacency matrix and by doing
   * corresponding GeMM to each tile, then doing SpMM for midway result,
   * calculates the output.
   */
  //  stats = new swiftware::benchmark::Stats("GCN_SingleLayerTiledFusedCSC",
  //  "GCN",
  //                                          7, tp._matrix_name, numThread);
  //  stats->OtherStats["PackingType"] = {Separated};
  //  GCNSingleLayerTiledFusedCSC *gcnSingleLayerTiledFusedCsc =
  //      new GCNSingleLayerTiledFusedCSC(inputs, stats, tileSize);
  //  gcnSingleLayerTiledFusedCsc->run();
  //  auto gcnSingleLayerTiledFusedCscStat =
  //      gcnSingleLayerTiledFusedCsc->printStats();
  //  delete stats;
  //  delete gcnSingleLayerTiledFusedCsc;

#ifdef __AVX2__
  /*
   * Method that works like `GCN_SingleLayerFusedCSC` but use vectorization for
   * computing partial products
   */
  //  stats = new
  //  swiftware::benchmark::Stats("GCN_SingleLayerFusedCSCVectorized",
  //                                          "GCN", 7, tp._matrix_name,
  //                                          numThread);
  //  stats->OtherStats["PackingType"] = {Separated};
  //  GCNSingleLayerFusedCSCParallelVectorized *gcnSingleLayerFusedCscVectorized
  //  =
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
//      new
//      swiftware::benchmark::Stats("GCN_SingleLayerTiledFusedCSCVectorized",
//                                      "GCN", 7, tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  GCNSingleLayerTiledFusedCSCVectorized *gcnSingleLayerTiledFusedCscVectorized
//  =
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

  std::cout << gcnOneLayerMKLStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerUnfusedCscStat << spStat + tpStat << std::endl;
  //  std::cout << gcnTiledFusedSingleLayerStat << spStat + tpStat << std::endl;
  //  std::cout << gcnSingleLayerTiledFusedParallelStat << spStat + tpStat
  //            << std::endl;
  //  std::cout << gcnSingleLayerFusedCscStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerFusedCscParallelStat << spStat + tpStat
            << std::endl;
  std::cout << gcnSingleLayerFusedCscParallelWithSchedulingKTilingStat
            << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerFusedCscParallelWithKTilingStat << spStat + tpStat
            << std::endl;
  //  std::cout << gcnSingleLayerTiledFusedCscStat << spStat + tpStat <<
  //  std::endl; std::cout << gcnSingleLayerFusedParallelStat << spStat + tpStat
  //  << std::endl;
  for (auto stat : combinedStats) {
    std::cout << stat << spStat + tpStat << std::endl;
  }

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
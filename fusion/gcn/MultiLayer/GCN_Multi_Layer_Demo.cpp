//
// Created by mehdi on 6/30/23.
//
#include "GCN_Multi_Layer_Demo_Utils.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"

using namespace sym_lib;

/*
 * this Demo is for comparing different fusion levels for a multi_layer GCN,
 * there are some sampling experiments in this Demo you can set the experiment
 * using the following parameter: --experiment_name=GCNWithDifferentFusionLevels
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
  int hiddenDim = tp._embed_dim;
  int numThread = sp._num_threads;
  int tileSize = sp.TileN;
  double *layer1Weight = generateRandomDenseMatrix(features->col, hiddenDim);
  double *layer2Weight = generateRandomDenseMatrix(hiddenDim, hiddenDim);

  int numOfSamples = std::ceil(tp._sampling_ratio * tp._dim1);
  GnnTensorInputs *inputs = new GnnTensorInputs(
      layer1Weight, layer2Weight, features, aCSCFull, aCSCFull->m, hiddenDim,
      numOfSamples, numThread, 1, "GCN_Demo");

  /*
   * The method that iterate over rows of the adjacency matrix and by doing the
   * corresponding GeMV to each nonzero, calculates the output for each layer.
   */
  stats = new swiftware::benchmark::Stats("GCN_IntraFused_Demo", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNIntraFusedSequential *gcnGnn = new GCNIntraFusedSequential(inputs, stats);
  gcnGnn->run();
  inputs->CorrectSol =
      new double[inputs->AdjacencyMatrix->m * inputs->EmbedDim];
  std::copy(gcnGnn->OutTensor->SecondLayerOutput,
            gcnGnn->OutTensor->SecondLayerOutput +
                inputs->AdjacencyMatrix->m * inputs->EmbedDim,
            inputs->CorrectSol);
  auto headerStat = gcnGnn->printStatsHeader();
  auto gcnStat = gcnGnn->printStats();
  delete gcnGnn;
  delete stats;

  stats = new swiftware::benchmark::Stats("GCN_IntraFusedParallel_Demo", "GCN",
                                          7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNIntraFusedParallel *gcnParallel = new GCNIntraFusedParallel(inputs, stats);
  gcnParallel->run();
  auto gcnParallelStat = gcnParallel->printStats();
  delete gcnParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("GCN_IntraUnfused_Demo", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNUnfused *gcnIntraUnfusedMKL = new GCNUnfused(inputs, stats);
  gcnIntraUnfusedMKL->run();
  auto gcnIntraUnfusedMKLStat = gcnIntraUnfusedMKL->printStats();
  delete gcnIntraUnfusedMKL;
  delete stats;
  /*
   * Method that iterates over row tiles of Adjacency matrix and by doing the
   * corresponding GeMM for each tile, then doing SpMM for midway result,
   * calculates the results.
   * BANDED SPECIFIC FOR NOW
   */
  stats = new swiftware::benchmark::Stats("GCN_IntraTiledFused_Demo", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNIntraLayerTiledFused *gcnIntraTiledFused =
      new GCNIntraLayerTiledFused(inputs, stats, tileSize);
  gcnIntraTiledFused->run();
  auto gcnIntraTiledFusedStats = gcnIntraTiledFused->printStats();
  delete gcnIntraTiledFused;
  delete stats;


  /* Method that iterates over columns of Adjacency matrix and by doing the
   * corresponding GeMV to each nonzero, calculates the partial products
   * of that column's non-zeros for each layer.
   */
  stats = new swiftware::benchmark::Stats("GCN_IntraFusedCSCSequential_Demo",
                                          "GCN", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNIntraFusedUsingCSCSequential *gcnIntraFusedCSCSequential =
      new GCNIntraFusedUsingCSCSequential(inputs, stats);
  gcnIntraFusedCSCSequential->run();
  auto gcnIFCSStats = gcnIntraFusedCSCSequential->printStats();
  delete gcnIntraFusedCSCSequential;
  delete stats;


  /* Method that iterates over tiles of columns of Adjacency matrix and by
   * doing corresponding GeMM to each tile, then doing SpMM for midway result,
   * calculates the output for each layer.
   */
  stats = new swiftware::benchmark::Stats("GCN_IntraTiledFusedCSC_Demo", "GCN",
                                          7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  GCNIntraTiledFusedUsingCSC *gcnIntraTiledFusedCSC =
      new GCNIntraTiledFusedUsingCSC(inputs, stats, tileSize);
  gcnIntraTiledFusedCSC->run();
  auto gcnIntraTiledFusedCSCStats = gcnIntraTiledFusedCSC->printStats();
  delete gcnIntraTiledFusedCSC;
  delete stats;


  /*
   * Method that iterates over tiles of columns of Adjacency matrix and by
   * doing corresponding GeMM to each tile, then doing SpMM for midway result,
   * calculates the output for each layer. Layers are scheduled to be run
   * interleaved in this version.
   * Inspection-Executor
   */
  stats = new swiftware::benchmark::Stats("GCN_AllTiledFusedCSC_Demo", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  GCNAllTiledFusedCSC *gcnAllFusedCSC =
      new GCNAllTiledFusedCSC(inputs, stats, tileSize);
  gcnAllFusedCSC->run();
  auto gcnAllFusedCSCStats = gcnAllFusedCSC->printStats();
  delete gcnAllFusedCSC;
  delete stats;

  /*
   * This Mehtod combines the fusion method used for SpMM-SpMM with the
   * GCN_IntraFused method. Inspection-Executor
   */
  stats = new swiftware::benchmark::Stats("GCN_AllFused_Demo", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  GCNAllFusedParallel *gcnAllFusedParallel =
      new GCNAllFusedParallel(inputs, stats, sp);
  gcnAllFusedParallel->run();
  auto gcnAllFusedParallelStats = gcnAllFusedParallel->printStats();
  delete gcnAllFusedParallel;
  delete stats;

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if (tp.print_header)
    std::cout << headerStat + spHeader + tpHeader << std::endl;
  std::cout << gcnStat << spStat + tpStat << std::endl;
  std::cout << gcnParallelStat << spStat + tpStat << std::endl;
  std::cout << gcnIntraUnfusedMKLStat << spStat + tpStat << std::endl;
  std::cout << gcnIntraTiledFusedStats << spStat + tpStat << std::endl;
  std::cout << gcnIFCSStats << spStat + tpStat << std::endl;
  std::cout << gcnIntraTiledFusedCSCStats << spStat + tpStat << std::endl;
  std::cout << gcnAllFusedCSCStats << spStat + tpStat << std::endl;
  std::cout << gcnAllFusedParallelStats << spStat + tpStat << std::endl;

  delete inputs;
  delete aCSC;
  delete aCSCFull;
}

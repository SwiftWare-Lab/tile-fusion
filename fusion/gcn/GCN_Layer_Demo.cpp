//
// Created by salehm32 on 12/10/23.
//
#include "GCN_Layer_Demo_Utils.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
using namespace sym_lib;
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
      numClasses, numOfSamples, numThread, 7, "GCN_Demo");

  stats = new swiftware::benchmark::Stats("GCN_SingleLayerFused", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSequential *gcnSingleLayerFused = new GCNSingleLayerFused(inputs, stats);
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

  stats = new swiftware::benchmark::Stats("GCN_SingleLayerTiledFused", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerTiledFused *gcnSingleLayerTiledFused = new GCNSingleLayerTiledFused(inputs, stats, tileSize);
  gcnSingleLayerTiledFused->run();
  auto gcnTiledFusedSingleLayerStat = gcnSingleLayerTiledFused->printStats();
  delete stats;
  delete gcnSingleLayerTiledFused;

  stats = new swiftware::benchmark::Stats("GCN_SingleLayerMKL", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerMKL *gcnSingleLayerMkl = new GCNSingleLayerMKL(inputs, stats);
  gcnSingleLayerMkl->run();
  auto gcnOneLayerMKLStat = gcnSingleLayerMkl->printStats();
  delete stats;
  delete gcnSingleLayerMkl;

  stats = new swiftware::benchmark::Stats("GCN_SingleLayerFusedCSC", "GCN", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerFusedCSC* gcnSingleLayerFusedCsc = new GCNSingleLayerFusedCSC(inputs, stats);
  gcnSingleLayerFusedCsc->run();
  auto gcnSingleLayerFusedCscStat = gcnSingleLayerFusedCsc->printStats();
  delete stats;
  delete gcnSingleLayerFusedCsc;

  stats = new swiftware::benchmark::Stats("GCN_SingleLayerFusedCSCParallel", "GCN", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerFusedCSCParallel* gcnSingleLayerFusedCscParallel = new GCNSingleLayerFusedCSCParallel(inputs, stats);
  gcnSingleLayerFusedCscParallel->run();
  auto gcnSingleLayerFusedCscParallelStat = gcnSingleLayerFusedCscParallel->printStats();
  delete stats;
  delete gcnSingleLayerFusedCscParallel;


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
  std::cout << gcnSingleLayerFusedCscStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerFusedCscParallelStat << spStat + tpStat << std::endl;

  delete inputs;
  delete aCSC;
  delete aCSCFull;
}
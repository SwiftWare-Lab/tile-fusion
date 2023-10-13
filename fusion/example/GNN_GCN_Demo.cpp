//
// Created by mehdi on 6/30/23.
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
  int hiddenDim = 50;
  int numClasses = 3;
  int numThread = sp._num_threads;
  int tileSize = sp.TileN;
  double *layer1Weight = generateRandomDenseMatrix(features->col, hiddenDim);
  double *layer2Weight = generateRandomDenseMatrix(hiddenDim, numClasses);

  int numOfSamples = std::ceil(tp._sampling_ratio * tp._dim1);
  GnnTensorInputs *inputs = new GnnTensorInputs(
      layer1Weight, layer2Weight, features, aCSCFull, aCSCFull->m, hiddenDim,
      numClasses, numOfSamples, numThread, 1, "GCN_Demo");

  stats = new swiftware::benchmark::Stats("GCN_Sequential_Demo", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSequential *gcnGnn = new GCNSequential(inputs, stats);
  gcnGnn->run();
  inputs->CorrectSol =
      new double[inputs->AdjacencyMatrix->m * inputs->NumOfClasses];
  std::copy(gcnGnn->OutTensor->SecondLayerOutput,
            gcnGnn->OutTensor->SecondLayerOutput +
                inputs->AdjacencyMatrix->m * inputs->NumOfClasses,
            inputs->CorrectSol);

  //  for (int i = 0; i < inputs->NumOfNodes; i++){
  //    for (int j = 0; j < inputs->NumOfClasses; j++){
  //      std::cout <<
  //      gcnGnn->OutTensor->SecondLayerOutput[i*inputs->NumOfClasses+j] << " ";
  //    }
  //    std::cout << std::endl;
  //  }
  auto headerStat = gcnGnn->printStatsHeader();
  auto gcnStat = gcnGnn->printStats();
  delete gcnGnn;
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

  stats = new swiftware::benchmark::Stats("GCN_Parallel_Demo", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNParallel *gcnParallel = new GCNParallel(inputs, stats);
  gcnParallel->run();
  auto gcnParallelStat = gcnParallel->printStats();
  delete gcnParallel;
  delete stats;
  std::cout << gcnParallelStat << spStat + tpStat << std::endl;

  if (tp.expariment_name == "GCNFusedParallel") {
    stats = new swiftware::benchmark::Stats("GCN_Fused_Demo", "GCN", 7,
                                            tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Interleaved};
    GCNFused *gcnFused = new GCNFused(inputs, stats, sp);
    gcnFused->run();
    auto gcnFusedStat = gcnFused->printStats();
    delete gcnFused;
    delete stats;

    std::cout << gcnFusedStat << spStat + tpStat << std::endl;
    stats = new swiftware::benchmark::Stats(
        "GCN_FusedParallelWithOmittingEmptyRows_Demo", "GCN", 7,
        tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Interleaved};
    GCNFusedParallelWithOmittingEmptyRows
        *gcnFusedParallelWithOmittingEmptyRows =
            new GCNFusedParallelWithOmittingEmptyRows(inputs, stats, sp);
    gcnFusedParallelWithOmittingEmptyRows->run();
    auto gcnFusedPWOERStat =
        gcnFusedParallelWithOmittingEmptyRows->printStats();
    delete gcnFusedParallelWithOmittingEmptyRows;
    delete stats;
    std::cout << gcnFusedPWOERStat << spStat + tpStat << std::endl;
  }

  if (tp.expariment_name == "GCNFusedBandedSpecific") {
    stats =
        new swiftware::benchmark::Stats("GCN_FusedWithOmittingEmptyRows_Demo",
                                        "GCN", 7, tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Interleaved};
    GCNFusedWithRegisterReuse *gcnFusedWithRegisterReuse =
        new GCNFusedWithRegisterReuse(inputs, stats, tileSize);
    gcnFusedWithRegisterReuse->run();
    //  for (int i = 0; i < inputs->NumOfNodes; i++){
    //    for (int j = 0; j < inputs->NumOfClasses; j++){
    //       std::cout <<
    //       gcnFusedWithRegisterReuse->OutTensor->SecondLayerOutput[i*inputs->NumOfClasses+j]
    //       << " ";
    //    }
    //    std::cout << std::endl;
    //  }
    auto gcnFusedWRRStat = gcnFusedWithRegisterReuse->printStats();
    delete gcnFusedWithRegisterReuse;
    delete stats;
    std::cout << gcnFusedWRRStat << spStat + tpStat << std::endl;
  }

  if (tp.expariment_name == "GCNFusedSequential") {
    stats =
        new swiftware::benchmark::Stats("GCN_FusedWithOmittingEmptyRows_Demo",
                                        "GCN", 7, tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Interleaved};
    GCNFusedWithOmittingEmptyRows *gcnFusedWithOmittingEmptyRows =
        new GCNFusedWithOmittingEmptyRows(inputs, stats, sp, tileSize);
    gcnFusedWithOmittingEmptyRows->run();
    auto gcnFusedWOERStat = gcnFusedWithOmittingEmptyRows->printStats();
    delete gcnFusedWithOmittingEmptyRows;
    delete stats;

    std::cout << gcnFusedWOERStat << spStat + tpStat << std::endl;
  }
  delete inputs;
  delete aCSC;
  delete aCSCFull;
}

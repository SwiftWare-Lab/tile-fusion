//
// Created by mehdi on 6/30/23.
//
#include "GCN_Multi_Layer_Demo_Utils.h"
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
  int hiddenDim = tp._embed_dim;
  int numThread = sp._num_threads;
  int tileSize = sp.TileN;
  double *layer1Weight = generateRandomDenseMatrix(features->col, hiddenDim);
  double *layer2Weight = generateRandomDenseMatrix(hiddenDim, hiddenDim);

  int numOfSamples = std::ceil(tp._sampling_ratio * tp._dim1);
  GnnTensorInputs *inputs = new GnnTensorInputs(
      layer1Weight, layer2Weight, features, aCSCFull, aCSCFull->m, hiddenDim,
      numOfSamples, numThread, 1, "GCN_Demo");

  stats = new swiftware::benchmark::Stats("GCN_Sequential_Demo", "GCN", 7,
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

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if (tp.print_header)
    std::cout << headerStat + spHeader + tpHeader << std::endl;
  std::cout << gcnStat << spStat + tpStat << std::endl;

//  stats = new swiftware::benchmark::Stats("GCN_IntraFusedParallel_Demo", "GCN", 7,
//                                          tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  GCNIntraFusedParallel *gcnParallel = new GCNIntraFusedParallel(inputs, stats);
//  gcnParallel->run();
//  auto gcnParallelStat = gcnParallel->printStats();
//  delete gcnParallel;
//  delete stats;
//  std::cout << gcnParallelStat << spStat + tpStat << std::endl;

  if (tp.expariment_name == "GCNFusedParallel") {
    stats = new swiftware::benchmark::Stats("GCN_AllFused_Demo", "GCN", 7,
                                            tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Interleaved};
    GCNAllFusedParallel *gcnFused = new GCNAllFusedParallel(inputs, stats, sp);
    gcnFused->run();
    auto gcnFusedStat = gcnFused->printStats();
    delete gcnFused;
    delete stats;

    std::cout << gcnFusedStat << spStat + tpStat << std::endl;

    stats = new swiftware::benchmark::Stats("GCN_IntraUnfused_Demo", "GCN", 7,
                                            tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Separated};
    GCNUnfused *gcnIntraUnfusedMKL = new GCNUnfused(inputs, stats);
    gcnIntraUnfusedMKL->run();
    auto gcnIntraUnfusedMKLStat = gcnIntraUnfusedMKL->printStats();
    delete gcnIntraUnfusedMKL;
    delete stats;
    std::cout << gcnIntraUnfusedMKLStat << spStat + tpStat << std::endl;
  }

  if (tp.expariment_name == "GCNFusedBatchingForBandedSpecific") {
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

  if (tp.expariment_name == "GCNFusedBatching") {
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

////    should be bug fixed
//    stats = new swiftware::benchmark::Stats(
//        "GCN_FusedParallelWithOmittingEmptyRows_Demo", "GCN", 7,
//        tp._matrix_name, numThread);
//    stats->OtherStats["PackingType"] = {Interleaved};
//    GCNFusedParallelWithOmittingEmptyRows
//        *gcnFusedParallelWithOmittingEmptyRows =
//            new GCNFusedParallelWithOmittingEmptyRows(inputs, stats, sp);
//    gcnFusedParallelWithOmittingEmptyRows->run();
//    auto gcnFusedPWOERStat =
//        gcnFusedParallelWithOmittingEmptyRows->printStats();
//    delete gcnFusedParallelWithOmittingEmptyRows;
//    delete stats;
//    std::cout << gcnFusedPWOERStat << spStat + tpStat << std::endl;
  }

  if (tp.expariment_name == "GCNWithDifferentFusionLevels") {

    stats = new swiftware::benchmark::Stats("GCN_IntraUnfused_Demo", "GCN", 7,
                                            tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Separated};
    GCNUnfused *gcnIntraUnfusedMKL = new GCNUnfused(inputs, stats);
    gcnIntraUnfusedMKL->run();
    auto gcnIntraUnfusedMKLStat = gcnIntraUnfusedMKL->printStats();
    delete gcnIntraUnfusedMKL;
    delete stats;
    std::cout << gcnIntraUnfusedMKLStat << spStat + tpStat << std::endl;

//    stats = new swiftware::benchmark::Stats("GCN_IntraTiledFused_Demo", "GCN", 7,
//                                            tp._matrix_name, numThread);
//    stats->OtherStats["PackingType"] = {Separated};
//    GCNIntraLayerTiledFused *gcnIntraTiledFused = new GCNIntraLayerTiledFused(inputs, stats, tileSize);
//    gcnIntraTiledFused->run();
//    auto gcnIntraTiledFusedStats = gcnIntraTiledFused->printStats();
//    delete gcnIntraTiledFused;
//    delete stats;
//    std::cout << gcnIntraTiledFusedStats << spStat + tpStat << std::endl;

    stats = new swiftware::benchmark::Stats("GCN_IntraFusedCSCSequential_Demo", "GCN", 7,
                                            tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Separated};
    GCNIntraFusedUsingCSCSequential *gcnIntraFusedCSCSequential = new GCNIntraFusedUsingCSCSequential(inputs, stats);
    gcnIntraFusedCSCSequential->run();
    auto gcnIFCSStats = gcnIntraFusedCSCSequential->printStats();
    delete gcnIntraFusedCSCSequential;
    delete stats;
    std::cout << gcnIFCSStats << spStat + tpStat << std::endl;

    stats = new swiftware::benchmark::Stats("GCN_IntraTiledFusedCSC_Demo", "GCN", 7,
                                            tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Interleaved};
    GCNIntraTiledFusedUsingCSC *gcnIntraTiledFusedCSC = new GCNIntraTiledFusedUsingCSC(inputs, stats, tileSize);
    gcnIntraTiledFusedCSC->run();
    auto gcnIntraTiledFusedCSCStats = gcnIntraTiledFusedCSC->printStats();
    delete gcnIntraTiledFusedCSC;
    delete stats;
    std::cout << gcnIntraTiledFusedCSCStats << spStat + tpStat << std::endl;

    stats = new swiftware::benchmark::Stats("GCN_AllTiledFusedCSC_Demo", "GCN", 7,
                                            tp._matrix_name, numThread);
    stats->OtherStats["PackingType"] = {Interleaved};
    GCNAllTiledFusedCSC *gcnAllFusedCSC = new GCNAllTiledFusedCSC(inputs, stats, tileSize);
    gcnAllFusedCSC->run();
    auto gcnAllFusedCSCStats = gcnAllFusedCSC->printStats();
    delete gcnAllFusedCSC;
    delete stats;
    std::cout << gcnAllFusedCSCStats << spStat + tpStat << std::endl;

//    stats = new swiftware::benchmark::Stats("GCN_AllFused_Demo","GCN", 7,
//                                            tp._matrix_name, numThread);
//    stats->OtherStats["PackingType"] = {Interleaved};
//    GCNAllFusedParallel *gcnAllFusedParallel = new GCNAllFusedParallel(inputs, stats, sp);
//    gcnAllFusedParallel->run();
//    auto gcnAllFusedParallelStats = gcnAllFusedParallel->printStats();
//    delete gcnAllFusedParallel;
//    delete stats;
//    std::cout << gcnAllFusedParallelStats << spStat + tpStat << std::endl;
  }
  delete inputs;
  delete aCSC;
  delete aCSCFull;
}

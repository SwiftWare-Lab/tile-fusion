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
  int hiddenDim = 8;
  int numClasses = 3;
  int numThread = sp._num_threads;
  double *layer1Weight = generateRandomDenseMatrix(features->col, hiddenDim);
  double *layer2Weight = generateRandomDenseMatrix(hiddenDim, numClasses);

  int numOfSamples = std::ceil(tp._sampling_ratio * tp._dim1);
  GnnTensorInputs *inputs = new GnnTensorInputs(
      layer1Weight, layer2Weight, features, aCSCFull, aCSCFull->m, hiddenDim,
      numClasses, numOfSamples, numThread, 1, "GCN_Demo");

  stats = new swiftware::benchmark::Stats("GCN_SequentialFusedLayer", "GCN", 1,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSequential *gcnSequentialFusedLayer = new GCNOneLayerFused(inputs, stats);
  gcnSequentialFusedLayer->run();
    for (int i = 0; i < inputs->NumOfNodes; i++){
      for (int j = 0; j < inputs->EmbedDim; j++){
        std::cout <<
        gcnSequentialFusedLayer->OutTensor->FirstLayerOutput[i*inputs->EmbedDim+j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  inputs->CorrectSol =
      new double[inputs->AdjacencyMatrix->m * inputs->EmbedDim];
  std::copy(gcnSequentialFusedLayer->OutTensor->FirstLayerOutput,
            gcnSequentialFusedLayer->OutTensor->FirstLayerOutput +
                inputs->AdjacencyMatrix->m * inputs->EmbedDim,
            inputs->CorrectSol);
  auto headerStat = gcnSequentialFusedLayer->printStatsHeader();
  auto gcnSequentialFusedLayerStat = gcnSequentialFusedLayer->printStats();
  delete stats;
  delete gcnSequentialFusedLayer;

  stats = new swiftware::benchmark::Stats("GCN_MKLUnfusedLayer", "GCN", 1,
                                          tp._matrix_name, numThread);
  GCNOneLayerMKL *gcnOneLayerMkl = new GCNOneLayerMKL(inputs, stats);
  gcnOneLayerMkl->run();
  auto gcnOneLayerMKLStat = gcnOneLayerMkl->printStats();
  for (int i = 0; i < inputs->NumOfNodes; i++){
      for (int j = 0; j < inputs->EmbedDim; j++){
        std::cout <<
            gcnOneLayerMkl->OutTensor->FirstLayerOutput[i*inputs->EmbedDim+j] << " ";
      }
      std::cout << std::endl;
  }
  delete stats;
  delete gcnOneLayerMkl;

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

  delete inputs;
  delete aCSC;
  delete aCSCFull;
}
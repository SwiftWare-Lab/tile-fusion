//
// Created by salehm32 on 15/11/23.
//

#include "SingleLayer/GCN_Single_Layer_Demo_Utils.h"
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
  Dense *features = get_dense_matrix_from_parameter(&tp, aCSC->m, tp._b_cols, tp._feature_matrix_path);
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
  Dense *layer1Weight = get_dense_matrix_from_parameter(&tp, tp._embed_dim, tp._b_cols, tp._weight1_matrix_path);
  Dense *layer2Weight =  get_dense_matrix_from_parameter(&tp, tp._embed_dim, tp._b_cols, tp._weight2_matrix_path);
  Dense *result = get_dense_matrix_from_parameter(&tp, aCSC->m, tp._b_cols, tp._weight2_matrix_path);
  int numOfSamples = std::ceil(tp._sampling_ratio * tp._dim1);

  GnnTensorInputs *inputs = new GnnTensorInputs(
      layer1Weight, layer2Weight, features, aCSCFull, aCSCFull->m, embedDim,
      numOfSamples, numThread, 7, "GCN_Demo");

  inputs->CorrectSol =
      new double[result->row * result->col];
  std::copy(result->a,
            result->a +
                result->col * result->row,
            inputs->CorrectSol);

  stats = new swiftware::benchmark::Stats("GCN_SingleLayerFused", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerSpMMGemVFused *gcnSingleLayerFused =
      new GCNSingleLayerSpMMGemVFused(inputs, stats);
  gcnSingleLayerFused->run();
  auto headerStat = gcnSingleLayerFused->printStatsHeader();
  auto gcnSequentialFusedLayerStat = gcnSingleLayerFused->printStats();
  delete stats;
  delete gcnSingleLayerFused;

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);


  if (tp.print_header)
    std::cout << headerStat + spHeader + tpHeader << std::endl;
  std::cout << gcnSequentialFusedLayerStat << spStat + tpStat << std::endl;
}
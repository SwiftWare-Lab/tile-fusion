//
// Created by salehm32 on 07/03/24.
//
#include "GCN_Single_Layer_SP_Demo_Utils.h"
#include "sparse-fusion/GraphColoring.h"
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

  float* weightSP = new float[embedDim*tp._b_cols];
  float* featuresSP = new float[aCSCFull->m * tp._b_cols];
  for(int i = 0; i < layer1Weight->row * layer1Weight->col; i++){
    weightSP[i] = (float)layer1Weight->a[i];
  }
  for(int i = 0; i < features->row * features->col; i++){
    featuresSP[i] = (float)features->a[i];
  }

  GnnTensorSpInputs *inputs = new GnnTensorSpInputs(
      weightSP, featuresSP, aCSCFull, aCSCFull->m, embedDim,
      features->col, numThread, 7, "GCN_Demo");

  delete features;
  /*
   * Method that calculates the output by doing a GeMM and then an SpMM to the
   * input.
   */
  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_UnFusedMKL", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerMKL_SP *gcnSingleLayerMkl = new GCNSingleLayerMKL_SP(inputs, stats);
  gcnSingleLayerMkl->run();
  auto gcnOneLayerMKLStat = gcnSingleLayerMkl->printStats();
  inputs->CorrectSol =
      new float[inputs->AdjacencyMatrix->m * layer1Weight->col];
  std::copy(gcnSingleLayerMkl->OutTensor->FirstLayerOutput,
            gcnSingleLayerMkl->OutTensor->FirstLayerOutput +
                inputs->AdjacencyMatrix->m * layer1Weight->col,
            inputs->CorrectSol);
  auto headerStat = gcnSingleLayerMkl->printStatsHeader();
  delete stats;
  delete gcnSingleLayerMkl;
  delete layer1Weight;

//  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_TACO", "GCN", 7,
//                                          tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  GCNSingleLayerLNRSP *gcnSingleLayerLNR =
//      new GCNSingleLayerLNRSP(inputs, stats);
//  gcnSingleLayerLNR->run();
//  auto gcnSingleLayerLNRStat = gcnSingleLayerLNR->printStats();
////  int N = gcnSingleLayerLNR->OutTensor->EmbedDim;
////  for (int i = 0; i < gcnSingleLayerLNR->OutTensor->NumOfNodes; i++) {
////    for (int j = 0; j < N; j++) {
////      std::cout << gcnSingleLayerLNR->OutTensor->FirstLayerOutput[i*N + j];
////    }
////    std::cout << std::endl;
////  }
//
//  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_LNR", "GCN", 7,
//                                          tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  GCNSingleLayerTaco *gcnSingleLayerTACO =
//      new GCNSingleLayerTaco(inputs, stats);
//  gcnSingleLayerTACO->run();
//  auto gcnSingleLayerTACOStat = gcnSingleLayerTACO->printStats();
//
//  delete stats;
//  delete gcnSingleLayerLNR;
//
//  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_CompressedGeMV", "GCN", 7,
//                                          tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  GCNSingleLayerSpMMGeMVFused *gcnSingleLayerCompressedGeMV =
//      new GCNSingleLayerSpMMGeMVFused(inputs, stats);
//  gcnSingleLayerCompressedGeMV->run();
//  auto gcnSingleLayerCompressedGeMVStat = gcnSingleLayerCompressedGeMV->printStats();
//  delete stats;
//  delete gcnSingleLayerCompressedGeMV;



  /* This is for narval that MKL SpMM doesn't work properly on it*/
  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_UnFused", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerUnFusedCSRMKLGeMMSP *gcnSingleLayerUnFused =
      new GCNSingleLayerUnFusedCSRMKLGeMMSP(inputs, stats);
  gcnSingleLayerUnFused->run();
  auto gcnSingleLayerUnFusedStat = gcnSingleLayerUnFused->printStats();
  delete stats;
  delete gcnSingleLayerUnFused;

  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_FusedSeperated", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerSparseFusedParallelWithGeMM_SP *gcnSingleLayerSparseFusedSeparated =
      new GCNSingleLayerSparseFusedParallelWithGeMM_SP(inputs, stats, sp);
  gcnSingleLayerSparseFusedSeparated->run();
  auto gcnSingleLayerSparseFusedSeparatedStat = gcnSingleLayerSparseFusedSeparated->printStats();
  delete stats;
  delete gcnSingleLayerSparseFusedSeparated;

  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_FusedSeperated_VT", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerSparseFusedParallelVTWithGeMM_SP *gcnSingleLayerSparseFusedSeparatedVT =
      new GCNSingleLayerSparseFusedParallelVTWithGeMM_SP(inputs, stats, sp);
  gcnSingleLayerSparseFusedSeparatedVT->run();
  auto gcnSingleLayerSparseFusedSeparatedVTStat = gcnSingleLayerSparseFusedSeparatedVT->printStats();
  delete stats;
  delete gcnSingleLayerSparseFusedSeparatedVT;

//  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_FusedSeperated_P2PThreading", "GCN", 7,
//                                          tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  GCNSingleLayerSparseFusedP2PThreadWithGeMM_SP *gcnSingleLayerSparseFusedSeparatedP2P =
//      new GCNSingleLayerSparseFusedP2PThreadWithGeMM_SP(inputs, stats, sp);
//  gcnSingleLayerSparseFusedSeparatedP2P->run();
//  auto gcnSingleLayerSparseFusedSeparatedP2PStat = gcnSingleLayerSparseFusedSeparatedP2P->printStats();
//  delete stats;
//  delete gcnSingleLayerSparseFusedSeparatedP2P;

  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_FusedSeperated_ReorderedUnfused", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerSparseFusedReorderedUnfusedWithGeMM_SP *gcnSingleLayerSparseFusedSeparatedROUF =
      new GCNSingleLayerSparseFusedReorderedUnfusedWithGeMM_SP(inputs, stats, sp);
  gcnSingleLayerSparseFusedSeparatedROUF->run();
  auto gcnSingleLayerSparseFusedSeparatedROUFStat = gcnSingleLayerSparseFusedSeparatedROUF->printStats();
  delete stats;
  delete gcnSingleLayerSparseFusedSeparatedROUF;

  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_FusedSeperated_ReorderedAdj", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerSparseFusedReorderedAdjWithGeMM_SP *gcnSingleLayerSparseFusedSeparatedROAdj =
      new GCNSingleLayerSparseFusedReorderedAdjWithGeMM_SP(inputs, stats, sp);
  gcnSingleLayerSparseFusedSeparatedROAdj->run();
  auto gcnSingleLayerSparseFusedSeparatedROAdjStat = gcnSingleLayerSparseFusedSeparatedROAdj->printStats();
  delete stats;
  delete gcnSingleLayerSparseFusedSeparatedROAdj;

  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_FusedSeperated_ReorderedAdj_VT", "GCN", 7,
                                          tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSingleLayerSparseFusedReorderedAdjVT_SP *gcnSingleLayerSparseFusedSeparatedROAdjVT =
      new GCNSingleLayerSparseFusedReorderedAdjVT_SP(inputs, stats, sp);
  gcnSingleLayerSparseFusedSeparatedROAdjVT->run();
  auto gcnSingleLayerSparseFusedSeparatedROAdjVTStat = gcnSingleLayerSparseFusedSeparatedROAdjVT->printStats();
  delete stats;
  delete gcnSingleLayerSparseFusedSeparatedROAdjVT;
//
//  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_Redundant_FusedSeperated", "GCN", 7,
//                                          tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  GCNSingleLayerSparseFusedRedundantParallelWithGeMM_SP *gcnSingleLayerSparseFusedRedundant =
//      new GCNSingleLayerSparseFusedRedundantParallelWithGeMM_SP(inputs, stats, sp);
//  gcnSingleLayerSparseFusedRedundant->run();
//  auto gcnSingleLayerSparseFusedRedundantStat = gcnSingleLayerSparseFusedRedundant->printStats();
//  delete stats;
//  delete gcnSingleLayerSparseFusedRedundant;
//
//  stats = new swiftware::benchmark::Stats("GCN_SingleLayer_CSCAtomic_FusedSeperated", "GCN", 7,
//                                          tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  GCNSingleLayerCSRAtomicFused *gcnSingleLayerFusedCSCAtomic =
//      new GCNSingleLayerCSRAtomicFused(inputs, stats, sp);
//  gcnSingleLayerFusedCSCAtomic->run();
//  auto gcnSingleLayerFusedCSCAtomicStat = gcnSingleLayerFusedCSCAtomic->printStats();
//  delete stats;
//  delete gcnSingleLayerFusedCSCAtomic;
//
//  stats = new swiftware::benchmark::Stats("SpMMGeMM_Fused", "GCN", 7,
//                                          tp._matrix_name, numThread);
//  stats->OtherStats["PackingType"] = {Separated};
//  GCNSingleLayerSpMMVectorizedGeMMFusedSp *gcnSingleLayerSpMMGeMM =
//      new GCNSingleLayerSpMMVectorizedGeMMFusedSp(inputs, stats, sp);
//  gcnSingleLayerSpMMGeMM->run();
//  auto gcnSingleLayerSpMMGeMMStat = gcnSingleLayerSpMMGeMM->printStats();
//  delete stats;
//  delete gcnSingleLayerSpMMGeMM;

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if (tp.print_header)
    std::cout << headerStat + spHeader + tpHeader << std::endl;

  std::cout << gcnOneLayerMKLStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerUnFusedStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerSparseFusedSeparatedStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerSparseFusedSeparatedVTStat << spStat + tpStat << std::endl;
//  std::cout << gcnSingleLayerSparseFusedSeparatedP2PStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerSparseFusedSeparatedROUFStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerSparseFusedSeparatedROAdjStat << spStat + tpStat << std::endl;
  std::cout << gcnSingleLayerSparseFusedSeparatedROAdjVTStat << spStat + tpStat << std::endl;
//  std::cout << gcnSingleLayerSparseFusedSeparatedStat << spStat + tpStat << std::endl;
//  std::cout << gcnSingleLayerSparseFusedRedundantStat << spStat + tpStat << std::endl;
//  std::cout << gcnSingleLayerFusedCSCAtomicStat << spStat + tpStat << std::endl;
//  std::cout << gcnSingleLayerSpMMGeMMStat << spStat + tpStat << std::endl;
//  std::cout << gcnSingleLayerLNRStat << spStat + tpStat << std::endl;
//  std::cout << gcnSingleLayerTACOStat << spStat + tpStat << std::endl;
//  std::cout << gcnSingleLayerCompressedGeMVStat << spStat + tpStat << std::endl;
  delete[] inputs->CorrectSol;
  delete inputs;
  delete aCSC;
  delete aCSCFull;
}
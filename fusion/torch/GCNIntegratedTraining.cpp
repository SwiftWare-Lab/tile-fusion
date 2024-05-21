
#include "E2E_IO.h"
#include "FusedGCNLayerModule.h"
#include "FusionWrapper.h"
#include "GCNModule.h"
#include "SWTensorBench.h"
#include "aggregation/def.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/SparseFusion.h"
#include <iostream>
#include <string>
using namespace sym_lib;

double calculateAverageTime(swiftware::benchmark::Timer &T) {
  double totalTime = 0;
  for (int i = 0; i < T.ElapsedTimeArray.size(); i++) {
    totalTime += T.ElapsedTimeArray[i].first;
  }
  return totalTime / T.ElapsedTimeArray.size();
}

int findBestParam(std::vector<double> &paramsAvgTime) {
  int bestParam = 0;
  double bestTime = paramsAvgTime[0];
  for (int i = 1; i < paramsAvgTime.size(); i++) {
    if (paramsAvgTime[i] < bestTime) {
      bestTime = paramsAvgTime[i];
      bestParam = i;
    }
  }
  return bestParam;
}

int tuneMatrix(CSR *Matrix, FloatDense *Features,
               sym_lib::ScheduleParameters &Sp, sym_lib::TestParameters &Tp, int InputDim, int OutputDim) {
  FloatDense *weight1 =
      generateRandomFloatDenseMatrix(InputDim, OutputDim);
  std::vector<int> parameters = {4,8,16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  std::vector<double> paramsAvgTime;
  float *out = new float[Matrix->m * OutputDim];
  float *matrixValues = new float[Matrix->nnz];
  for (int i = 0; i < Matrix->nnz; i++) {
    matrixValues[i] = (float)Matrix->x[i];
  }
  for (int j = 0; j < parameters.size(); j++) {
    Sp.IterPerPartition = parameters[j];
    sym_lib::MultiDimensionalSet *fusedCompSet = generateFusedScheduleForCSRFused(Matrix, Sp);
    swiftware::benchmark::Timer t1;
    for (int i = 0; i < 7; i++) {
      std::memset(out, 0, Matrix->m * OutputDim * sizeof(float));
      t1.start();
      forwardForOneLayerFusedParallelSeparatedVectorizedSP(
          Matrix->m, Matrix->p, Matrix->i, matrixValues, InputDim,
          OutputDim, Features->a, weight1->a, out, Sp._num_threads,
          fusedCompSet->n1_, fusedCompSet->ptr1_, fusedCompSet->ptr2_,
          fusedCompSet->ker_begin_, fusedCompSet->id_);
      t1.stop();
    }
    delete fusedCompSet;
    double avgTime = calculateAverageTime(t1);
    paramsAvgTime.push_back(avgTime);
//    std::cout << "Param: " << parameters[j] << " Time: " << std::to_string(avgTime)
//              << std::endl;
  }

  delete[] out;
  delete[] matrixValues;
  delete weight1;

  int bestParamIndex = findBestParam(paramsAvgTime);
  return parameters[bestParamIndex];
}


int main(const int argc, const char *argv[]) {
  TestParameters tp;
  tp._order_method = SYM_ORDERING::NONE;
  ScheduleParameters sp;
  swiftware::benchmark::Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  CSC *aCSC = get_matrix_from_parameter_with_adding_self_loops(&tp, true);
  CSC *aCSCFull = NULLPNTR;
  if (aCSC->stype == -1 || aCSC->stype == 1) {
    aCSCFull = sym_lib::make_full(aCSC);
  } else {
    aCSCFull = sym_lib::copy_sparse(aCSC);
  }
  delete aCSC;
  CSR *aCSR = csc_to_csr(aCSCFull);
  delete aCSCFull;
  normalizeAdjacencyMatrix(aCSR);
  FloatDense *featuresData = readFloatDenseMatrixFromParameter(
      &tp, aCSR->m, tp._embed_dim, tp.e2e_data_path + "/features.mtx");
//  std::cout << sp.IterPerPartition << std::endl;
//  sp.IterPerPartition = 1024;
//    float *weight1Data = readFloatDenseMatrixFromParameter(&tp, 16, 1433,
  //                                                         tp.e2e_data_path +
  //                                                         "/weight1.mtx");
  //  float *weight2Data = readFloatDenseMatrixFromParameter(&tp, 7, 16,
  //                                                         tp.e2e_data_path +
  //                                                         "/weight2.mtx");
  //  auto testWeight1 = torch::from_blob(weight1Data, {16, 1433},
  //  torch::kFloat32); auto testWeight2 = torch::from_blob(weight2Data, {7,
  //  16}, torch::kFloat32);;
  long *labels = getTargetsFromParameter(&tp, 1, aCSR->m,
                                         tp.e2e_data_path + "/labels.mtx");
  if (tp.print_header){
    std::cout << "Impl,Graph,Time,t1,t2," << std::endl;
  }
  int numClasses = 0;
  for (int i = 0; i < aCSR->m; i++) {
    if (labels[i] + 1 > numClasses) {
      numClasses = labels[i] + 1;
    }
  }
//  std::cout << "Num Classes: " << numClasses << std::endl;
  int trainSize = 200;
  torch::Tensor targets =
      torch::from_blob(labels, {long(aCSR->m)},  [](void* ptr){
            delete[] static_cast<long*>(ptr);
          }, torch::kInt64);
  auto features = torch::from_blob(
      featuresData->a, {(long)featuresData->row, (long)featuresData->col}, [](void* ptr){
        delete[] static_cast<float*>(ptr);
      },
      torch::kFloat32);

//  std::cout << "Features Dim" << features.sizes() << std::endl;
  //  torch::Tensor adj = readCSCMatrix("../data/cora/Cora.mtx");
  float* values;
  torch::Tensor adj = convertCSRToTorchTensor(*aCSR, values);
  // Create a new Net.
//  std::cout << "Building module" << std::endl;
  //  auto *net = new GCN(adj, features, tp._embed_dim, 32, 4, 8, 7);
  int ipL2 = tuneMatrix(aCSR, featuresData, sp, tp, tp._embed_dim, numClasses);
  int ipL1 = tuneMatrix(aCSR, featuresData, sp, tp, featuresData->col, tp._embed_dim);
  sp.IterPerPartition = ipL1;
  sym_lib::MultiDimensionalSet *fusedCompSet1 =
      generateFusedScheduleForCSRFused(aCSR, sp);
  sp.IterPerPartition = ipL2;
  sym_lib::MultiDimensionalSet *fusedCompSet2 =
      generateFusedScheduleForCSRFused(aCSR, sp);


  //  auto *net = new GCN(adj, features, tp._embed_dim, sp.TileN, 4,
  //  sp._num_threads, numClasses);
  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  //  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
//  std::cout << "Net parameters: " << net->parameters().size() << std::endl;
//  std::cout << "Training..." << std::endl;
//  torch::set_num_interop_threads(sp._num_threads);
//  torch::set_num_threads(sp._num_threads);
  GCN *net;
  if (tp.expariment_name == "MKL"){
    net = new MKLGCN(adj, features, tp._embed_dim, numClasses, sp._num_threads, fusedCompSet1);
  }else if (tp.expariment_name == "TiledFused"){
    net = new CSRFusedGCN(adj, features, tp._embed_dim, numClasses,
                          sp._num_threads, fusedCompSet1, fusedCompSet2);

  }
  torch::optim::Adam optimizer(net->parameters(), /*lr=*/0.01);
  swiftware::benchmark::Timer t1;
  t1.start();
  sparse_matrix_t MKLAdj;
  matrix_descr d;
  float* FirstLayerInput = new float[featuresData->row * featuresData->col];
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  mkl_sparse_s_create_csr(&MKLAdj, SPARSE_INDEX_BASE_ZERO, aCSR->m,
                          aCSR->n, aCSR->p, aCSR->p + 1, aCSR->i,
                          values);
  mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, MKLAdj, d,
                  SPARSE_LAYOUT_ROW_MAJOR, featuresData->a,
                  featuresData->col, featuresData->col, 0,
                  FirstLayerInput, featuresData->col);
//  for (int i = 0; i < featuresData->row; i++) {
//    for (int j = 0; j < featuresData->col; j++) {
//      if (FirstLayerInput[i * featuresData->col + j] != 0)
//      std::cout << FirstLayerInput[i * featuresData->col + j] << " ";
//    }
//  }
  auto firstLayerInputTensor = torch::from_blob(
      FirstLayerInput, {(long)featuresData->row, (long)featuresData->col}, [](void* ptr){
        delete[] static_cast<float*>(ptr);
      },
      torch::kFloat32);
  for (size_t epoch = 1; epoch <= 100; ++epoch) {
    // Reset gradients.
    optimizer.zero_grad();
    // Execute the model on the input data.
//    swiftware::benchmark::Timer t2;
//    t2.start();
    torch::Tensor prediction = net->forward(features, adj, firstLayerInputTensor);
//    t2.stop();
//    std::cout << "backward time: " << t2.printTimeCsv(0) << std::endl;
    // Compute a loss value to judge the prediction of our model.
    auto loss = torch::cross_entropy_loss(prediction.slice(0, 0, trainSize),
                                          targets.slice(0, 0, trainSize));
    // Compute gradients of the loss w.r.t. the parameters of our model.
//    swiftware::benchmark::Timer t3;
//    t3.start();
    loss.backward();
//    t3.stop();
//    std::cout << "backward time: " << t3.printTimeCsv(0) << std::endl;
//     Update the parameters based on the calculated gradients.
    optimizer.step();
    std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>() << std::endl;
  }
  t1.stop();
  std::cout <<  tp.expariment_name << "," << tp._matrix_name << "," << t1.printTimeCsv(0)
            << "," << ipL1 << "," << ipL2<< "," << std::endl;
  delete net;
  delete[] values;
  delete aCSR;
  delete fusedCompSet1;
  delete fusedCompSet2;
}

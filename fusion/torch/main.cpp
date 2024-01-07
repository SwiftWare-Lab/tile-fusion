
#include "FusedGCNLayerModule.h"
#include "FusionWrapper.h"
#include "GCNModule.h"
#include "SWTensorBench.h"
#include "aggregation/def.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/SparseFusion.h"
#include <iostream>

torch::Tensor convertCSCToTorchTensor(sym_lib::CSC &matrix) {
  return torch::sparse_csc_tensor(
      torch::from_blob(matrix.p, {long(matrix.n)}, torch::kInt32),
      torch::from_blob(matrix.i, {long(matrix.nnz)}, torch::kInt32),
      torch::from_blob(matrix.x, {long(matrix.nnz)}, torch::kFloat32),
      {long(matrix.m), long(matrix.n)}, torch::kFloat32);
}

torch::Tensor convertDenseMatrixToTensor(sym_lib::Dense &matrix){
  return torch::from_blob(matrix.a, {(long)matrix.row, (long)matrix.col}, torch::kFloat32);
}

using namespace sym_lib;

int main(const int argc, const char *argv[]) {
  TestParameters tp;tp._order_method=SYM_ORDERING::NONE;
  ScheduleParameters sp;
  swiftware::benchmark::Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  CSC *aCSC = get_matrix_from_parameter(&tp);
  Dense *featuresDense = get_dense_matrix_from_parameter(&tp, aCSC->m, tp._b_cols,
                                                    tp._feature_matrix_path);
  auto features = convertDenseMatrixToTensor(*featuresDense);
//  torch::Tensor adj = readCSCMatrix("../data/cora/Cora.mtx");
  torch::Tensor adj = convertCSCToTorchTensor(*aCSC);
  //    std::cout << adj.ccol_indices() << std::endl;
//  torch::Tensor features = torch::rand({2708, 1433});
  torch::Tensor targets = torch::ones({2708}, torch::kLong);
  // Create a new Net.
  std::cout << "Building module" << std::endl;
  auto *net = new GCN(adj, features, 16, 64, 64, 4);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.05);
  std::cout << "Training..." << std::endl;

  for (size_t epoch = 1; epoch <= 100; ++epoch) {
    // Reset gradients.
    optimizer.zero_grad();
    // Execute the model on the input data.
    torch::Tensor prediction = net->forward(features, adj);
    // Compute a loss value to judge the prediction of our model.
    auto loss = torch::nll_loss(prediction, targets);
    // Compute gradients of the loss w.r.t. the parameters of our model.
    loss.backward();
    // Update the parameters based on the calculated gradients.
    optimizer.step();
    std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>()
              << std::endl;
  }
}

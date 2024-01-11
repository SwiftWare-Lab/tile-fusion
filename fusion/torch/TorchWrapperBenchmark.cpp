//
// Created by salehm32 on 09/01/24.
//
#include "E2E_IO.h"
#include "FusedGCNForward.h"
#include "FusionWrapper.h"
#include "Stats.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"

int main(const int argc, const char *argv[]) {
  sym_lib::TestParameters tp;
  tp._order_method = sym_lib::SYM_ORDERING::NONE;
  sym_lib::ScheduleParameters sp;
  swiftware::benchmark::Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  sym_lib::CSC *aCSC = get_matrix_from_parameter(&tp, true);
  sym_lib::CSR *aCSR = sym_lib::csc_to_csr(aCSC);
  normalizeAdjacencyMatrix(aCSR);
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);
  int embedDim = tp._embed_dim;
  int numClasses = 3;
  int numThread = sp._num_threads;
  int tileSize = sp.TileN;
  int kTileSize = 8;
  FloatDense *layer1Weight = readFloatDenseMatrixFromParameter(
      &tp, tp._embed_dim, tp._b_cols, tp.e2e_data_path + "/weight1.mtx");
  FloatDense *layer2Weight = readFloatDenseMatrixFromParameter(
      &tp, tp._embed_dim, tp._embed_dim, tp.e2e_data_path + "/weight2.mtx");
  auto weight1 = torch::from_blob(
      layer1Weight, {(long)layer1Weight->row, (long)layer1Weight->col},
      torch::kFloat32);
  sym_lib::MultiDimensionalSet *fusedCompSet =
      generateFusedScheduleForCSRFused(aCSR, sp);
  std::vector<torch::Tensor> fusedCompSetTensor =
      createScheduleForCSR(fusedCompSet);
  float *floatValue = new float[aCSR->nnz];
  for (int i = 0; i < aCSR->nnz; i++) {
    floatValue[i] = (float)aCSR->x[i];
  }
  auto adj = torch::sparse_csr_tensor(
      torch::from_blob(aCSR->p, {long(aCSR->n + 1)}, torch::kInt32),
      torch::from_blob(aCSR->i, {long(aCSR->nnz)}, torch::kInt32),
      torch::from_blob(floatValue, {long(aCSR->nnz)}, torch::kFloat32),
      {long(aCSR->m), long(aCSR->n)}, torch::kFloat32);
  FloatDense *featuresData = readFloatDenseMatrixFromParameter(
      &tp, aCSR->m, tp._b_cols, tp.e2e_data_path + "/features.mtx");
  auto feature = torch::from_blob(
      featuresData->a, {(long)featuresData->row, (long)featuresData->col},
      torch::kFloat32);
//  torch::set_num_threads(numThread);
//  torch::set_num_interop_threads(numThread);
  for (int i = 0; i < 1; i++) {
    swiftware::benchmark::Timer t1;
    t1.start();
    auto fusedAnswer = CSRFusedGCNForwardFunction::apply(
        feature, adj, weight1, fusedCompSetTensor[0], fusedCompSetTensor[1],
        fusedCompSetTensor[2], fusedCompSetTensor[3], numThread,
        fusedCompSetTensor[4][0].item<int>());
    t1.stop();
    std::cout << "Fused Time: " << t1.printTimeCsv(i) << std::endl;
    swiftware::benchmark::Timer t2;
    t2.start();
    auto unfusedAnswer = CSRUnFusedGCNForwardFunction::apply(feature, adj, weight1);
    t2.stop();
    std::cout << "UnFused Time: " << t2.printTimeCsv(i) << std::endl;
    std::cout << "Fused Answer: " << fusedAnswer << std::endl;
    std::cout << "UnFused Answer: " << unfusedAnswer << std::endl;
  }
}
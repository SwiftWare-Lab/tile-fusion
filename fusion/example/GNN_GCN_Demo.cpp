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
  CSR *adjMat = sym_lib::csc_to_csr(aCSC);
  Dense *features = get_feature_matrix_from_parameter(&tp);
  if (aCSC->m != aCSC->n) {
    return -1;
  }
  tp._dim1 = aCSC->m;
  tp._dim2 = aCSC->n;
  tp._nnz = aCSC->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);
  int hiddenDim = 128;
  int numThread = sp._num_threads;
  GnnTensorInputs *inputs =
      new GnnTensorInputs(features, adjMat, adjMat->m,
                          hiddenDim, 3, 512, numThread, 1, "GCN_Demo");
  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo", "SpMM", 7, tp._matrix_name, numThread);
  GCNGnn *gcnGnn = new GCNGnn(inputs, stats);
  gcnGnn->run();
  auto headerStat = gcnGnn->printStatsHeader();
  auto gcnStat = gcnGnn->printStats();
  delete gcnGnn;
  delete stats;
  delete aCSC;

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if(tp.print_header)
    std::cout<<headerStat+spHeader+tpHeader<<std::endl;
  std::cout<< gcnStat <<spStat+tpStat<<std::endl;
  delete features;
}

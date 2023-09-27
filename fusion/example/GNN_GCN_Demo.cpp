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
  Dense *features = get_feature_matrix_from_parameter(&tp);
  CSC *aCSCFull = nullptr;
  if(aCSC->stype == -1 || aCSC->stype == 1){
    aCSCFull = sym_lib::make_full(aCSC);
  } else{
    aCSCFull = sym_lib::copy_sparse(aCSC);
  }
  if (aCSC->m != aCSC->n) {
    return -1;
  }
  tp._dim1 = aCSCFull->m;
  tp._dim2 = aCSCFull->n;
  tp._nnz = aCSCFull->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);
  int hiddenDim = 128;
  int numThread = sp._num_threads;
  GnnTensorInputs *inputs =
      new GnnTensorInputs(features, aCSCFull, aCSCFull->m,
                          hiddenDim, 3, aCSCFull->m, numThread, 1, "GCN_Demo");
  stats = new swiftware::benchmark::Stats("GCN_Sequential_Demo", "GCN", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNSequential *gcnGnn = new GCNSequential(inputs, stats);
  gcnGnn->run();
  auto headerStat = gcnGnn->printStatsHeader();
  auto gcnStat = gcnGnn->printStats();
  delete gcnGnn;
  delete stats;

  stats = new swiftware::benchmark::Stats("GCN_Parallel_Demo", "GCN", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Separated};
  GCNParallel *gcnParallel = new GCNParallel(inputs, stats);
  gcnParallel->run();
  auto gcnParallelStat = gcnParallel->printStats();
  delete gcnParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("GCN_Fused_Demo", "GCN", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  GCNFused *gcnFused = new GCNFused(inputs, stats, sp);
  gcnFused->run();
  auto gcnFusedStat = gcnFused->printStats();
  delete gcnFused;
  delete stats;

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if(tp.print_header)
    std::cout<<headerStat+spHeader+tpHeader<<std::endl;
  std::cout<< gcnStat <<spStat+tpStat<<std::endl;
  std::cout<< gcnParallelStat <<spStat+tpStat<<std::endl;
  std::cout<< gcnFusedStat <<spStat+tpStat<<std::endl;
  delete features;
  delete aCSC;
  delete aCSCFull;
}

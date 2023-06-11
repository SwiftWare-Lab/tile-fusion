//
// Created by kazem on 02/05/23.
//

#include "aggregation/def.h"
#include "SpMM_SpMM_Demo_Utils.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/SparseFusion.h"
#include <fstream>

using namespace sym_lib;
// A is MxK, C is KxN, B is LxM, and D is LxN; AC is MxN
int main(const int argc, const char *argv[]){
  TestParameters tp;tp._order_method=SYM_ORDERING::NONE;
  ScheduleParameters sp;
  Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  CSC *aLtCsc=NULLPNTR;
  CSC *aCSC = get_matrix_from_parameter(&tp);
  if(aCSC->m != aCSC->n){
    return -1;
  }
  tp._dim1 = aCSC->m; tp._dim2 = aCSC->n; tp._nnz = aCSC->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);
  CSC *bCSC = sym_lib::copy_sparse(aCSC);
  auto *alCSC = make_half(aCSC->n, aCSC->p, aCSC->i, aCSC->x);
  std::vector<CSC*> orderedVec;
  if(tp._order_method != SYM_ORDERING::NONE){
    // applies ordering here
    get_reorderd_matrix(alCSC, orderedVec);
    delete alCSC;
    alCSC = orderedVec[0];
  }
  //print_csc(1,"",aCSC);
  int numThread = sp._num_threads, numTrial = 7; std::string expName = "SpMM_SpMM_Demo";
  auto *inSpMM = new TensorInputs<double>(aCSC->m,  tp._b_cols, aCSC->n,
                                         bCSC->m, aCSC, bCSC,
                                          numThread, numTrial, expName);

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo", "SpMM", 7, tp._matrix_name, numThread);
  auto *unfused = new SpMMSpMMUnFused(inSpMM, stats);
  unfused->run();
  //unfused->OutTensor->printDx();
  inSpMM->CorrectSol = std::copy(unfused->OutTensor->Dx, unfused->OutTensor->Dx + unfused->OutTensor->M * unfused->OutTensor->N, inSpMM->CorrectMul);
  inSpMM->IsSolProvided = true;
  auto headerStat = unfused->printStatsHeader();
  auto baselineStat = unfused->printStats();
  delete unfused;
  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo_UnFusedParallel", "SpMM", 7, tp._matrix_name, numThread);
  auto *unfusedParallel = new SpMMSpMMUnFusedParallel(inSpMM, stats);
  unfusedParallel->run();
  //unfusedParallel->OutTensor->printDx();
  auto unfusedParallelStat = unfusedParallel->printStats();
  delete unfusedParallel;
  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo_InnerProduct_UnFusedParallel", "SpMM", 7, tp._matrix_name, numThread);
  auto *unfusedOutParallel = new SpMMSpMMUnFusedInnerParallel(inSpMM, stats);
  unfusedOutParallel->run();
  //unfusedParallel->OutTensor->printDx();
  auto unfusedOutParallelStat = unfusedOutParallel->printStats();
  delete unfusedOutParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo_CTiled_UnFusedParallel", "SpMM", 7, tp._matrix_name, numThread);
  auto *unfusedCTiledParallel = new SpMMSpMMUnFusedCTiledParallel(inSpMM, stats);
  unfusedCTiledParallel->run();
  auto unfusedCTiledParallelStat = unfusedCTiledParallel->printStats();
  delete unfusedCTiledParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *fusedParallel = new SpMMSpMMFusedInterLayer(inSpMM, stats);
  fusedParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelStat = fusedParallel->printStats();
  delete fusedParallel;
  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_OuterProduct_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *fusedOuterParallel = new SpMMSpMMFusedInnerProdInterLayer(inSpMM, stats);
  fusedOuterParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelOutStat = fusedOuterParallel->printStats();
  delete fusedOuterParallel;
  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Separated_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Separated};
  auto *fusedSepParallel = new SpMMSpMMFusedSepInterLayer(inSpMM, stats);
  fusedSepParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelSepStat = fusedSepParallel->printStats();
  delete fusedSepParallel;
  delete stats;


  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if(tp.print_header)
    std::cout<<headerStat+spHeader+tpHeader<<std::endl;
  std::cout<<baselineStat<<spStat+tpStat<<std::endl;
  std::cout<<unfusedParallelStat<<spStat+tpStat<<std::endl;
  std::cout<<unfusedOutParallelStat<<spStat+tpStat<<std::endl;
  std::cout<<unfusedCTiledParallelStat<<spStat+tpStat<<std::endl;
  std::cout<<fusedParallelStat<<spStat+tpStat<<std::endl;
  std::cout<<fusedParallelOutStat<<spStat+tpStat<<std::endl;
  std::cout<<fusedParallelSepStat<<spStat+tpStat;

//  sp._num_w_partition = 2;
//  //print_csc(1,"",A_csc);
//  auto *sf01 = new SparseFusion(&sp, 2);
//  auto *mvDAG =  diagonal(alCSC->n, 1.0);
//  sf01->fuse(0, mvDAG, NULLPNTR);
//  //sf01->print_final_list();
//  sf01->fuse(1, mvDAG, alCSC);
//  sf01->print_final_list();
//
//
//  auto *sf02 = new SparseFusion(&sp, 2);
//  sf02->fuse(0, mvDAG, NULLPNTR);
//  //sf01->print_final_list();
//  sf02->fuse(1, mvDAG, alCSC);
//  auto *fusedCompSet = sf02->getFusedCompressed();
//  fusedCompSet->print_3d();
//
//
//
//
//  auto tpCsv = tp.print_csv(tp.print_header);
//  auto spCsv = sp.print_csv(tp.print_header);
//  if(tp.print_header){
//    std::cout<<std::get<0>(tpCsv)<<std::get<0>(spCsv)<<"\n";
//  }
//  std::cout<<std::get<1>(tpCsv)<<std::get<1>(spCsv);

  delete aCSC;
  delete bCSC;
  delete alCSC;
  delete inSpMM;

  return 0;
}

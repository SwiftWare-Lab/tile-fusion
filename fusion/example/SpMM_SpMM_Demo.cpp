//
// Created by kazem on 02/05/23.
//

#include "sparse-fusion/SparseFusion.h"
#include "SpMM_SpMM_Demo_Utils.h"
#include "aggregation/def.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include <fstream>

using namespace sym_lib;

// A is MxK, C is KxN, B is LxN, and D is LxM; AC is MxN
int main(const int argc, const char *argv[]){
  TestParameters tp;tp._order_method=SYM_ORDERING::NONE;
  ScheduleParameters sp;
  Stats *stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo", "SpMM", 7);
  parse_args(argc, argv, &sp, &tp);
  CSC *aLtCsc=NULLPNTR;
  CSC *aCSC = get_matrix_from_parameter(&tp);
  CSC *bCSC = sym_lib::copy_sparse(aCSC);
  auto *alCSC = make_half(aCSC->n, aCSC->p, aCSC->i, aCSC->x);
  std::vector<CSC*> orderedVec;
  if(tp._order_method != SYM_ORDERING::NONE){
    // applies ordering here
    get_reorderd_matrix(alCSC, orderedVec);
    delete alCSC;
    alCSC = orderedVec[0];
  }
  print_csc(1,"",aCSC);
  int numThread = 1, numTrial = 7; std::string expName = "SpMM_SpMM_Demo";
  auto *inSpMM = new TensorInputs<double>(aCSC->m, aCSC->n, 4,
                                         bCSC->m, aCSC, bCSC,
                                          numThread, numTrial, expName);

  auto *unfused = new SpMMSpMMUnFused(inSpMM, stats);
  unfused->run();
  unfused->OutTensor->printDx();
  inSpMM->CorrectSol = std::copy(unfused->OutTensor->Dx, unfused->OutTensor->Dx + unfused->OutTensor->M * unfused->OutTensor->N, inSpMM->CorrectMul);
  inSpMM->IsSolProvided = true;
  auto headerStat = unfused->printStatsHeader();
  auto baselineStat = unfused->printStats();
  delete unfused;

  auto *unfusedParallel = new SpMMSpMMUnFusedParallel(inSpMM, stats);
  unfusedParallel->run();
  unfusedParallel->OutTensor->printDx();
  auto unfusedParallelStat = unfusedParallel->printStats();
  delete unfusedParallel;

  auto *fusedParallel = new SpMMSpMMFusedInterLayer(inSpMM, stats);
  fusedParallel->run();
  fusedParallel->OutTensor->printDx();
  auto fusedParallelStat = fusedParallel->printStats();
  delete fusedParallel;


  std::cout<<headerStat<<std::endl;
  std::cout<<baselineStat<<std::endl;
  std::cout<<unfusedParallelStat<<std::endl;

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
  delete stats;

  return 0;
}
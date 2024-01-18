//
// Created by salehm32 on 17/01/24.
//


#include "SpMM_SpMM_Demo_Utils.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/SparseFusion.h"
#include <fstream>

using namespace sym_lib;
// A is MxK, C is KxN, B is LxM, and D is LxN; AC is MxN
int main(const int argc, const char *argv[]){
  TestParameters tp;tp._order_method=SYM_ORDERING::NONE;
  ScheduleParameters sp;
  swiftware::benchmark::Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  CSC *aLtCsc=NULLPNTR;
  CSC *aCSC = get_matrix_from_parameter(&tp);
  if(aCSC->m != aCSC->n){
    return -1;
  }
  CSC *aCSCFull = nullptr;
  if(aCSC->stype == -1 || aCSC->stype == 1){
    aCSCFull = sym_lib::make_full(aCSC);
  } else{
    aCSCFull = sym_lib::copy_sparse(aCSC);
  }
  tp._dim1 = aCSCFull->m; tp._dim2 = aCSCFull->n; tp._nnz = aCSCFull->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);

  CSC *bCSC = sym_lib::copy_sparse(aCSCFull);
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
  auto *inSpMM = new TensorInputs<double>(aCSCFull->m,  tp._b_cols, aCSCFull->n,
                                          bCSC->m, aCSCFull, bCSC,
                                          numThread, numTrial, expName);

  //  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo", "SpMM", 7, tp._matrix_name, numThread);
  //  stats->OtherStats["PackingType"] = {Interleaved};
  //  auto *unfused = new SpMMSpMMUnFused(inSpMM, stats);
  //  unfused->run();
  //  //unfused->OutTensor->printDx();
  //  inSpMM->CorrectSol = std::copy(unfused->OutTensor->Dx, unfused->OutTensor->Dx + unfused->OutTensor->M * unfused->OutTensor->N, inSpMM->CorrectMul);
  //  inSpMM->IsSolProvided = true;
  //  auto headerStat = unfused->printStatsHeader();
  //  auto baselineStat = unfused->printStats();
  //  delete unfused;
  //  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_MKL", "SpMM", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *singleSpmmMKL = new SpMMMKLImpl(inSpMM, stats);
  singleSpmmMKL->run();
  //singleSpmmMKL->OutTensor->printDx();
  inSpMM->CorrectSol = std::copy(singleSpmmMKL->OutTensor->Dx,
                singleSpmmMKL->OutTensor->Dx +
                    singleSpmmMKL->OutTensor->M * singleSpmmMKL->OutTensor->N, inSpMM->CorrectMul);
  inSpMM->IsSolProvided = true;
  auto headerStat = singleSpmmMKL->printStatsHeader();
  auto spmmMKLStat = singleSpmmMKL->printStats();
  delete singleSpmmMKL;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_Vectorized_4_8","SpMM", 7,tp._matrix_name,numThread);
  auto *spmmVectorized48 = new SpMMParallelVectorizedUnroll48(inSpMM, stats, sp);
  spmmVectorized48->run();
  //unfused->OutTensor->printDx();
  auto spmmVectorized48Stats = spmmVectorized48->printStats();
  delete spmmVectorized48;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_Vectorized_2_16","SpMM", 7,tp._matrix_name,numThread);
  auto *spmmVectorized216 = new SpMMParallelVectorizedUnroll216(inSpMM, stats, sp);
  spmmVectorized216->run();
  //unfused->OutTensor->printDx();
  auto spmmVectorized216Stats = spmmVectorized216->printStats();
  delete spmmVectorized216;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_Vectorized_16","SpMM", 7,tp._matrix_name,numThread);
  auto *spmmVectorized16 = new SpMMParallelVectorizedUnroll16(inSpMM, stats, sp);
  spmmVectorized16->run();
  //unfused->OutTensor->printDx();
  auto spmmVectorized16Stats = spmmVectorized16->printStats();
  delete spmmVectorized16;
  delete stats;


  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if(tp.print_header)
    std::cout<<headerStat+spHeader+tpHeader<<std::endl;
  std::cout<< spmmMKLStat <<spStat+tpStat<<std::endl;
  std::cout<<spmmVectorized48Stats<<spStat+tpStat<<std::endl;
  std::cout<<spmmVectorized216Stats<<spStat+tpStat<<std::endl;
  std::cout<<spmmVectorized16Stats<<spStat+tpStat<<std::endl;

#ifdef __AVX512F__
  stats = new swiftware::benchmark::Stats("SpMM_Vectorized_AVX512","SpMM", 7,tp._matrix_name,numThread);
  auto spmmVectorizedAvx512 = new SpMMParallelVectorizedAVX512_128(inSpMM, stats, sp);
  spmmVectorizedAvx512->run();
  //unfused->OutTensor->printDx();
  auto spmmVectorizedAvx512 = spmmVectorizedAvx512->printStats();
  delete spmmVectorizedAvx512;
  delete stats;
  std::cout<<spmmVectorizedAvx512<<spStat+tpStat<<std::endl;

#endif
  delete aCSC;
  delete aCSCFull;
  delete bCSC;
  delete alCSC;
  delete inSpMM;
  //  delete dsaturColoring;
  //  delete dsaturColoringWithKTiling;

  return 0;
}
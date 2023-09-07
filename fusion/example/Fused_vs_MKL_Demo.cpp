//
// Created by mehdi on 5/25/23.
//

#include "SpMM_SpMM_Demo_Utils.h"
#include "SpMM_SpMM_MKL_Demo_Utils.h"
#include "aggregation/def.h"
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
  CSC *aCSC = get_matrix_from_parameter(&tp);
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

  int numThread = sp._num_threads, numTrial = 7; std::string expName = "SpMM_SpMM_Demo";
  auto *inSpMM = new TensorInputs<double>(aCSC->m,  tp._b_cols, aCSC->n,
                                          bCSC->m, aCSC, bCSC,
                                          numThread, numTrial, expName);

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Demo", "SpMM", 7, tp._matrix_name, numThread);
  auto *unfused = new SpMMSpMMUnFused(inSpMM, stats);
  unfused->run();
  inSpMM->CorrectSol = std::copy(unfused->OutTensor->Dx, unfused->OutTensor->Dx + unfused->OutTensor->M * unfused->OutTensor->N, inSpMM->CorrectMul);
  inSpMM->IsSolProvided = true;
  auto headerStat = unfused->printStatsHeader();
  auto baselineStat = unfused->printStats();
  delete unfused;
  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_MKL", "SpMM", 7, tp._matrix_name, numThread);
  auto *mklImpl = new SpMMSpMMMKL(inSpMM, stats);
  mklImpl->run();
  auto mklImplStat = mklImpl->printStats();
  delete mklImpl;
  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel", "SpMM", 7, tp._matrix_name, numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *fusedParallel = new SpMMSpMMFusedInterLayer(inSpMM, stats, sp);
  fusedParallel->run();
  auto fusedParallelStat = fusedParallel->printStats();
  delete fusedParallel;
  delete stats;


  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedParallel_BFS","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto spBfs = sp; spBfs.SeedPartitioningParallelism = BFS;
  auto *fusedParallelBfs = new SpMMSpMMFusedInterLayer(inSpMM, stats, spBfs);
  fusedParallelBfs->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelStatBfs = fusedParallelBfs->printStats();
  delete fusedParallelBfs;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_FusedTiledParallel_Redundant_General","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Separated};
  auto *fusedTiledParallelGen = new SpMMSpMMFusedInterLayerRedundant(inSpMM, stats, sp);
  fusedTiledParallelGen->run();
  //fusedTiledParallelGen->OutTensor->printDx();
  auto fusedTiledParallelGenStat = fusedTiledParallelGen->printStats();
  auto profileInfoRed = fusedTiledParallelGen->getSpInfo().printCSV(true);
  std::string profHeaderRed = std::get<0>(profileInfoRed);
  std::string profStatRed = std::get<1>(profileInfoRed);
  delete fusedTiledParallelGen;
  delete stats;



  stats = new swiftware::benchmark::Stats("SpMM_SpMM_OuterProduct_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *fusedOuterParallel = new SpMMSpMMFusedInnerProdInterLayer(inSpMM, stats, sp);
  fusedOuterParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelOutStat = fusedOuterParallel->printStats();
  delete fusedOuterParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Mixed_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Interleaved};
  auto *fusedMixedParallel = new SpMMSpMMFusedInnerProdInterLayer(inSpMM, stats, sp);
  fusedMixedParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelMixedStat = fusedMixedParallel->printStats();
  delete fusedMixedParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Separated_FusedParallel","SpMM", 7,tp._matrix_name,numThread);
  stats->OtherStats["PackingType"] = {Separated};
  auto *fusedSepParallel = new SpMMSpMMFusedSepInterLayer(inSpMM, stats, sp);
  fusedSepParallel->run();
  //fusedParallel->OutTensor->printDx();
  auto fusedParallelSepStat = fusedSepParallel->printStats();
  delete fusedSepParallel;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_SpMM_Profiler","SpMM", 7,tp._matrix_name,numThread);
  auto *fusionProfiler = new SpMMSpMMFusionProfiler(inSpMM, stats, sp);
  fusionProfiler->run();
  //unfused->OutTensor->printDx();
  inSpMM->IsSolProvided = true;
  auto profileInfo = fusionProfiler->getSpInfo().printCSV(true);
  std::string profHeader = std::get<0>(profileInfo);
  std::string profStat = std::get<1>(profileInfo);
  //delete fusionProfiler;
  delete stats;



  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if(tp.print_header)
    std::cout<<headerStat+spHeader+tpHeader+profHeader<<std::endl;
  std::cout<<baselineStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<fusedParallelStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<fusedParallelStatBfs<<spStat+tpStat+profStat<<std::endl;
  std::cout<<fusedTiledParallelGenStat<<spStat+tpStat+profStatRed<<std::endl;
  std::cout<<fusedParallelOutStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<fusedParallelMixedStat<<spStat+tpStat+profStat<<std::endl;
  std::cout<<fusedParallelSepStat<<spStat+tpStat+profStat;


  delete aCSC;
  delete bCSC;
  delete alCSC;
  delete inSpMM;

  return 0;
}

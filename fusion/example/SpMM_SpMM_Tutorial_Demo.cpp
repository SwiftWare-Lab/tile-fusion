
#include "sparse-fusion/SparseFusion.h"
#include "SpMM_SpMM_Demo_Utils.h"
#include "aggregation/def.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include <fstream>


int main(const int argc, const char* argv[]){
    sym_lib::TestParameters tp; tp._order_method=sym_lib::SYM_ORDERING::NONE;
    sym_lib::ScheduleParameters sp;
    sym_lib::parse_args(argc, argv, &sp, &tp);
    sym_lib::CSC *aCSC = get_matrix_from_parameter(&tp);
    tp._dim1 = aCSC->m; tp._dim2 = aCSC->n; tp._nnz = aCSC->nnz;
    tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);
    sym_lib::CSC *bCSC = sym_lib::copy_sparse(aCSC);

    int numThread = sp._num_threads, numTrial = 7; std::string expName = "SpMM_SpMM_Tutorial_Demo";
    auto *inSpMM = new TensorInputs<double>(aCSC->m,  tp._b_cols, aCSC->n,
                                            bCSC->m, aCSC, bCSC,
                                            numThread, numTrial, expName);
    swiftware::benchmark::Stats* stats = new swiftware::benchmark::Stats("SpMM_SpMM_Tutorial_Demo_UnFusedParallel", "SpMM", 7, tp._matrix_name, numThread);
    auto *unfusedParallel = new SpMMSpMMUnFusedParallel(inSpMM, stats);
    unfusedParallel->run();
    inSpMM->CorrectSol = std::copy(unfusedParallel->OutTensor->Dx, unfusedParallel->OutTensor->Dx + unfusedParallel->OutTensor->M * unfusedParallel->OutTensor->N, inSpMM->CorrectMul);
    inSpMM->IsSolProvided = true;
    auto headerStat = unfusedParallel->printStatsHeader();
    auto unfusedParallelStat = unfusedParallel->printStats();
    delete unfusedParallel;
    delete stats;

    stats = new swiftware::benchmark::Stats("SpMM_SpMM_Tutorial_Demo_UnFusedParallel2", "SpMM", 7, tp._matrix_name, numThread);
    auto *unfusedParallel2 = new SpMMSpMMUnFusedParallel2(inSpMM, stats);
    unfusedParallel2->run();
    auto unfusedParallelStat2 = unfusedParallel2->printStats();
    delete unfusedParallel2;
    delete stats;

    auto csvInfo = sp.print_csv(true);
    std::string spHeader = std::get<0>(csvInfo);
    std::string spStat = std::get<1>(csvInfo);

    auto tpCsv = tp.print_csv(true);
    std::string tpHeader = std::get<0>(tpCsv);
    std::string tpStat = std::get<1>(tpCsv);

    if(tp.print_header)
        std::cout<<headerStat+spHeader+tpHeader<<std::endl;
    std::cout<<unfusedParallelStat<<spStat+tpStat<<std::endl;
    std::cout<<unfusedParallelStat2<<spStat+tpStat;
    
    delete aCSC;
    delete bCSC;
    delete inSpMM;

    
}
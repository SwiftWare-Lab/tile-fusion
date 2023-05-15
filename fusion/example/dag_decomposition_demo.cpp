//
// Created by Kazem on 2023-02-18.
//

#include <fstream>

#include "sparse-fusion/SparseFusion.h"
#include "sparse-fusion/Fusion_Utils.h"

#include "aggregation/def.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "aggregation/test_utils.h"

using namespace sym_lib;

int main(const int argc, const char *argv[]){
 TestParameters tp;tp._order_method=SYM_ORDERING::SYM_METIS;
 ScheduleParameters sp;
 parse_args(argc, argv, &sp, &tp);
 CSC *aLtCsc=NULLPNTR;
 CSC *aCsc = get_matrix_from_parameter(&tp);
 auto *alCsc = make_half(aCsc->n, aCsc->p, aCsc->i, aCsc->x);
 std::vector<CSC*> orderedVec;
 if(tp._order_method != SYM_ORDERING::NONE){
  // applies ordering here
  get_reorderd_matrix(alCsc, orderedVec);
  delete alCsc;
  alCsc = orderedVec[0];
 }
 auto *alCsr = csc_to_csr(alCsc);

 //print_csc(1,"",A_csc);
 auto *sf01 = new SparseFusion(&sp, 2);
 sf01->fuse(0, alCsc, NULLPNTR);

 auto *mvDag =  diagonal(alCsc->n,1.0);
 sf01->fuse(1, mvDag, mvDag);
 sf01->print_final_list();


 auto tpCsv = tp.print_csv(tp.print_header);
 auto spCsv = sp.print_csv(tp.print_header);
 if(tp.print_header){
  std::cout<<std::get<0>(tpCsv)<<std::get<0>(spCsv)<<"\n";
 }
 std::cout<<std::get<1>(tpCsv)<<std::get<1>(spCsv);

 return 0;
}
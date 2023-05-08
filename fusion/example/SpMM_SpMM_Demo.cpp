//
// Created by kazem on 02/05/23.
//

#include <fstream>

#include "sparse-fusion/SparseFusion.h"
#include "sparse-fusion/Fusion_Utils.h"

#include "aggregation/def.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"

using namespace sym_lib;

int main(const int argc, const char *argv[]){
 TestParameters tp;tp._order_method=SYM_ORDERING::NONE;
 ScheduleParameters sp;
 parse_args(argc, argv, &sp, &tp);
 CSC *aLtCsc=NULLPNTR;
 CSC *aCSC = get_matrix_from_parameter(&tp);
 auto *alCSC = make_half(aCSC->n, aCSC->p, aCSC->i, aCSC->x);
 std::vector<CSC*> orderedVec;
 if(tp._order_method != SYM_ORDERING::NONE){
  // applies ordering here
  get_reorderd_matrix(alCSC, orderedVec);
  delete alCSC;
  alCSC = orderedVec[0];
 }
 print_csc(1,"",alCSC);
 auto *alCSR = csc_to_csr(alCSC);

 sp._num_w_partition = 2;
 //print_csc(1,"",A_csc);
 auto *sf01 = new SparseFusion(&sp, 2);
 auto *mvDAG =  diagonal(alCSC->n, 1.0);
 sf01->fuse(0, mvDAG, NULLPNTR);
 //sf01->print_final_list();
 sf01->fuse(1, mvDAG, alCSC);
 sf01->print_final_list();


 auto *sf02 = new SparseFusion(&sp, 2);
 sf02->fuse(0, mvDAG, NULLPNTR);
 //sf01->print_final_list();
 sf02->fuse(1, mvDAG, alCSC);
 auto *fusedCompSet = sf02->getFusedCompressed();
 fusedCompSet->print_3d();




 auto tpCsv = tp.print_csv(tp.print_header);
 auto spCsv = sp.print_csv(tp.print_header);
 if(tp.print_header){
  std::cout<<std::get<0>(tpCsv)<<std::get<0>(spCsv)<<"\n";
 }
 std::cout<<std::get<1>(tpCsv)<<std::get<1>(spCsv);

 return 0;
}
//
// Created by mehdi on 6/30/23.
//
#include "GCN_Layer_Demo_Utils.h"
#include "sparse-fusion/Fusion_Utils.h"

using namespace sym_lib;
int main(const int argc, const char *argv[]) {
  TestParameters tp;tp._order_method=SYM_ORDERING::NONE;
  ScheduleParameters sp;
  Stats *stats;
  parse_args(argc, argv, &sp, &tp);
  CSC *aCSC = get_matrix_from_parameter(&tp);
  Dense *features = get_feature_matrix_from_parameter(&tp);
  if(aCSC->m != aCSC->n){
    return -1;
  }
  tp._dim1 = aCSC->m; tp._dim2 = aCSC->n; tp._nnz = aCSC->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);
}

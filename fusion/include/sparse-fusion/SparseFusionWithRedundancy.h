//
// Created by Kazem on 2023-07-24.
//

#ifndef SPARSE_FUSION_SPARSEFUSIONWITHREDUNDANCY_H
#define SPARSE_FUSION_SPARSEFUSIONWITHREDUNDANCY_H

#include "sparse-fusion/SparseFusion.h"

namespace sym_lib{

 class SparseFusionWithRedundancy : public SparseFusion {

  public:
  SparseFusionWithRedundancy(ScheduleParameters *Sp, int LoopCnt):SparseFusion(Sp,LoopCnt){}

  void fuse(int LoopId, CSC *Gi, CSC *Di) override;

  void pairing(int LoopId, CSC *Gi, CSC *Di) override;


 };


 } // namespace sym_lib

#endif //SPARSE_FUSION_SPARSEFUSIONWITHREDUNDANCY_H

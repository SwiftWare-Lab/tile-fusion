//
// Created by Kazem on 2023-07-24.
//

#include "sparse-fusion/SparseFusionWithRedundancy.h"
#include "sparse-fusion/Fusion_Utils.h"

namespace sym_lib{


 int replicateKernelFromKernel(int LNo, int SrcLoopID, int DstLoopID,
                                                CSC *Dm,
                                                std::vector<std::vector<FusedNode*>> &FinalNodeList){
  int height = FinalNodeList.size();
  if(height <= LNo )
   return 0;
   for (int j = 0; j < FinalNodeList[LNo].size(); ++j) {
    std::vector<int> reachedItersList;
    for (int k = 0; k < FinalNodeList[LNo][j]->_list[SrcLoopID].size(); ++k) {
     auto srcIter = FinalNodeList[LNo][j]->_list[SrcLoopID][k];
     for (int cc = 0, jj = Dm->p[srcIter]; jj < Dm->p[srcIter + 1]; ++cc, ++jj){
      auto reachedIter = Dm->i[jj];
      reachedItersList.push_back(reachedIter);
     }
    }
    std::sort(reachedItersList.begin(), reachedItersList.end());
    auto last = std::unique(reachedItersList.begin(), reachedItersList.end());
    reachedItersList.erase(last, reachedItersList.end());
    FinalNodeList[LNo][j]->_list[DstLoopID] = reachedItersList;
   }
  return 1;
 }



 void SparseFusionWithRedundancy::fuse(int LoopId, CSC *Gi, CSC *Di){
   // when the first DAG comes
  if(_final_node_list.empty()){
   _lb_g_prev = 0;
   // applies LBC to the first DAG
   _partitioned_DAG = new DAG();
   // starts with building schedule from loop id 1 (second loop) based on Di
   LBC((CSC*)Gi, Di, _sp, 1, _loop_count, _final_node_list,
       _partitioned_DAG, _vertex_to_part, _part_to_coordinate );
   //_partitioned_DAG = new DAG(_final_node_list.size(), _final_node_list);
   //_partitioned_DAG->print();
  } else {// it is a new coming loop
   pairing(LoopId, Gi, Di);
  }
 }


 void SparseFusionWithRedundancy::pairing(int LoopId, CSC *Gi, CSC *Di){
  // for every fused node of the last loop
  _visited_g_prev_sofar.resize(Gi->m, 0);
  _visited_g_cur_sofar.resize(Gi->m, 0);
  int lastLevel = 0;
  std::fill(_visited_g_cur_sofar.begin(), _visited_g_cur_sofar.end(), 0);
  std::fill(_visited_g_prev_sofar.begin(), _visited_g_prev_sofar.end(), 0);
  // replicate from the create schedule of the second loop
  // it is like fusing LoopId with its next loop
  replicateKernelFromKernel(0,LoopId+1,LoopId, Di, _final_node_list);

 }


} // namespace sym_lib
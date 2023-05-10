//
// Created by kazem on 2023-02-17.
//

#include "sparse-fusion/SparseFusion.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/Utils.h"

namespace sym_lib{


 SparseFusion::SparseFusion(ScheduleParameters *Sp, int LoopCnt): _loop_count(LoopCnt){
  _sp=Sp;
 }

 SparseFusion::~SparseFusion(){
  for (auto & i : _final_node_list) {
   for (auto & j : i) {
    delete j;
   }
  }
  for (auto & i : _cur_node_list) {
   for (auto & j : i) {
    delete j;
   }
  }
  delete _partitioned_DAG;
 }

 void SparseFusion::fuse(int LoopId, CSC *Gi, CSC *Di){
 // when the first DAG comes
  if(_final_node_list.empty()){
   _lb_g_prev = 0;
   // applies LBC to the first DAG
   _partitioned_DAG = new DAG();
   LBC((CSC*)Gi, _sp, 0, _loop_count, _final_node_list,
       _partitioned_DAG, _vertex_to_part, _part_to_coordinate );
   //_partitioned_DAG = new DAG(_final_node_list.size(), _final_node_list);
   //_partitioned_DAG->print();
  } else{// it is a new coming loop
   pairing(LoopId, Gi, Di);
  }
 }



 void SparseFusion::pairing(int LoopId, CSC *Gi, CSC *Di){
  // for every fused node of the last loop
  _visited_g_prev_sofar.resize(Gi->m, 0);
  _visited_g_cur_sofar.resize(Gi->m, 0);
  int lastLevel = 0;
  std::fill(_visited_g_cur_sofar.begin(), _visited_g_cur_sofar.end(), 0);
  std::fill(_visited_g_prev_sofar.begin(), _visited_g_prev_sofar.end(), 0);
  //assert(_cur_node_list.size() == (_ub_g_prev - _lb_g_prev));
  for (int i = 0; i < _part_to_coordinate.size(); ++i) { // iterate over the list of fused node of last kernel
   // do the pairing
   for (int j = 0; j < _part_to_coordinate[i].size(); ++j) {
    int i1 = _part_to_coordinate[i][j].first; int j1 = _part_to_coordinate[i][j].second;
    lastLevel = std::max(lastLevel, i1);
    // pick a fused iteration group
    auto *fn = _final_node_list[i1][j1];
    // set G1 vertices since they are already there
    //std::cout<<"\n";
    for (int k : fn->_list[LoopId-1]) {
     _visited_g_prev_sofar[k] = true;
     //std::cout<< _visited_g_prev_sofar[k] <<",";
    }
    //std::cout<<"\n";
    forward_pairing(Gi, Di, fn->_list[LoopId-1], fn->_list[LoopId],
                    _visited_g_prev_sofar, _visited_g_cur_sofar);
   }
   // update visited arrays for the ith level
   for (int j = 0; j < _part_to_coordinate[i].size(); ++j) {
    int i1 = _part_to_coordinate[i][j].first; int j1 = _part_to_coordinate[i][j].second;
    // pick a fused iteration group
    auto *fn = _final_node_list[i1][j1];
    // mark G2 vertices as visited
    for (int k : fn->_list[LoopId]) {
     _visited_g_cur_sofar[k] = true;
    }
   }
//   auto new_fn = new FusedNode();
//   for (int j = 0; j < _cur_node_list[i].size(); ++j) {
//    // pick a fused iteration group
//    auto fn = _cur_node_list[i][j];
//    forward_pairing(Gi, Di, fn->_list[loop_id], new_fn->_list[loop_id+1],
//                    _visited_g_prev_sofar, _visited_g_cur_sofar);
//    // put the cur node to the final set using the cur_to_final mapping
//    auto coord = _cur_to_final[i][j];
//    //_final_node_list[coord.first][coord.second+1] = new_fn; // expand the list first
//    // update the mapping
//    _cur_to_final[i][j].first++; _cur_to_final[i][j].second++;
//   }
  }
  // For unvisited node, build an unfused schedule of the current DAG

  // sum of unvisited vertices.
  int numUnvisitedVert = 0;
  std::vector<int> unvisitedVert;
  unvisitedVert.reserve(Gi->m);
  // count the number of unvisited vertices in the current DAG
  for (int i = 0; i < Gi->m; ++i) {
   if(!_visited_g_cur_sofar[i]) {
    unvisitedVert.push_back(i);
    numUnvisitedVert++;
   }
  }
  // build an unfused schedule for the unvisited vertices
  std::vector<int> partitionedSet;
  auto numLoops = _final_node_list[0][0]->_num_loops;
  if(numUnvisitedVert > 0) {
   int i1 = lastLevel+1; //_final_node_list.size();
   _final_node_list.resize(i1+1);
   std::vector<double> cost(Gi->m, 1);
   // partition the unvisited vertices into a given part no
   sym_lib::partitionByWeight(numUnvisitedVert,
                              unvisitedVert.data(),
                              cost.data(),
                              _sp->_num_w_partition,
                              NULLPNTR,
                              partitionedSet);
   //_final_node_list[i1].resize(partitionedSet.size());
   for (int i = 0; i < partitionedSet.size()-1; ++i) {
    auto lstSize = partitionedSet[i+1] - partitionedSet[i];
    // taking a portion of unvisitedVert and point to the curList
    auto curList = unvisitedVert.begin() + partitionedSet[i];
    auto *fn = new FusedNode(numLoops, i1, lstSize,
                             curList.operator->(), i);
    _final_node_list[i1].push_back(fn);
   }
   //_final_node_list.push_back(unfused_node);
  }

  // update the current from the updated mapping
//  for (int i = 0; i < _cur_to_final.size(); ++i) {
//   for (int j = 0; j < _cur_to_final[i].size(); ++j) {
//    auto coord = _cur_to_final[i][j];
//    _cur_node_list[i][j] = _final_node_list[coord.first][coord.second];
//   }
//  }

 }

 void SparseFusion::merge_pairs(){

 }

 void SparseFusion::build_set(){

 }



 void SparseFusion::print_final_list(){
  std::cout<<"Fused Set: \n";
  for (int i = 0; i < _final_node_list.size(); ++i) {
   for (int j = 0; j < _final_node_list[i].size(); ++j) {
    for (int k = 0; k < _final_node_list[i][j]->_list.size(); ++k) {
     for (int l = 0; l < _final_node_list[i][j]->_list[k].size(); ++l) {
      std::cout<<"( "<<i<<" , "<<j<<" ), Loop = "<<k <<" " <<_final_node_list[i][j]->_list[k][l]<<" , ";
     }
    }
    std::cout<<"\n";
   }
  }
 }

 MultiDimensionalSet *SparseFusion::getFusedCompressed() {
  auto *ret = new MultiDimensionalSet(_final_node_list);
  return ret;
 }

} // end namespace sym_lib
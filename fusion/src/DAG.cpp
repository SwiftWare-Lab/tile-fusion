//
// Created by Kazem on 2023-02-19.
//

#include <iostream>

#include "sparse-fusion/Fusion_Defs.h"
#include "sparse-fusion/DAG.h"

namespace sym_lib{

 DAG::DAG(int nCW, const std::vector<std::vector<FusedNode*>>& cur_node_list){

/*  _number_vertex= cur_node_list.size()*nCW;
  _part_vertex_list.resize(_number_vertex);
  auto vertex_added = new bool[_number_vertex]();
  _cdag_to_coordinate.resize(_number_vertex);
  for (int i = 0; i < cur_node_list.size(); ++i) {
   for (int j = 0; j < cur_node_list[i].size(); ++j) {
    int cur_ver = cur_node_list[i][j]->_vertex_id;
    _cdag_to_coordinate[cur_ver] = std::make_pair(i,j);
    // see the destinations of vertices in the current fused node
    for (int k = 0; k < cur_node_list[i][j]->_list.size(); ++k) {
     int cur_v = cur_node_list[i][j]->_list[j];
     int cur_vid = cur_node_list[i][j]->_vertex_id;
     // if not visited, otherwise it is already there
     if(!vertex_added[cur_vid]){
      vertex_added[cur_vid] = true;
      _part_vertex_list[cur_v].push_back(cur_vid);
     }
    }
   }
  }*/
 }

 DAG::DAG(int n_v){
  _part_vertex_list.resize(n_v);// resize based the given estimate of number of parts
 }

 void inline DAG::add_vertex(int src, int dst) {
  if(src >= _part_vertex_list.size()){
   _part_vertex_list.emplace_back();
  }
  _part_vertex_list[src].push_back(dst);
  _number_vertex++;
 }

 void DAG::print(){
  std::cout<<"DAG: \n";
  for (int i = 0; i < _part_vertex_list.size(); ++i) {
   std::cout<<i<<" : ";
   for (int j = 0; j < _part_vertex_list[i].size(); ++j) {
    std::cout<<_part_vertex_list[i][j]<<",";
   }
   std::cout<<"\n";
  }
 }

 void DAG::build_DAG_from_mapping_CSC(const int n_parts, const int n_w_parts, const int *v_to_part, const CSC* G){
  std::vector<bool> destination(n_parts);
  _part_vertex_bool.resize(n_parts);
  for (int i = 0; i < n_parts; ++i) {
   _part_vertex_bool[i].resize(n_w_parts);
  }
  // set dests to true
  for (int i = 0; i < G->m; ++i) {
   int src = v_to_part[i];
   for (int j = G->p[i]; j < G->p[i+1]; ++j) {
    int dst = v_to_part[G->i[j]];
    _part_vertex_bool[src][dst] = true;
   }
  }
  for (int i = 0; i < n_parts; ++i) {
   for (int j = 0; j < n_w_parts; ++j) {
    if(_part_vertex_bool[i][j]){
     add_vertex(i, j);
    }
   }
  }
 }
}
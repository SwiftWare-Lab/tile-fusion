//
// Created by kazem on 2023-02-17.
//

#ifndef FUSION_DAG_H
#define FUSION_DAG_H

#include <vector>

#include "aggregation/def.h"

namespace sym_lib{
 // DAG of fused nodes
 class DAG{
  // _part_vertex_list[i].size is the number of outgoing edges in i
  std::vector<std::vector<int>> _part_vertex_list;

  // equal to _part_vertex_list.size()
  int _number_vertex{};

  // mapping between DAG vertices and the nodes in the FusedNode list
  std::vector<std::pair<int,int>> _cdag_to_coordinate;

 public:
  std::vector<std::vector<bool>> _part_vertex_bool;
  /// populates the coarsened DAG from the created schedule
  /// \param nCW
  /// \param cur_node_list
  DAG(int nCW, const std::vector<std::vector<FusedNode*>>& cur_node_list);

  DAG(int number_ver);

  DAG(){};

  void update(int n_id);

  void inline add_vertex(int src, int dst);

  CSC* to_csc();

  void print();

  void build_DAG_from_mapping_CSC(int n_parts, int n_w_parts, const int *v_to_part, const CSC* G);
 };

}
#endif //FUSION_DAG_H

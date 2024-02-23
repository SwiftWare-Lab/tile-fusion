//
// Created by kazem on 2023-02-17.
//

#ifndef FUSION_SPARSEFUSION_H
#define FUSION_SPARSEFUSION_H
#include <vector>

#include "aggregation/def.h"
#include "MultiDimensionalSet.h"
#include "sparse-fusion/DAG.h"
#include "sparse-fusion/Fusion_Defs.h"
#include <tuple>

namespace sym_lib{

  enum PackingType{
    Interleaved,
    Separated,
    Tiled
  };

  enum TilingMethod{
    Fixed,
    Variable
  };

  struct SparsityProfileInfo {
    int TotalReuseC{}, RedundantIterations{}, RedundantOperations{},
        UniqueIterations{};

    std::tuple<std::string,std::string>  printCSV(bool Header){
      std::string headerText, row;
      if(Header){
        headerText = "TotalReuseC,RedundantIterations,RedundantOperations,";
      }
      row = std::to_string(TotalReuseC) + "," +
            std::to_string(RedundantIterations) + "," +
            std::to_string(RedundantOperations) + ",";
      return std::make_tuple(headerText,row);
    }
  };

 class SparseFusion {
  protected:
  // List of fused node
  std::vector<std::vector<FusedNode*>> _cur_node_list;
  //
  std::vector<std::vector<std::pair<int,int>>> _cur_to_final;
  // Final node list
  std::vector<std::vector<FusedNode*>> _final_node_list;



  // mappings
  std::vector<int> _vertex_to_part;
  // Coordinate of partitions of last DAG
  std::vector<std::vector<std::pair<int,int>>> _part_to_coordinate;


  // list of pairs
  std::vector<int> _pair_list;

  // ID of nodes in node_list that has last graph
  int _lb_g_prev, _ub_g_prev;
  std::vector<int> _vertex_node_g1;

  // Visited vertices in G1 so far
  int dim1_g_prev, dim1_g_cur;
  std::vector<bool> _visited_g_prev_sofar, _visited_g_cur_sofar;

  // The infered partitioned DAG
  DAG *_partitioned_DAG;

  ScheduleParameters *_sp;

  // total number of loops to be fused.
  int _loop_count{};

 public:

  explicit SparseFusion(ScheduleParameters *Sp, int LoopCnt);

  virtual void fuse(int LoopId, CSC *Gi, CSC *Di);

  MultiDimensionalSet *getFusedCompressed(int PT);

  virtual void pairing(int LoopId, CSC *Gi, CSC *Di);

  void merge_pairs();

  void build_set();

  void print_final_list();

  // get final node list
  std::vector<std::vector<FusedNode*>> getFinalNodeList(){
    return _final_node_list;
  }

  SparsityProfileInfo measureReuse(CSC *Gi);
  void measureRedundancy(CSC *Gi, SparsityProfileInfo &spInfo);

  ~SparseFusion();




 };

}


#endif //FUSION_SPARSEFUSION_H

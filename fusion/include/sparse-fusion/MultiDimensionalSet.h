//
// Created by Kazem on 2023-05-07.
//

#ifndef SPARSE_FUSION_MULTIDIMENSIONALSET_H
#define SPARSE_FUSION_MULTIDIMENSIONALSET_H

#define NULLPNTR nullptr
#include "sparse-fusion/Fusion_Defs.h"
#include "aggregation/def.h"
#include <vector>

namespace sym_lib{

 struct MultiDimensionalSet {
  int n1_{}, n2_{}, n3_{};
  int d_{};
  int *ptr1_{}, *ptr2_{};
  int *id_{}, *type_{};
  int *w_par_type_{}; // the type of each w-partition
  int *ker_begin_{}; // ker_begin[i][j], starting index of kernel j in w-partition i

  // redundancy mode
  bool *is_redundancy_{};
  int *map_redundancy_{};

  MultiDimensionalSet() : ptr1_(NULLPNTR), ptr2_(NULLPNTR), id_(NULLPNTR),
                          type_(NULLPNTR), w_par_type_(NULLPNTR), ker_begin_(NULLPNTR),
                          is_redundancy_(NULLPNTR), map_redundancy_(NULLPNTR){};

  MultiDimensionalSet(const int n_hl, int *hl_ptr, int *par_ptr,
                      int *partition);

  explicit MultiDimensionalSet(int N);

  /*
   * For 2D
   */
  MultiDimensionalSet(int N1, int N3);

  /*
   * for 3D
   */
  MultiDimensionalSet(int N1, int N2, int N3);

/*
 * for 3D with depth
 */
  MultiDimensionalSet(int N1, int N2, int N3, int D, int Red);

  MultiDimensionalSet(
    const std::vector<std::vector<FusedNode*>> &FusedSchedule);
  MultiDimensionalSet(
      const std::vector<std::vector<FusedNode*>> &FusedSchedule,
      int PerPartition);

  ~MultiDimensionalSet() ;

  int getNumberOfFusedNodes();
  void print();
  void print_3d();
  int *build_node_to_level();
 };

}

#endif //SPARSE_FUSION_MULTIDIMENSIONALSET_H

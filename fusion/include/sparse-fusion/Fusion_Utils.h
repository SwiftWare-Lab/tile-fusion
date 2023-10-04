//
// Created by Kazem on 2023-02-18.
//

#ifndef SPARSE_FUSION_FUSION_UTILS_H
#define SPARSE_FUSION_FUSION_UTILS_H

#include "sparse-fusion/Fusion_Defs.h"
#include "sparse-fusion/DAG.h"

#include "SparseFusion.h"
#include "aggregation/def.h"

namespace sym_lib{
 /*
 * Processes the argument and fill out the schedule parameters
  * TODO: Mehdi: add arg details here and delete the comments in the .cpp file
 */
 void parse_args(const int argc, const char **argv,  ScheduleParameters *sp,
                TestParameters *tp);

  /// Reorders the matrix and retunrs reorderd and its transpose reordered
  /// \param L1_csc
  /// \param mat_vec
  /// \return
  int get_reorderd_matrix(const CSC *L1_csc, std::vector<CSC*>& mat_vec);

  /// Generates a random matrix or read it from file
  /// \param tp
  /// \return
  CSC* get_matrix_from_parameter(const TestParameters *tp);

  Dense* random_dense_matrix(int M, int N);

  Dense* get_feature_matrix_from_parameter(const TestParameters *tp, int NumOfNodes);
 // starts from in_set in G1 and reaches to all unvisited vertices in G2
// G1 -> G2 , D is transpose of dependence
 void forward_pairing(CSC *G2, CSC *D, const std::vector<int>& in_set, std::vector<int>& out_set,
                      const std::vector<bool>& visited_g2, const std::vector<bool>& visited_g1);

  /// Computes LBC of G, from loop with loop_id and storeds it into cur_node_list (reserved with the hint)
  /// \param G
  /// \param Di
  /// \param sp
  /// \param loop_id
  /// \param hint_tot_loops
  /// \param cur_node_list
  /// \return
  int LBC(const CSC *G, const CSC *Di, ScheduleParameters* sp, int loop_id, int hint_tot_loops,
          std::vector<std::vector<FusedNode*>>& cur_node_list, DAG *out_dag,
          std::vector<int>& v_to_part, std::vector<std::vector<std::pair<int,int>>>& part_to_coord);

  /// Get where each vertex is mapped in the fusedNode schedule
  /// \param nIterations
  /// \param cur_node_list
  /// \param iteration_to_part
  void get_iteration_schedule_mapping(int nIterations,
                                      const std::vector<std::vector<FusedNode*>>& cur_node_list,
                                      std::vector<std::pair<int,int>>& iteration_to_part);

  /// balancing using redundant computation
  /// \param FinalNodeList
  /// \param UpdatedNodeList
  /// \param Dm
  /// \param BalancedRatio
  void BalanceWithRedundantComputation(const std::vector<std::vector<FusedNode*>> &FinalNodeList,
                                       std::vector<std::vector<FusedNode*>> &UpdatedNodeList,
                                       const CSC *Dm,
                                       double BalancedRatio
  );

  /// measure the redundancy of the schedule
  /// \param Gi
  /// \param Spi
  /// \param FinalNodeList
  void measureRedundancy(sym_lib::CSC *Gi, sym_lib::SparsityProfileInfo &Spi,
                         const std::vector<std::vector<FusedNode*>> &FinalNodeList);


} // End of namespace sym_lib


#endif //SPARSE_FUSION_FUSION_UTILS_H

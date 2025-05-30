//
// Created by Kazem on 2023-02-18.
//

#ifndef SPARSE_FUSION_FUSION_DEFS_H
#define SPARSE_FUSION_FUSION_DEFS_H

#include <string>
#include <vector>

namespace sym_lib {
enum SYM_ORDERING { NONE, SYM_METIS, SYM_AMD, SYM_SCOTCH };

enum SeedPartType { CONSECUTIVE, BFS };

std::string get_ordering_string(SYM_ORDERING symOrdering);

/*
 * Scheduling-related parameters
 */
struct ScheduleParameters {
  int _lbc_agg, _lbc_initial_cut, _num_w_partition,
      IterPerPartition;   // aggregation params
  int _min_workload_size; // min workload size to run in parallel for
                          // tiledFusedCSCCombined
  int _num_threads;
  int TileM{}, TileN{}, TileK{};
  SeedPartType SeedPartitioningParallelism{};

  ScheduleParameters() : _lbc_agg(2), _lbc_initial_cut(2), _num_threads(1),
  TileK(1), TileM(1), TileN(1), IterPerPartition(1), _min_workload_size(1){
    _num_w_partition = _num_threads;
  }

  explicit ScheduleParameters(int nt) : ScheduleParameters() {
    _num_threads = nt;
  }
  ScheduleParameters(ScheduleParameters &sp){
    this->_num_threads=sp._num_threads;
    this->IterPerPartition=sp.IterPerPartition;
    this->TileM = sp.TileM;
    this->TileN = sp.TileN;
    this->TileK = sp.TileK;
    this->_min_workload_size = sp._min_workload_size;
    this->_lbc_agg = sp._lbc_agg;
    this->_lbc_initial_cut = sp._lbc_initial_cut;
    this->_num_w_partition = sp._num_w_partition;
    this->SeedPartitioningParallelism = sp.SeedPartitioningParallelism;
  }

  /*
   * Prints header and info in csv format
   */
  std::tuple<std::string, std::string> print_csv(bool header = false) const;
};

/*
 * Holding the test parameters
 */
struct TestParameters {
  SYM_ORDERING _order_method; // type of reordering
  std::string _matrix_name{}, _matrix_path{}, _feature_matrix_path{},
      _weight1_matrix_path{}, _weight2_matrix_path{}, _result_matrix_path{}
      , e2e_data_path{}; // matrix name and path
  float _sampling_ratio{};
  std::string expariment_name{};
  std::string _mode{};                //"Random" or "MTX"
  std::string _gnn_parameters_mode{}; // "Random" or "MTX": defines whether read
                                      // gnn parameter matrices(features,
                                      // weights) from file or generate randomly.
  std::string _algorithm_choice{};
  double _density{}, _dim1{}, _dim2{}, _nnz{}; // for random mode
  bool print_header{};
  int _b_cols{};    // in gnn experiments bcols is regarded as feature dimension
  int _embed_dim{}; // embed_dim only is used in gnn experiments and is regarded
                    // as hidden dimension
  TestParameters() { _mode = "Random"; }

  /*
   * Prints header and info in csv format
   */
  std::tuple<std::string, std::string> print_csv(bool header = false) const;
};

/*
 struct HWaveFront{
  int _n_coarsened_wf{}, _num_vertices{};
  std::vector<int> _c_wf_pntr, _w_part_pntr, _vertices;
  std::vector<int> _vertex_to_coord;

  HWaveFront()=default;

  /// Builds the set from the LBC arrays.
  /// \param m number of vertices
  /// \param final_level_no
  /// \param fina_level_ptr
  /// \param final_part_ptr
  /// \param final_node_ptr
  HWaveFront(const int m, const int final_level_no,
             const int* fina_level_ptr,
             const int* final_part_ptr, const int*final_node_ptr);
 };*/

struct FusedNode {
  // list[i][j] show jth vertex from loop i
  // all vertices can run in parallel with others, thread safe
  std::vector<std::vector<int>> _list; // list of vertices
  std::vector<int> _kernel_ID;         // not used yet
  int _num_loops{};
  int _vertex_id{}; // id in the partitioned DAG
  bool _is_redundant{};

  FusedNode() = default;
  FusedNode(int loop_no, int ID, int lst_size, const int *lst, int v_no);
  FusedNode(const FusedNode &other);
};

} // namespace sym_lib

#endif // SPARSE_FUSION_FUSION_DEFS_H

//
// Created by Kazem on 2023-02-18.
//

#include "sparse-fusion/Fusion_Defs.h"
#include <tuple>


namespace sym_lib{
 std::string get_ordering_string(SYM_ORDERING symOrdering){
  switch (symOrdering) {
   case SYM_ORDERING::NONE:
    return "NONE";
   case SYM_ORDERING::SYM_METIS:
    return "METIS";
   case SYM_ORDERING::SYM_AMD:
    return "AMD";
   case SYM_ORDERING::SYM_SCOTCH:
    return "SCOTCH";
   default:
    return "UnIdentified";
  }
 }

 std::tuple<std::string,std::string> TestParameters::print_csv(bool header) const {
  std::string header_text, row;
  if(header){
   header_text  = "MatrixName,Density,nRows,nCols,NNZ,Mode,Ordering,Algorithm,bCols,";
  }
  row = _matrix_name+","+ std::to_string(_density)+","+ std::to_string(_dim1)+","+
    std::to_string(_dim2)+","+ std::to_string(_nnz)+","+_mode+","+
    get_ordering_string(_order_method)+","+_algorithm_choice+","
      +std::to_string(_b_cols)+",";
  return std::make_tuple(header_text, row);
 }

 std::tuple<std::string,std::string> ScheduleParameters::print_csv(bool header) const{
  std::string header_text, row;
  if(header){
   header_text = "nThreads,LBC Agg,LBC InitialCut,LBC WPART,";
  }
  row = std::to_string(_num_threads) + "," + std::to_string(_lbc_agg) +"," +
    std::to_string(_lbc_initial_cut) + "," + std::to_string(_num_w_partition) +",";
  return std::make_tuple(header_text, row);
 }

 FusedNode::FusedNode(const int loop_no, const int ID,
                      const int lst_size, const int *lst,
                      int v_no) {
  _num_loops = loop_no;
  _kernel_ID.push_back(ID);
  _list.resize(loop_no);
  _list[ID].reserve(lst_size);
  std::copy(lst, lst+lst_size, std::back_inserter(_list[ID]));
  _vertex_id = v_no;
 }

/* HWaveFront::HWaveFront(const int m, const int final_level_no,
                        const int* fina_level_ptr,
                        const int* final_part_ptr, const int*final_node_ptr) {
  _n_coarsened_wf = final_level_no; _num_vertices = m;
  _c_wf_pntr.resize(_n_coarsened_wf, 0);
  int num_w_part = fina_level_ptr[final_level_no-1];
  _w_part_pntr.resize(num_w_part); _vertices.resize(_num_vertices);
  _vertex_to_coord.resize(_num_vertices);
  _c_wf_pntr.insert(_c_wf_pntr.end(), fina_level_ptr, fina_level_ptr+_n_coarsened_wf);
  _w_part_pntr.insert(_w_part_pntr.end(), final_part_ptr, final_part_ptr+num_w_part);
  _vertices.insert(_vertices.end(), final_node_ptr, final_node_ptr+m);
  //_w_part_pntr
//  for (int i = 0; i < _n_coarsened_wf; ++i) {
//   for (int j = fina_level_ptr[i]; j < fina_level_ptr[i + 1]; ++j) {
//    for (int k = final_part_ptr[j]; k < final_part_ptr[j + 1]; ++k) {
//
//    }
//   }
//  }
 }*/

}
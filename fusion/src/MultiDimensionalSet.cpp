//
// Created by Kazem on 2023-05-07.
//

#include <iostream>
#include "sparse-fusion/MultiDimensionalSet.h"

namespace sym_lib {

 MultiDimensionalSet::MultiDimensionalSet(const int n_hl, int *hl_ptr,
                                          int *par_ptr, int *partition):n1_
                                                                          (n_hl), ptr1_(hl_ptr), ptr2_
                                                                          (par_ptr), id_(partition) {
  n2_=0;
  type_= NULLPNTR;
  w_par_type_ = NULLPNTR;
  ker_begin_ = NULLPNTR;
  is_redundancy_=NULLPNTR;
  map_redundancy_=NULLPNTR;
 }

 MultiDimensionalSet::MultiDimensionalSet(int N) {
  n1_ = n2_ = N;
  ptr2_ = NULLPNTR;
  ptr1_ = new int[n1_ + 1]();
  id_ = new int[n1_];
  type_= new int[n1_];
  w_par_type_ = NULLPNTR;
  ker_begin_ = NULLPNTR;
  is_redundancy_=NULLPNTR;
  map_redundancy_=NULLPNTR;
 }

 MultiDimensionalSet::MultiDimensionalSet(int N1, int N3) {
  n1_ =  N1;
  n2_ = 0;
  n3_ = N3;
  ptr2_ = NULLPNTR;
  ptr1_ = new int[n1_ + 1]();
  id_ = new int[n3_];
  type_= new int[n3_];
  w_par_type_ = NULLPNTR;
  ker_begin_ = NULLPNTR;
  is_redundancy_=NULLPNTR;
  map_redundancy_=NULLPNTR;
 }

 MultiDimensionalSet::MultiDimensionalSet(int N1, int N2, int N3):MultiDimensionalSet() {
  n1_ = N1;
  n2_ = N2;
  n3_ = N3;
  ptr1_ = new int[n1_ + 1]();
  ptr2_ = new int[n2_ + 1]();
  id_ = new int[n3_];
  type_= new int[n3_];
  w_par_type_ = NULLPNTR;
  ker_begin_ = NULLPNTR;
  is_redundancy_=NULLPNTR;
  map_redundancy_=NULLPNTR;
 }

 MultiDimensionalSet::MultiDimensionalSet(int N1, int N2, int N3,
                                          int D, int Red) {
  n1_ = N1;
  n2_ = N2;
  n3_ = N3;
  d_ = D;
  ptr1_ = new int[n1_ + 1]();
  ptr2_ = new int[n2_ + 1]();
  id_ = new int[n3_];
  type_= new int[n3_];
  w_par_type_ = new int[n1_*n2_]();
  ker_begin_ = new int[n2_*d_]();
  is_redundancy_ = new bool[d_]();
  map_redundancy_ = new int[Red]();
 }

 MultiDimensionalSet::MultiDimensionalSet(
   const std::vector<std::vector<FusedNode*>> &FusedSchedule){
  int totalNode = 0, height = FusedSchedule.size(), width = 0, partNo = 0;
  for (int i = 0; i < FusedSchedule.size(); ++i) {
   width = std::max(width, (int) FusedSchedule[i].size());
   partNo += FusedSchedule[i].size();
   for (int j = 0; j < FusedSchedule[i].size(); ++j) {
    for (int k = 0; k < FusedSchedule[i][j]->_list.size(); ++k) {
     totalNode+=FusedSchedule[i][j]->_list[k].size();
    }
   }
  }
  n1_ = height;
  n2_ = partNo;
  n3_ = totalNode;
  ptr1_ = new int[n1_ + 1]();
  ptr2_ = new int[n2_ + 1]();
  id_ = new int[n3_];
  type_= new int[n3_];
  w_par_type_ = NULLPNTR;
  ker_begin_ = NULLPNTR;
  is_redundancy_=NULLPNTR;
  map_redundancy_=NULLPNTR;

  int cnt = 0;
  int cntW = 0;
  ptr1_[0]=0;
  ptr2_[0]=0;
  for (int l = 0; l < FusedSchedule.size(); ++l) { // over levels
   for (int i = 0; i < FusedSchedule[l].size() ; ++i) {
    //reading each w-partition
    for (int j = 0; j < FusedSchedule[l][i]->_list.size(); ++j) { // each fused loop
     for (int k = 0; k < FusedSchedule[l][i]->_list[j].size(); ++k) { // each iteration
      id_[cnt] = FusedSchedule[l][i]->_list[j][k];
      type_[cnt] = j;
      cnt++;
     }
    }
    cntW++;
    ptr2_[cntW] = cnt;
   }
   ptr1_[l+1] = cntW;
  }
 }

 MultiDimensionalSet::~MultiDimensionalSet() {
  delete []ptr1_;
  delete []ptr2_;
  delete []id_;
  delete []type_;
  delete []w_par_type_;
  delete []ker_begin_;
  delete []is_redundancy_;
  delete []map_redundancy_;
 }

 void MultiDimensionalSet::print() {
  for (int i = 0; i < n1_; ++i) {
   for (int j = ptr1_[i]; j < ptr1_[i + 1]; ++j) {
    std::cout << "(" << id_[j] << "," << (type_? type_[j]:0) << "),";
   }
   std::cout << ";\n";
  }
 }

 void MultiDimensionalSet::print_3d() {
  for (int i = 0; i < n1_; ++i) {
   for (int j = ptr1_[i]; j < ptr1_[i + 1]; ++j) {
    for (int k = ptr2_[j]; k < ptr2_[j + 1]; ++k) {
     std::cout << "(" << id_[k] << "," << (type_? type_[k]:0)<< "),";
    }
    std::cout<<"; ";
   }
   std::cout << "\n";
  }
 }

 int *MultiDimensionalSet::build_node_to_level(){
  int n = ptr1_[n1_];
  int *n2l = new int[n]();
  for (int i = 0; i < n1_; ++i) {
   for (int j = ptr1_[i]; j < ptr1_[i + 1]; ++j) {
    n2l[id_[j]] = i;
   }
  }
  return n2l;
 }

}
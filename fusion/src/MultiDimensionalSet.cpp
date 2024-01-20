//
// Created by Kazem on 2023-05-07.
//

#include "sparse-fusion/MultiDimensionalSet.h"
#include <iostream>

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


 MultiDimensionalSet::MultiDimensionalSet(
     const std::vector<std::vector<FusedNode*>> &FusedSchedule,
     int PerPartition){
  int totalNode = 0, height = FusedSchedule.size(), width = 0, partNo = 0;
  for (int i = 0; i < FusedSchedule.size(); ++i) {
   width = std::max(width, (int) FusedSchedule[i].size());
   partNo += FusedSchedule[i].size();
   for (int j = 0; j < FusedSchedule[i].size(); ++j) {
    for (int k = 0; k < FusedSchedule[i][j]->_list.size(); ++k) { // loop id
     totalNode+=FusedSchedule[i][j]->_list[k].size();
    }
   }
  }
  auto numLoops = FusedSchedule[0][0]->_num_loops;
  n1_ = height;
  n2_ = partNo;
  n3_ = totalNode;
  d_ = numLoops;
  ptr1_ = new int[n1_ + 1]();
  ptr2_ = new int[n2_ + 1]();
  id_ = new int[n3_];
  type_= new int[n3_];
  w_par_type_ = NULLPNTR;
  is_redundancy_=NULLPNTR;
  map_redundancy_=NULLPNTR;
  ker_begin_ = new int[n2_*d_]();

  int cnt = 0;
  int cntW = 0;
  ptr1_[0]=0;
  ptr2_[0]=0;
  for (int l = 0; l < n1_; ++l) { // over levels
   for (int i = 0; i < FusedSchedule[l].size() ; ++i) {
    //reading each w-partition
    int cKerCnt = 0;
    for (int j = 0; j < FusedSchedule[l][i]->_list.size(); ++j) {// for each loop
     int bnd2 = (int)  FusedSchedule[l][i]->_list[j].size();
     for (int k = 0; k < bnd2; ++k) {
      id_[cnt] = FusedSchedule[l][i]->_list[j][k];
      type_[cnt] = j;
      cnt++;
     }
     ker_begin_[cntW*d_ + j] = cnt;
     if(! FusedSchedule[l][i]->_list[j].empty() )
      cKerCnt++;
    }
    cntW++;
    ptr2_[cntW] = cnt;
   }
   ptr1_[l+1] = cntW;
  }
  //print();
  // copy redundancy part
//  if(map_redundancy_){
//   for (int m = 0; m < space_; ++m) {
//    compressed_level_set->map_redundancy_[m] = map_redundancy_[m];
//   }
//   for (int n = 0; n < depth_; ++n) {
//    compressed_level_set->is_redundancy_[n] = redundant_kernels_[n];
//   }
//  }
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
  std::cout << "\n";
  if(ker_begin_){
    for (int k = 0; k < n2_; ++k) {
      for (int i = 0; i < d_; ++i) {
      std::cout<<ker_begin_[k*d_+i]<<",";
      }
      std::cout<<"\n";
    }
  }
  std::cout << "\n";
 }

 void MultiDimensionalSet::print_3d() {
  int fused_counter = 0;
  int counter = 0;
  for (int i = 0; i < n1_; ++i) {
   for (int j = ptr1_[i]; j < ptr1_[i + 1]; ++j) {
    for (int k = ptr2_[j]; k < ptr2_[j + 1]; ++k) {
      if (i == 0 && type_[k] == 1){
       fused_counter += 1;
      }
      if(type_[k] == 1)
        counter += 1;
     std::cout << "(" << id_[k] << "," << (type_? type_[k]:0)<< "),";
    }
    std::cout<<"; ";
   }
   std::cout << "\n";
  }
  std:: cout << "number of second loop iterations: " << counter << std::endl;
  std:: cout << "number of second loop fused iterations: " << fused_counter << std::endl;
  std:: cout << "fused ratio: " << double(fused_counter)/counter << std::endl;
 }

 int MultiDimensionalSet::getNumberOfFusedNodes() {
   int fusedCounter = 0;
   for (int i = 0; i < n1_; ++i) {
     for (int j = ptr1_[i]; j < ptr1_[i + 1]; ++j) {
       for (int k = ptr2_[j]; k < ptr2_[j + 1]; ++k) {
         if (i == 0 && type_[k] == 1){
           fusedCounter += 1;
         }
       }
     }
   }
   return fusedCounter;
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
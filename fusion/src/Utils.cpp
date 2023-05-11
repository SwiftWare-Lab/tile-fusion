//
// Created by kazem on 02/05/23.
//

#include <iostream>
#include <cmath>
#include <vector>

namespace sym_lib{

 /// Calculates the summation of a vector
 /// \tparam type
 /// \param n
 /// \param vec
 /// \return
 template<class type> type sum_vector(int n, type *vec){
  double sum = 0;
  for (int i = 0; i < n; ++i) sum += vec[i];
  return sum;
 }

 /// Partitions the set into n_parts based on the weight of each element
 void partitionByWeight(int N,const int *Set, const double *Weight,
                          int NParts,
                          double *TargetWeight,
                          std::vector<int> &Indices){
  double *evenWeight;
  if(TargetWeight)
   evenWeight = TargetWeight;
  else{
   evenWeight = new double[NParts];
   double evenW = std::ceil(sum_vector(N, Weight) / NParts);
   std::fill_n(evenWeight, NParts, evenW);
  }
  //int *indices = new int[NParts+1]();
  Indices.resize(NParts+1);
  int j = 0;
  for (int i = 0; i < NParts; ++i) {
   double cWgt = 0;
   while (cWgt < evenWeight[i] && j < N){
    int cN = Set[j];
    cWgt += Weight[cN];
    j++;
   }
   Indices[i+1] = j;
  }
  if(!TargetWeight)
   delete []evenWeight;
 }


} // namespace sym_lib
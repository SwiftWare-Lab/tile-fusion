//
// Created by mehdi on 6/27/23.
//


#ifndef SPARSE_FUSION_GCNCONV_H
#define SPARSE_FUSION_GCNCONV_H

#include "aggregation/def.h"
namespace sym_lib {
namespace gnn {
class GCNConv {
private:
  sym_lib::CSR* AdjMatrix;
  size_t InputNum;
  size_t OutputNum;
  float * Weight;
  float *Output;
  float *vecMatMul(int M, int N, float *Vec, float *Mat);
  void aggregateMessage(int Dim, float *Messages, float *NeighborMessage);
  void normalizeMessage(int Dim, float DegI, float DegJ,
                            float *NeighborMessage);

public:
  GCNConv(CSR* AdjMatrix, float *Output, float *Weight, size_t InputNum,
          size_t OutputNum);
  void forward(float *Features);
};
} // namespace gnn
} // namespace sym_lib
#endif // SPARSE_FUSION_GCNCONV_H

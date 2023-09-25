//
// Created by mehdi on 6/27/23.
//


#ifndef SPARSE_FUSION_GCNCONV_H
#define SPARSE_FUSION_GCNCONV_H

#include "aggregation/def.h"
#include <set>
namespace sym_lib {
namespace gnn {

class GCNConv{
public:
virtual void forward(double *Features, std::vector<int> Mask){
}
};
class GCNConvSequential: public GCNConv {
protected:
  sym_lib::CSR* AdjMatrix;
  size_t InputNum;
  size_t OutputNum;
  double *Weight;
  double *Output;
  void vecMatMul(int M, int N, double *Vec, double *Mat, double* result);
  void aggregateMessage(int Dim, double *Messages, double *NeighborMessage);
  void normalizeMessage(int Dim, double DegI, double DegJ,
                            double *NeighborMessage);

public:
  GCNConvSequential(CSR* AdjMatrix, double *Output, double *Weight, size_t InputNum,
          size_t OutputNum);
  void forward(double *Features, std::vector<int> Mask) override;
};

class GCNConvParallel : public GCNConvSequential {
protected:
  int NThreads;
public:
  GCNConvParallel(CSR* AdjMatrix, double *Output, double *Weight, size_t InputNum,
          size_t OutputNum, int NThreads);
  void forward(double *Features, std::vector<int> Mask) override;
};
class GCNConvFused : public GCNConvParallel {
protected:
  int NThreads;
public:
  GCNConvFused(CSR* AdjMatrix, double *Output, double *Weight, size_t InputNum,
               size_t OutputNum, int NThreads);
  void forward(double *Features, std::vector<int> Mask) override;
};

} // namespace gnn
} // namespace sym_lib

#endif // SPARSE_FUSION_GCNCONV_H

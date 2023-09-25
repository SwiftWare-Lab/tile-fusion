//
// Created by mehdi on 6/27/23.
//

#include "sparse-fusion/GCNConv.h"
#include <math.h>
namespace sym_lib {
namespace gnn {
GCNConvSequential::GCNConvSequential(CSR *AdjMatrix, double *Output,
                                     double *Weight, size_t InputNum,
                                     size_t OutputNum)
    : AdjMatrix(AdjMatrix), Output(Output), Weight(Weight), InputNum(InputNum),
      OutputNum(OutputNum) {}

void GCNConvSequential::forward(double *Features, std::vector<int> mask) {
  int *Ap = AdjMatrix->p;
  int *Ai = AdjMatrix->i;
  double *Ax = AdjMatrix->x;
  double degrees[AdjMatrix->m];
  for (int i = 0; i < AdjMatrix->m; i++) {
    degrees[i] = 0;
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      degrees[i] += 1;
    }
  }
  double *neighborMessage = new double[OutputNum];
  for (auto i : mask) {
    double *messages = Output + OutputNum * i;
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      int n = Ai[j];
      vecMatMul(this->InputNum, this->OutputNum,
                Features + (n * this->InputNum), this->Weight, neighborMessage);
      normalizeMessage(this->OutputNum, degrees[i], degrees[Ai[j]],
                       neighborMessage);
      aggregateMessage(this->OutputNum, messages, neighborMessage);
    }
  }
  delete[] neighborMessage;
}

void GCNConvSequential::vecMatMul(int M, int N, double *Vec, double *Mat,
                                  double *result) {
  for (int j = 0; j < N; j++) {
    result[j] = 0;
    for (int i = 0; i < M; i++) {
      result[j] += Vec[i] * Mat[i * N + j];
    }
  }
}

void GCNConvSequential::aggregateMessage(int Dim, double *Messages,
                                         double *NeighborMessage) {
  for (int i = 0; i < Dim; i++) {
    Messages[i] += NeighborMessage[i];
  }
}

void GCNConvSequential::normalizeMessage(int Dim, double DegI, double DegJ,
                                         double *NeighborMessage) {
  for (int i = 0; i < Dim; i++) {
    NeighborMessage[i] = NeighborMessage[i] / sqrt(DegI * DegJ);
  }
}

GCNConvParallel::GCNConvParallel(CSR *AdjMatrix, double *Output, double *Weight,
                                 size_t InputNum, size_t OutputNum,
                                 int NThreads1)
    : GCNConvSequential(AdjMatrix, Output, Weight, InputNum, OutputNum),
      NThreads(NThreads1) {}

void GCNConvParallel::forward(double *Features, std::vector<int> Mask) {
  int *Ap = AdjMatrix->p;
  int *Ai = AdjMatrix->i;
  double *Ax = AdjMatrix->x;
  double degrees[AdjMatrix->m];
  for (int i = 0; i < AdjMatrix->m; i++) {
    degrees[i] = 0;
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      degrees[i] += 1;
    }
  }
#pragma omp parallel num_threads(this->NThreads)
  {
#pragma omp for
    for (auto ii = Mask.begin(); ii != Mask.end(); ii++) {
      double *neighborMessage = new double[OutputNum];
      auto i = *ii;
      double *messages = Output + OutputNum * i;
      for (int j = Ap[i]; j < Ap[i + 1]; j++) {
        int n = Ai[j];
        vecMatMul(this->InputNum, this->OutputNum,
                  Features + (n * this->InputNum), this->Weight,
                  neighborMessage);
        normalizeMessage(this->OutputNum, degrees[i], degrees[Ai[j]],
                         neighborMessage);
        aggregateMessage(this->OutputNum, messages, neighborMessage);
      }
      delete[] neighborMessage;
    }
  }
}

void GCNConvFused::forward(double *Features, std::vector<int> Mask) {
  GCNConvSequential::forward(Features, Mask);
}
GCNConvFused::GCNConvFused(CSR *AdjMatrix, double *Output, double *Weight,
                           size_t InputNum, size_t OutputNum, int NThreads)
    : GCNConvParallel(AdjMatrix, Output, Weight, InputNum, OutputNum,
                      NThreads) {}
} // namespace gnn
} // namespace sym_lib


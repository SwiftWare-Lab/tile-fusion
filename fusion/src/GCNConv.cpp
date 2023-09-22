//
// Created by mehdi on 6/27/23.
//

#include "sparse-fusion/GCNConv.h"
#include <math.h>
namespace sym_lib {
namespace gnn {
GCNConv::GCNConv(CSR *AdjMatrix, double *Output, double *Weight,
                 size_t InputNum, size_t OutputNum)
    : AdjMatrix(AdjMatrix), Output(Output), Weight(Weight), InputNum(InputNum),
      OutputNum(OutputNum) {}

void GCNConv::forward(double *Features) {
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
  for (int i = 0; i < AdjMatrix->m; i++) {
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

void GCNConv::vecMatMul(int M, int N, double *Vec, double *Mat,
                        double *result) {
  for (int j = 0; j < N; j++) {
    result[j] = 0;
    for (int i = 0; i < M; i++) {
      result[j] += Vec[i] * Mat[i * N + j];
    }
  }
}

void GCNConv::aggregateMessage(int Dim, double *Messages,
                               double *NeighborMessage) {
  for (int i = 0; i < Dim; i++) {
    Messages[i] += NeighborMessage[i];
  }
}

void GCNConv::normalizeMessage(int Dim, double DegI, double DegJ,
                               double *NeighborMessage) {
  for (int i = 0; i < Dim; i++) {
    NeighborMessage[i] = NeighborMessage[i] / sqrt(DegI * DegJ);
  }
}

} // namespace gnn
} // namespace sym_lib
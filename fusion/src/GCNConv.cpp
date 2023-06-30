//
// Created by mehdi on 6/27/23.
//

#include "sparse-fusion/GCNConv.h"
#include <math.h>
namespace sym_lib {
namespace gnn {
GCNConv::GCNConv(CSR* AdjMatrix, float* Output, float* Weight, size_t InputNum, size_t OutputNum)
    : AdjMatrix(AdjMatrix), Output(Output), Weight(Weight), InputNum(InputNum), OutputNum(OutputNum) {}

void GCNConv::forward(float *Features) {
  int *Ap = AdjMatrix->p;
  int *Ai = AdjMatrix->i;
  double *Ax = AdjMatrix->x;
  for (int i = 0; i < AdjMatrix->m; i++) {
    float *messages = Output + OutputNum*i;
    double degI = 0;
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      degI += Ax[j];
    }
    for (int j = Ap[i]; j < Ap[j + 1]; j++) {
      int n = Ai[j];
      double degJ = 0;
      for (int k = Ap[n]; k < Ap[n + 1]; k++) {
        degJ += Ax[k];
      }
      float* neighborMessage = vecMatMul(this->InputNum, this->OutputNum, Features + (n*this->InputNum), this->Weight);
      normalizeMessage(this->OutputNum, degI, degJ, neighborMessage);
      aggregateMessage(this->OutputNum, messages, neighborMessage);
      delete[] neighborMessage;
    }
  }
}

float *GCNConv::vecMatMul(int M, int N, float *Vec, float *Mat) {
  float* out = new float[N];
  for (int i = 0; i < M; i++) {
    out[i] = 0;
    for (int j = 0; j < N; j++) {
      out[i] += Vec[i] * Mat[i*M + j];
    }
  }
  return out;
}

void GCNConv::aggregateMessage(int Dim, float *Messages, float *NeighborMessage) {
  for (int i = 0; i < Dim; i++) {
    Messages[i] += NeighborMessage[i];
  }
}

void GCNConv::normalizeMessage(int Dim, float DegI, float DegJ,
                          float *NeighborMessage) {
  for (int i = 0; i < Dim; i++) {
    NeighborMessage[i] = NeighborMessage[i] / sqrt(DegI * DegJ);
  }
}

} // namespace gnn
} // namespace sym_lib
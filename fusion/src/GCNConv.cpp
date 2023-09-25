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

void GCNConvFused::forward(double *Features, int LevelNo, const int *LevelPtr,
                           const int *ParPtr, const int *Partition,
                           const int *ParType, std::vector<int> Mask) {
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
  for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(this->NThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
        double *neighborMessage1 = new double[OutputNum];
        double *neighborMessage2 = new double[HiddenDim];
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {

            double *messages = Output + HiddenDim * i;
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int n = Ai[j];
              vecMatMul(this->InputNum, this->HiddenDim,
                        Features + (n * this->InputNum), this->Layer1Weight,
                        neighborMessage2);
              normalizeMessage(this->HiddenDim, degrees[i], degrees[Ai[j]],
                               neighborMessage2);
              aggregateMessage(this->HiddenDim, messages, neighborMessage2);
            }
          } else {
            double *messages = Output + OutputNum * i;
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int n = Ai[j];
              vecMatMul(this->HiddenDim, this->OutputNum,
                        HiddenOutput + (n * this->HiddenDim),
                        this->Layer2Weight, neighborMessage1);
              normalizeMessage(this->OutputNum, degrees[i], degrees[Ai[j]],
                               neighborMessage1);
              aggregateMessage(this->OutputNum, messages, neighborMessage1);
            }
          }
        }
        delete[] neighborMessage1;
        delete[] neighborMessage2;
      }
    }
  }
}
GCNConvFused::GCNConvFused(CSR *AdjMatrix, double *Output, double *HiddenOutput,
                           double *Layer1Weight, double *Layer2Weight,
                           size_t InputNum, size_t OutputNum, size_t HiddenDim,
                           int NThreads)
    : AdjMatrix(AdjMatrix), Output(Output), HiddenOutput(HiddenOutput),
      Layer1Weight(Layer1Weight), Layer2Weight(Layer2Weight),
      InputNum(InputNum), OutputNum(OutputNum), HiddenDim(HiddenDim),
      NThreads(NThreads) {}
void GCNConvFused::aggregateMessage(int Dim, double *Messages,
                                    double *NeighborMessage) {
  for (int i = 0; i < Dim; i++) {
    Messages[i] += NeighborMessage[i];
  }
}
void GCNConvFused::vecMatMul(int M, int N, double *Vec, double *Mat,
                             double *Result) {
  for (int j = 0; j < N; j++) {
    Result[j] = 0;
    for (int i = 0; i < M; i++) {
      Result[j] += Vec[i] * Mat[i * N + j];
    }
  }
}
void GCNConvFused::normalizeMessage(int Dim, double DegI, double DegJ,
                                    double *NeighborMessage) {
  for (int i = 0; i < Dim; i++) {
    NeighborMessage[i] = NeighborMessage[i] / sqrt(DegI * DegJ);
  }
}
} // namespace gnn
} // namespace sym_lib

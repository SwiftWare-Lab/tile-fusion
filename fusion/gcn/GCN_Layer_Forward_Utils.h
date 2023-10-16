//
// Created by salehm32 on 10/10/23.
//

#include <cmath>
#include <cstring>
#include <iostream>
#ifndef SPARSE_FUSION_GCN_LAYER_FORWARD_UTILS_H
#define SPARSE_FUSION_GCN_LAYER_FORWARD_UTILS_H

void vecMatMul(int M, int N, double *Vec, double *Mat, double *Result, double Alfa) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      Result[j] = Result[j] * Alfa * Vec[i] * Mat[i * N + j];
    }
  }
}

void aggregateMessage(int Dim, double *Messages, double *NeighborMessage) {
  for (int i = 0; i < Dim; i++) {
    Messages[i] += NeighborMessage[i];
  }
}

void normalizeMessage(int Dim, double DegI, double DegJ,
                      double *NeighborMessage) {
  for (int i = 0; i < Dim; i++) {
    NeighborMessage[i] = NeighborMessage[i] / sqrt(DegI * DegJ);
  }
}

void forwardForOneLayer(int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
                        int OutputChannelDim, int *Degrees, double *Features,
                        double *Weight, double *Output) {
  for (int i = 0; i < M; i++) {
    double *messages = Output + OutputChannelDim * i;
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      int n = Ai[j];
      vecMatMul(InputChannelDim, OutputChannelDim,
                Features + (n * InputChannelDim), Weight, messages, Ax[j]);
    }
  }
}

void forwardForOneLayerParallel(int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
                                int OutputChannelDim, int *Degrees,
                                double *Features, double *Weight,
                                double *Output, int NumThreads) {
#pragma omp parallel num_threads(NumThreads)
  double *neighborMessage = new double[OutputChannelDim];
#pragma omp parallel for
  for (int i = 0; i < M; i++) {
    double *messages = Output + OutputChannelDim * i;
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      int n = Ai[j];
      vecMatMul(InputChannelDim, OutputChannelDim,
                Features + (n * InputChannelDim), Weight, neighborMessage, 1.);
      normalizeMessage(OutputChannelDim, Degrees[i], Degrees[Ai[j]],
                       neighborMessage);
      aggregateMessage(OutputChannelDim, messages, neighborMessage);
    }
  }
  delete[] neighborMessage;
}

void forwardForFusedLayersParallelWithBatching(
    int M, int *Ap, int *Ai, double *Ax, int *Bp, int *Bi, double *Bx, int InputChannelDim,
    int HiddenChannelDim, int OutputChannelDim, int *Degrees, double *Features,
    double *Layer1Weight, double *Layer2Weight, double *Output,
    double *HiddenOutput, int NumThreads, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType) {
  for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(this->NThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
        double *neighborMessage = new double[OutputChannelDim];
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            double *messages = HiddenOutput + HiddenChannelDim * i;
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int n = Ai[j];
              vecMatMul(InputChannelDim, HiddenChannelDim,
                        Features + (n * InputChannelDim), Layer1Weight,
                        neighborMessage, 1.);
              normalizeMessage(HiddenChannelDim, Degrees[i], Degrees[n],
                               neighborMessage);
              aggregateMessage(HiddenChannelDim, messages, neighborMessage);
            }
          } else {
            for (int j = Bp[i]; j < Bp[i + 1]; j++) {
              double *messages = Output + OutputChannelDim * i;
              int n = Bi[j];
              vecMatMul(HiddenChannelDim, OutputChannelDim,
                        HiddenOutput + (n * HiddenChannelDim), Layer2Weight,
                        neighborMessage, 1.);
              normalizeMessage(OutputChannelDim, Degrees[i], Degrees[n],
                               neighborMessage);
              aggregateMessage(OutputChannelDim, messages, neighborMessage);
            }
          }
        }
        delete []neighborMessage;
      }
    }
  }
}

void forwardForFusedLayersWithBatching(
    int M, int *Ap, int *Ai, double *Ax, int *Bp, int *Bi, double *Bx, int InputChannelDim,
    int HiddenChannelDim, int OutputChannelDim, int *Degrees, double *Features,
    double *Layer1Weight, double *Layer2Weight, double *Output,
    double *HiddenOutput, int NumThreads, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType) {
  for (int i1 = 0; i1 < LevelNo; i1++) {
    for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
      double *neighborMessage = new double[OutputChannelDim];
      for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
        int i = Partition[k1];
        int t = ParType[k1];
        if (t == 0) {
          double *messages = HiddenOutput + HiddenChannelDim * i;
          for (int j = Ap[i]; j < Ap[i + 1]; j++) {
            int n = Ai[j];
            vecMatMul(InputChannelDim, HiddenChannelDim,
                      Features + (n * InputChannelDim), Layer1Weight,
                      neighborMessage, 1.);
            normalizeMessage(HiddenChannelDim, Degrees[i], Degrees[n],
                             neighborMessage);
            aggregateMessage(HiddenChannelDim, messages, neighborMessage);
          }
        } else {
          for (int j = Bp[i]; j < Bp[i + 1]; j++) {
            double *messages = Output + OutputChannelDim * i;
            int n = Bp[j];
            vecMatMul(HiddenChannelDim, OutputChannelDim,
                      HiddenOutput + (n * HiddenChannelDim), Layer2Weight,
                      neighborMessage, 1.);
            normalizeMessage(OutputChannelDim, Degrees[i], Degrees[n],
                             neighborMessage);
            aggregateMessage(OutputChannelDim, messages, neighborMessage);
          }
        }
      }
      delete []neighborMessage;
    }
  }
}

void forwardForFusedLayersWithBatchingRegisterReuse(
    int M, int *Ap, int *Ai, double *Ax, int *Bp, int *Bi, double *Bx, int InputChannelDim,
    int HiddenChannelDim, int OutputChannelDim, int *Degrees, double *Features,
    double *Layer1Weight, double *Layer2Weight, double *Output,
    double *HiddenOutput, int TileSize) {
  double tempOut[TileSize * HiddenChannelDim];
  int flag = false;
  for (int i = 0; i < M; i += TileSize) {
    std::memset(tempOut, 0, sizeof(double) * HiddenChannelDim * TileSize);
    for (int ii = 0; ii < TileSize; ii++) {
      if (ii + i >= M) {
        flag = true;
        break;
      }
      for (int j = Ap[ii + i]; j < Ap[ii + i + 1]; j++) {
        int n = Ai[j];
        vecMatMul(InputChannelDim,
                    HiddenChannelDim, Features + (n * InputChannelDim),
                    Layer1Weight,
                    tempOut + ii * HiddenChannelDim, Ax[j]);
      }
    }
    for (int ii = 1; ii < TileSize - 1; ii++) {
      if (ii + i >= M)
        break;
      double *messages = Output + OutputChannelDim * (i + ii);
      for (int j = Bp[ii + i]; j < Bp[ii + i + 1]; j++) {
        int n = Bi[j];
        vecMatMul(HiddenChannelDim,
                    OutputChannelDim, tempOut + (n % TileSize) * HiddenChannelDim,
                    Layer2Weight, messages, Bx[j]);
      }
    }
    if (flag)
      std::copy(tempOut, tempOut + HiddenChannelDim * (M % TileSize),
                HiddenOutput + i * HiddenChannelDim);
    else
      std::copy(tempOut, tempOut + HiddenChannelDim * TileSize,
                HiddenOutput + i * HiddenChannelDim);
  }
  for (int i = 0; i < M; i += TileSize) {
    int ii = TileSize - 1;
    int ii1 = 0;
    if (ii1 + i >= M)
      break;
    double *messages = Output + OutputChannelDim * (i + ii1);
    for (int j = Bp[i + ii1]; j < Bp[i + ii1 + 1]; j++) {
      int n = Bi[j];
      vecMatMul(HiddenChannelDim, OutputChannelDim, HiddenOutput + n * HiddenChannelDim,
                  Layer2Weight, messages, Bx[j]);
    }
    if (ii + i >= M)
      break;
    messages = Output + OutputChannelDim * (i + ii);
    for (int j = Bp[i + ii]; j < Bp[i + ii + 1]; j++) {
      int n = Bi[j];
      vecMatMul(HiddenChannelDim, OutputChannelDim, HiddenOutput + n * HiddenChannelDim,
                  Layer2Weight, messages, Bx[j]);
    }
  }
}

#endif // SPARSE_FUSION_GCN_LAYER_FORWARD_UTILS_H

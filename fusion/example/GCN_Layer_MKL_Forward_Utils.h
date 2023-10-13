//
// Created by salehm32 on 29/09/23.
//

#include "SWTensorBench.h"
#include <cstring>
#include <math.h>

#ifdef MKL
#include <mkl.h>
#endif

#ifndef SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H
#define SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H
using namespace swiftware::benchmark;

void forwardForOneLayer(int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
                        int OutputChannelDim, int *Degrees, double *Features,
                        double *Weight, double *Output) {
  for (int i = 0; i < M; i++) {
    double *messages = Output + OutputChannelDim * i;
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      int n = Ai[j];
      cblas_dgemv(CblasRowMajor, CblasTrans, InputChannelDim, OutputChannelDim,
                  Ax[j], // alpha
                  Weight, OutputChannelDim, Features + (n * InputChannelDim), 1,
                  1., // beta
                  messages, 1);
      for (int k = 0; k < OutputChannelDim; k++){
        std::cout << messages[k] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

}

void forwardForOneLayerParallel(int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
                                int OutputChannelDim, int *Degrees,
                                double *Features, double *Weight,
                                double *Output, int NumThreads) {
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < M; i++) {
      double *messages = Output + OutputChannelDim * i;
      for (int j = Ap[i]; j < Ap[i + 1]; j++) {
        int n = Ai[j];
        cblas_dgemv(
            CblasRowMajor, CblasTrans, InputChannelDim, OutputChannelDim,
            Ax[j], // alpha
            Weight, OutputChannelDim, Features + (n * InputChannelDim), 1,
            1., // beta
            messages, 1);
      }
    }
  }
}

void forwardForFusedLayersParallel(int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
                                   int HiddenChannelDim, int OutputChannelDim,
                                   int *Degrees, double *Features,
                                   double *Layer1Weight, double *Layer2Weight,
                                   double *Output, double *HiddenOutput,
                                   int NumThreads, int LevelNo,
                                   const int *LevelPtr, const int *ParPtr,
                                   const int *Partition, const int *ParType) {
  for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(this->NThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            double *messages = HiddenOutput + HiddenChannelDim * i;
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int n = Ai[j];
              cblas_dgemv(CblasRowMajor, CblasTrans, InputChannelDim,
                          HiddenChannelDim,
                          Ax[j], // alpha
                          Layer1Weight, HiddenChannelDim,
                          Features + (n * InputChannelDim), 1, 1., // beta
                          messages, 1);
            }
          } else {
            double *messages = Output + OutputChannelDim * i;
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int n = Ai[j];
              cblas_dgemv(CblasRowMajor, CblasTrans, HiddenChannelDim,
                          OutputChannelDim,
                          Ax[j], // alpha
                          Layer2Weight, OutputChannelDim,
                          HiddenOutput + (n * HiddenChannelDim), 1, 1., // beta
                          messages, 1);
            }
          }
        }
      }
    }
  }
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
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            double *messages = HiddenOutput + HiddenChannelDim * i;
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int n = Ai[j];
              cblas_dgemv(CblasRowMajor, CblasTrans, InputChannelDim,
                          HiddenChannelDim,
                          Ax[j], // alpha
                          Layer1Weight, HiddenChannelDim,
                          Features + (n * InputChannelDim), 1, 1., // beta
                          messages, 1);
            }
          } else {
            double *messages = Output + OutputChannelDim * i;
            for (int j = Bp[i]; j < Bp[i + 1]; j++) {
              int n = Bi[j];
              cblas_dgemv(CblasRowMajor, CblasTrans, HiddenChannelDim,
                          OutputChannelDim,
                          Bx[j], // alpha
                          Layer2Weight, OutputChannelDim,
                          HiddenOutput + (n * HiddenChannelDim), 1, 1., // beta
                          messages, 1);
            }
          }
        }
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
      for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
        int i = Partition[k1];
        int t = ParType[k1];
        if (t == 0) {
          double *messages = HiddenOutput + HiddenChannelDim * i;
          for (int j = Ap[i]; j < Ap[i + 1]; j++) {
            int n = Ai[j];
            cblas_dgemv(CblasRowMajor, CblasTrans, InputChannelDim,
                        HiddenChannelDim,
                        Ax[j], // alpha
                        Layer1Weight, HiddenChannelDim,
                        Features + (n * InputChannelDim), 1, 1., // beta
                        messages, 1);
          }
        } else {
          double *messages = Output + OutputChannelDim * i;
          for (int j = Bp[i]; j < Bp[i + 1]; j++) {
            int n = Bi[j];
            cblas_dgemv(CblasRowMajor, CblasTrans, HiddenChannelDim,
                        OutputChannelDim,
                        Bx[j], // alpha
                        Layer2Weight, OutputChannelDim,
                        HiddenOutput + (n * HiddenChannelDim), 1, 1., // beta
                        messages, 1);
          }
        }
      }
    }
  }
}

// only works without batching for now
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
        cblas_dgemv(CblasRowMajor, CblasTrans, InputChannelDim,
                    HiddenChannelDim,
                    Ax[j], // alpha
                    Layer1Weight, HiddenChannelDim,
                    Features + (n * InputChannelDim), 1, 1., // beta
                    tempOut + ii * HiddenChannelDim, 1);
      }
    }
    for (int ii = 1; ii < TileSize - 1; ii++) {
      if (ii + i >= M)
        break;
      double *messages = Output + OutputChannelDim * (i + ii);
      for (int j = Bp[ii + i]; j < Bp[ii + i + 1]; j++) {
        int n = Bi[j];
        cblas_dgemv(CblasRowMajor, CblasTrans, HiddenChannelDim,
                    OutputChannelDim,
                    Bx[j], // alpha
                    Layer2Weight, OutputChannelDim,
                    tempOut + (n % TileSize) * HiddenChannelDim, 1, 1., // beta
                    messages, 1);
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
      cblas_dgemv(CblasRowMajor, CblasTrans, HiddenChannelDim, OutputChannelDim,
                  Bx[j], // alpha
                  Layer2Weight, OutputChannelDim,
                  HiddenOutput + n * HiddenChannelDim, 1, 1., // beta
                  messages, 1);
    }
    if (ii + i >= M)
      break;
    messages = Output + OutputChannelDim * (i + ii);
    for (int j = Bp[i + ii]; j < Bp[i + ii + 1]; j++) {
      int n = Bi[j];
      cblas_dgemv(CblasRowMajor, CblasTrans, HiddenChannelDim, OutputChannelDim,
                  Bx[j], // alpha
                  Layer2Weight, OutputChannelDim,
                  HiddenOutput + n * HiddenChannelDim, 1, 1., // beta
                  messages, 1);
    }
  }
}

void forwardForOneLayerWithGeMMAndSpMM(int NumOfNodes, sparse_matrix_t AdjMatrix,
                            double *Features, int FeatDim, double *Weight,
                            int OutDim, double *Output) {
  double *temp = new double[NumOfNodes * OutDim]{};
  matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NumOfNodes, OutDim,
              FeatDim, 1., Features, FeatDim, Weight, OutDim, 0., temp,
              OutDim);
  for (int i = 0; i < NumOfNodes; i++){
    for (int j = 0; j < OutDim; j++){
      std::cout <<
          temp[i*OutDim+j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, AdjMatrix, d, SPARSE_LAYOUT_ROW_MAJOR, temp,
                  OutDim, OutDim, 0, Output, OutDim);
  delete []temp;
}
#endif // SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H

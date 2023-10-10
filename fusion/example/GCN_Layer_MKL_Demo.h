//
// Created by salehm32 on 29/09/23.
//

#include "SWTensorBench.h"
#include <math.h>
#ifdef MKL
  #include <mkl.h>
#endif

#ifndef SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H
#define SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H
using namespace swiftware::benchmark;

void forwardForOneLayer(int M, int *Ap, int *Ai, int InputChannelDim,
                        int OutputChannelDim, int *Degrees, double *Features,
                        double *Weight, double *Output) {
  for (int i = 0; i < M; i++) {
    double *messages = Output + OutputChannelDim * i;
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      int n = Ai[j];
      cblas_dgemv(CblasRowMajor, CblasTrans, InputChannelDim, OutputChannelDim,
                  1. / sqrt(Degrees[i] * Degrees[n]), // alpha
                  Weight, OutputChannelDim, Features + (n * InputChannelDim), 1,
                  1., // beta
                  messages, 1);
    }
  }
}

void forwardForOneLayerParallel(int M, int *Ap, int *Ai, int InputChannelDim,
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
            1. / sqrt(Degrees[i] * Degrees[n]), // alpha
            Weight, OutputChannelDim, Features + (n * InputChannelDim), 1,
            1., // beta
            messages, 1);
      }
    }
  }
}

void forwardForFusedLayersParallel(int M, int *Ap, int *Ai, int InputChannelDim,
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
                          1. / sqrt(Degrees[i] * Degrees[n]), // alpha
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
                          1. / sqrt(Degrees[i] * Degrees[n]), // alpha
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
    int M, int *Ap, int *Ai, int *Bp, int *Bi, int InputChannelDim,
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
                          1. / sqrt(Degrees[i] * Degrees[n]), // alpha
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
                          1. / sqrt(Degrees[i] * Degrees[n]), // alpha
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
    int M, int *Ap, int *Ai, int *Bp, int *Bi, int InputChannelDim,
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
                        1. / sqrt(Degrees[i] * Degrees[n]), // alpha
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
                        1. / sqrt(Degrees[i] * Degrees[n]), // alpha
                        Layer2Weight, OutputChannelDim,
                        HiddenOutput + (n * HiddenChannelDim), 1, 1., // beta
                        messages, 1);
          }
        }
      }
    }
  }
}

#endif // SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H

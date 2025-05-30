//
// Created by salehm32 on 29/09/23.
//

#include "SWTensorBench.h"
#include <cstring>
#include <math.h>
#include <omp.h>

#ifdef MKL
#include <mkl.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifndef SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H
#define SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H
using namespace swiftware::benchmark;

void forwardForOneLayer(int M, int *Ap, int *Ai, double *Ax,
                        int InputChannelDim, int OutputChannelDim, int *Degrees,
                        double *Features, double *Weight, double *Output) {
  for (int i = 0; i < M; i++) {
    double *messages = Output + OutputChannelDim * i;
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      int n = Ai[j];
      cblas_dgemv(CblasRowMajor, CblasNoTrans, InputChannelDim,
                  OutputChannelDim,
                  Ax[j], // alpha
                  Weight, InputChannelDim, Features + (n * InputChannelDim), 1,
                  1., // beta
                  messages, 1);
    }
  }
}

void forwardForOneLayerSpMMGemVFusedSp(int M, int *Ap, int *Ai, float *Ax,
                        int InputChannelDim, int OutputChannelDim,
                        float *Features, float *Weight, float *Output, int NumThreads) {
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < M; i++) {
      float *messages = Output + OutputChannelDim * i;
      for (int j = Ap[i]; j < Ap[i + 1]; j++) {
        int n = Ai[j];
        cblas_sgemv(
            CblasRowMajor, CblasTrans, InputChannelDim, OutputChannelDim,
            Ax[j], // alpha
            Weight, OutputChannelDim, Features + (n * InputChannelDim), 1,
            1., // beta
            messages, 1);
      }
    }
  }
}

void forwardForOneLayerFusedParallel(int M, int *Ap, int *Ai, double *Ax,
                                     int InputChannelDim, int OutputChannelDim,
                                     int *Degrees, double *Features,
                                     double *Weight, double *Output,
                                     int NumThreads, int LevelNo,
                                     const int *LevelPtr, const int *ParPtr,
                                     const int *Partition, const int *ParType) {
  double *intermediateResult = new double[M*OutputChannelDim];
  for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0){
            cblas_dgemv(
                CblasRowMajor, CblasNoTrans, OutputChannelDim, InputChannelDim,
                1, // alpha
                Weight, InputChannelDim, Features + (i * InputChannelDim), 1,
                0, // beta
                intermediateResult + i*OutputChannelDim, 1);
          }
          else {
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int ip = OutputChannelDim * i;
              for (int k = 0; k < OutputChannelDim; k++) {
                Output[ip + k] += Ax[j] * intermediateResult[Ai[j] * OutputChannelDim + k];
              }
            }
          }
        }
      }
    }
  }
}

void forwardForOneLayerFusedParallelRedundantSp(
    int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
    int OutputChannelDim, float *Features, float *Weight,
    float *Output, float *Ws, int NumThreads,const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *MixPtr) {

  int numKer = 2;
  auto *cxBufAll = Ws; // new double[MTile * NTile * NThreads]();
  // First level benefits from Fusion
  int iBoundBeg = LevelPtr[0], iBoundEnd = LevelPtr[1];
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int j1 = iBoundBeg; j1 < iBoundEnd; ++j1) {
      auto *cxBuf = cxBufAll + omp_get_thread_num() * M * OutputChannelDim;
      int kBegin = ParPtr[j1], kEnd = MixPtr[j1 * numKer];
      int ii = Partition[kBegin]; // first iteration of tile
      int mTileLoc = kEnd - kBegin;
      // first loop, for every k-tile
      for (int k1 = kBegin; k1 < kEnd; k1++) { // i-loop
        int i = Partition[k1];
        // reset cxBuf, I used dot product to avoid the following
        // std::fill_n(cxBuf + i * NTile,  NTile, 0.0);
        cblas_sgemv(
            CblasRowMajor, CblasTrans, InputChannelDim, OutputChannelDim,
            1, // alpha
            Weight, OutputChannelDim, Features + (i * InputChannelDim), 1,
            0, // beta
            cxBuf + i*OutputChannelDim, 1);
      }
      int kEndL2 = MixPtr[j1 * numKer + 1];
      for (int k1 = kEnd; k1 < kEndL2; ++k1) {
        int i = Partition[k1];
        for (int kk = 0; kk < OutputChannelDim; kk += 32) {
          int ip = i * OutputChannelDim;
          auto dxV1 = _mm256_loadu_ps(Output + ip + kk);
          auto dxV2 = _mm256_loadu_ps(Output + ip + kk + 8);
          auto dxV3 = _mm256_loadu_ps(Output + ip + kk + 16);
          auto dxV4 = _mm256_loadu_ps(Output + ip + kk + 24);
          int k = Ap[i];
          for (; k < Ap[i + 1] - 1; k += 2) {
            int bij1 = Ai[k] * OutputChannelDim;
            int bij2 = Ai[k + 1] * OutputChannelDim;
            auto bxV1 = _mm256_set1_ps(Ax[k]);
            auto bxV2 = _mm256_set1_ps(Ax[k + 1]);
            auto acxV11 = _mm256_loadu_ps(cxBuf + bij1 + kk);
            auto acxV12 = _mm256_loadu_ps(cxBuf + bij1 + kk + 8);
            auto acxV13 = _mm256_loadu_ps(cxBuf + bij1 + kk + 16);
            auto acxV14 = _mm256_loadu_ps(cxBuf + bij1 + kk + 24);
            auto acxV21 = _mm256_loadu_ps(cxBuf + bij2 + kk);
            auto acxV22 = _mm256_loadu_ps(cxBuf + bij2 + kk + 8);
            auto acxV23 = _mm256_loadu_ps(cxBuf + bij2 + kk + 16);
            auto acxV24 = _mm256_loadu_ps(cxBuf + bij2 + kk + 24);
            dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
            dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
            dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
            dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
            dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
            dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
            dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
            dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
          }
          for (; k < Ap[i + 1]; ++k) {
            int bij = Ai[k] * OutputChannelDim;
            auto bxv0 = _mm256_set1_ps(Ax[k]);
            auto cxV11 = _mm256_loadu_ps(cxBuf + bij + kk);
            auto cxV12 = _mm256_loadu_ps(cxBuf + bij + kk + 8);
            auto cxV13 = _mm256_loadu_ps(cxBuf + bij + kk + 16);
            auto cxV14 = _mm256_loadu_ps(cxBuf + bij + kk + 24);
            dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
            dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
            dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
            dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
          }
          _mm256_storeu_ps(Output + ip + kk, dxV1);
          _mm256_storeu_ps(Output + ip + kk + 8, dxV2);
          _mm256_storeu_ps(Output + ip + kk + 16, dxV3);
          _mm256_storeu_ps(Output + ip + kk + 24, dxV4);
        }
      }
    }
  }
}

void forwardForOneLayerFusedParallelSeparatedSP(
    int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
    int OutputChannelDim, float *Features, float *Weight, float *Output,
    float *IntermediateResult, int NumThreads, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *MixPtr, const int *Partition) {
  int numKernels = 2;
  for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
        int kBeginL1 = ParPtr[j1];
        int kEndL1 = MixPtr[j1 * numKernels];
        int iL1 = Partition[kBeginL1];
        int tileSize = kEndL1 - kBeginL1;
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans, tileSize, OutputChannelDim,
            InputChannelDim, 1., Features + iL1 * InputChannelDim,
            InputChannelDim, Weight, InputChannelDim, 0.,
            IntermediateResult + iL1 * OutputChannelDim, OutputChannelDim);
        int kEndL2 = MixPtr[j1 * numKernels + 1];
        for (int k1 = kEndL1; k1 < kEndL2; ++k1) {
          int i = Partition[k1];
          for (int j = Ap[i]; j < Ap[i + 1]; j++) {
            int ip = OutputChannelDim * i;
            for (int k = 0; k < OutputChannelDim; k++) {
              Output[ip + k] +=
                  Ax[j] * IntermediateResult[Ai[j] * OutputChannelDim + k];
            }
          }
        }
      }
    }
  }
}

void forwardForOneLayerFusedParallelSeparatedVectorizedSP(
    const int M, const int *__restrict__ Ap, const int *__restrict__ Ai, const float *__restrict__ Ax, const int InputChannelDim,
    const int OutputChannelDim, const float *__restrict__ Features, const float *__restrict__ Weight, float *__restrict__ Output,
    float *__restrict__ IntermediateResult, const int NumThreads, const int LevelNo, const int *__restrict__ LevelPtr,
    const int *__restrict__ ParPtr, const int *__restrict__ MixPtr, const int *__restrict__ Partition) {
  int numKernels = 2;
  for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
        int kBeginL1 = ParPtr[j1];
        int kEndL1 = MixPtr[j1 * numKernels];
        int iL1 = Partition[kBeginL1];
        int tileSize = kEndL1 - kBeginL1;
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize, OutputChannelDim,
            InputChannelDim, 1., Features + iL1 * InputChannelDim,
            InputChannelDim, Weight, OutputChannelDim, 0.,
            IntermediateResult + iL1 * OutputChannelDim, OutputChannelDim);
        int kEndL2 = MixPtr[j1 * numKernels + 1];
        for (int k1 = kEndL1; k1 < kEndL2; ++k1) {
          int i = Partition[k1];
          for (int kk = 0; kk < OutputChannelDim; kk += 32) {
            int ip = i * OutputChannelDim;
            auto dxV1 = _mm256_loadu_ps(Output + ip + kk);
            auto dxV2 = _mm256_loadu_ps(Output + ip + kk + 8);
            auto dxV3 = _mm256_loadu_ps(Output + ip + kk + 16);
            auto dxV4 = _mm256_loadu_ps(Output + ip + kk + 24);
            int k = Ap[i];
            for (; k < Ap[i + 1]-1; k+=2) {
              int bij1 = Ai[k] * OutputChannelDim;
              int bij2 = Ai[k+1] * OutputChannelDim;
              auto bxV1 = _mm256_set1_ps(Ax[k]);
              auto bxV2 = _mm256_set1_ps(Ax[k+1]);
              auto acxV11 = _mm256_loadu_ps(IntermediateResult + bij1 + kk);
              auto acxV12 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 8);
              auto acxV13 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 16);
              auto acxV14 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 24);
              auto acxV21 = _mm256_loadu_ps(IntermediateResult + bij2 + kk);
              auto acxV22 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 8);
              auto acxV23 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 16);
              auto acxV24 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 24);
              dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
              dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
              dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
              dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
              dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
              dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
              dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
              dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
            }
            for (; k < Ap[i + 1]; ++k) {
              int bij = Ai[k] * OutputChannelDim;
              auto bxv0 = _mm256_set1_ps(Ax[k]);
              auto cxV11 = _mm256_loadu_ps(IntermediateResult + bij + kk);
              auto cxV12 = _mm256_loadu_ps(IntermediateResult + bij + kk + 8);
              auto cxV13 = _mm256_loadu_ps(IntermediateResult + bij + kk + 16);
              auto cxV14 = _mm256_loadu_ps(IntermediateResult + bij + kk + 24);
              dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
              dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
              dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
              dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
            }
            _mm256_storeu_ps(Output + ip + kk, dxV1);
            _mm256_storeu_ps(Output + ip + kk + 8, dxV2);
            _mm256_storeu_ps(Output + ip + kk + 16, dxV3);
            _mm256_storeu_ps(Output + ip + kk + 24, dxV4);
          }
        }
      }
    }
  }
}

//only for two kernels(levels) now.
void forwardForOneLayerFusedParallelSeparatedReorderedAdjVectorizedSP(
    const int M, const int *__restrict__ Ap, const int *__restrict__ Ai,
    const float *__restrict__ Ax, const int InputChannelDim,
    const int OutputChannelDim, const float *__restrict__ Features,
    const float *__restrict__ Weight, float *__restrict__ Output,
    float *__restrict__ IntermediateResult, const int NumThreads,
    const int LevelNo, const int *__restrict__ LevelPtr,
    const int *__restrict__ MixPtr, const int *__restrict__ L2Ptr) {
  int numKernels = 2;
  int k1Counter = 0;
  for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
        int kBeginL1 = MixPtr[j1 * numKernels];
        int kEndL1 = MixPtr[(j1+1) * numKernels];
        int tileSize = kEndL1 - kBeginL1;
        k1Counter += tileSize;
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize, OutputChannelDim,
            InputChannelDim, 1., Features + kBeginL1 * InputChannelDim,
            InputChannelDim, Weight, OutputChannelDim, 0.,
            IntermediateResult + kBeginL1 * OutputChannelDim, OutputChannelDim);
        int kBeginL2 = MixPtr[j1 * numKernels + 1];
        int kEndL2 = MixPtr[(j1+1) * numKernels + 1];
        for (int k1 = kBeginL2; k1 < kEndL2; ++k1) {
          int i = L2Ptr[k1];
          int row = k1;
          for (int kk = 0; kk < OutputChannelDim; kk += 32) {
            int ip = i * OutputChannelDim;
            auto dxV1 = _mm256_loadu_ps(Output + ip + kk);
            auto dxV2 = _mm256_loadu_ps(Output + ip + kk + 8);
            auto dxV3 = _mm256_loadu_ps(Output + ip + kk + 16);
            auto dxV4 = _mm256_loadu_ps(Output + ip + kk + 24);
            int k = Ap[row];
            for (; k < Ap[row + 1]-1; k+=2) {
              int bij1 = Ai[k] * OutputChannelDim;
              int bij2 = Ai[k+1] * OutputChannelDim;
              auto bxV1 = _mm256_set1_ps(Ax[k]);
              auto bxV2 = _mm256_set1_ps(Ax[k+1]);
              auto acxV11 = _mm256_loadu_ps(IntermediateResult + bij1 + kk);
              auto acxV12 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 8);
              auto acxV13 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 16);
              auto acxV14 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 24);
              auto acxV21 = _mm256_loadu_ps(IntermediateResult + bij2 + kk);
              auto acxV22 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 8);
              auto acxV23 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 16);
              auto acxV24 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 24);
              dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
              dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
              dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
              dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
              dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
              dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
              dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
              dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
            }
            for (; k < Ap[row + 1]; ++k) {
              int bij = Ai[k] * OutputChannelDim;
              auto bxv0 = _mm256_set1_ps(Ax[k]);
              auto cxV11 = _mm256_loadu_ps(IntermediateResult + bij + kk);
              auto cxV12 = _mm256_loadu_ps(IntermediateResult + bij + kk + 8);
              auto cxV13 = _mm256_loadu_ps(IntermediateResult + bij + kk + 16);
              auto cxV14 = _mm256_loadu_ps(IntermediateResult + bij + kk + 24);
              dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
              dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
              dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
              dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
            }
            _mm256_storeu_ps(Output + ip + kk, dxV1);
            _mm256_storeu_ps(Output + ip + kk + 8, dxV2);
            _mm256_storeu_ps(Output + ip + kk + 16, dxV3);
            _mm256_storeu_ps(Output + ip + kk + 24, dxV4);
          }
        }
      }
    }
  }
}

//only for two levels for now.
void forwardForOneLayerFusedParallelSeparatedReorderedUnfusedVectorizedSP(
    const int M, const int * Ap, const int * Ai, const float * Ax,
    const int * UFAp, const int * UFAi, const float *UFAx,  const int InputChannelDim,
    const int OutputChannelDim, const float *__restrict__ Features, const float *__restrict__ Weight, float *__restrict__ Output,
    float *__restrict__ IntermediateResult, const int NumThreads, const int LevelNo, const int *__restrict__ LevelPtr,
    const int *__restrict__ ParPtr, const int *__restrict__ MixPtr, const int *__restrict__ Partition) {
  int numKernels = 2;
  int unfusedStart = MixPtr[LevelPtr[1] * numKernels];
  const int* ap;
  const int* ai;
  const float* ax;
  for (int i1 = 0; i1 < LevelNo; i1++) {
    if(i1 == 0){
      ap = Ap;
      ai = Ai;
      ax = Ax;
    }
    else{
      ap = UFAp;
      ai = UFAi;
      ax = UFAx;
    }
#pragma omp parallel num_threads(NumThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
        int kBeginL1 = ParPtr[j1];
        int kEndL1 = MixPtr[j1 * numKernels];
        int iL1 = Partition[kBeginL1];
        int tileSize = kEndL1 - kBeginL1;
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize, OutputChannelDim,
            InputChannelDim, 1., Features + iL1 * InputChannelDim,
            InputChannelDim, Weight, OutputChannelDim, 0.,
            IntermediateResult + iL1 * OutputChannelDim, OutputChannelDim);
        int kEndL2 = MixPtr[j1 * numKernels + 1];
        for (int k1 = kEndL1; k1 < kEndL2; ++k1) {
          int i = Partition[k1];
          int sRow;
          if(i1 == 1){
            sRow = k1 - unfusedStart;
          }
          else{
            sRow = i;
          }
          for (int kk = 0; kk < OutputChannelDim; kk += 32) {
            int ip = i * OutputChannelDim;
            auto dxV1 = _mm256_loadu_ps(Output + ip + kk);
            auto dxV2 = _mm256_loadu_ps(Output + ip + kk + 8);
            auto dxV3 = _mm256_loadu_ps(Output + ip + kk + 16);
            auto dxV4 = _mm256_loadu_ps(Output + ip + kk + 24);
            int k = ap[sRow];
            for (; k < ap[sRow + 1]-1; k+=2) {
              int bij1 = ai[k] * OutputChannelDim;
              int bij2 = ai[k+1] * OutputChannelDim;
              auto bxV1 = _mm256_set1_ps(ax[k]);
              auto bxV2 = _mm256_set1_ps(ax[k+1]);
              auto acxV11 = _mm256_loadu_ps(IntermediateResult + bij1 + kk);
              auto acxV12 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 8);
              auto acxV13 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 16);
              auto acxV14 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 24);
              auto acxV21 = _mm256_loadu_ps(IntermediateResult + bij2 + kk);
              auto acxV22 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 8);
              auto acxV23 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 16);
              auto acxV24 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 24);
              dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
              dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
              dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
              dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
              dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
              dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
              dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
              dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
            }
            for (; k < ap[sRow + 1]; ++k) {
              int bij = ai[k] * OutputChannelDim;
              auto bxv0 = _mm256_set1_ps(ax[k]);
              auto cxV11 = _mm256_loadu_ps(IntermediateResult + bij + kk);
              auto cxV12 = _mm256_loadu_ps(IntermediateResult + bij + kk + 8);
              auto cxV13 = _mm256_loadu_ps(IntermediateResult + bij + kk + 16);
              auto cxV14 = _mm256_loadu_ps(IntermediateResult + bij + kk + 24);
              dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
              dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
              dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
              dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
            }
            _mm256_storeu_ps(Output + ip + kk, dxV1);
            _mm256_storeu_ps(Output + ip + kk + 8, dxV2);
            _mm256_storeu_ps(Output + ip + kk + 16, dxV3);
            _mm256_storeu_ps(Output + ip + kk + 24, dxV4);
          }
        }
      }
    }
  }
}


#ifdef __AVX2__
#define BUSY_WAIT_CONSTRUCT(task) {\
  int n = NParents[task]; \
  int *c = Parents[task]; \
  for (int pCntr = 0; pCntr < n; ++pCntr) \
    while (!TaskFinished[c[pCntr]]) _mm_pause(); \
}
#else
#define BUSY_WAIT_CONSTRUCT(task) {\
  int n = NParents[task]; \
  int *c = Parents[task]; \
  for (int i = 0; i < n; ++i) \
    while (!TaskFinished[c[i]]); \
}
#endif

void geMMSpMMFusedParallelSeparatedP2PThreadVectorizedSP(
    const int M, const int *__restrict__ Ap, const int *__restrict__ Ai, const float *__restrict__ Ax, const int InputChannelDim,
    const int OutputChannelDim, const float *__restrict__ Features, const float *__restrict__ Weight, float *__restrict__ Output,
    float *__restrict__ IntermediateResult, const int NumThreads, const int LevelNo, const int *__restrict__ LevelPtr,
    const int *__restrict__ ParPtr, const int *__restrict__ MixPtr, const int *__restrict__ Partition,
    int *__restrict__ NParents, int **__restrict__ Parents, bool *__restrict__ TaskFinished) {
  int numKernels = 2;
#pragma omp parallel num_threads(NumThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[0]; j1 < LevelPtr[2]; j1++) {
        BUSY_WAIT_CONSTRUCT(j1);
        int kBeginL1 = ParPtr[j1];
        int kEndL1 = MixPtr[j1 * numKernels];
        int iL1 = Partition[kBeginL1];
        int tileSize = kEndL1 - kBeginL1;
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize, OutputChannelDim,
            InputChannelDim, 1., Features + iL1 * InputChannelDim,
            InputChannelDim, Weight, OutputChannelDim, 0.,
            IntermediateResult + iL1 * OutputChannelDim, OutputChannelDim);
        int kEndL2 = MixPtr[j1 * numKernels + 1];
        for (int k1 = kEndL1; k1 < kEndL2; ++k1) {
          int i = Partition[k1];
          for (int kk = 0; kk < OutputChannelDim; kk += 32) {
            int ip = i * OutputChannelDim;
            auto dxV1 = _mm256_loadu_ps(Output + ip + kk);
            auto dxV2 = _mm256_loadu_ps(Output + ip + kk + 8);
            auto dxV3 = _mm256_loadu_ps(Output + ip + kk + 16);
            auto dxV4 = _mm256_loadu_ps(Output + ip + kk + 24);
            int k = Ap[i];
            for (; k < Ap[i + 1]-1; k+=2) {
              int bij1 = Ai[k] * OutputChannelDim;
              int bij2 = Ai[k+1] * OutputChannelDim;
              auto bxV1 = _mm256_set1_ps(Ax[k]);
              auto bxV2 = _mm256_set1_ps(Ax[k+1]);
              auto acxV11 = _mm256_loadu_ps(IntermediateResult + bij1 + kk);
              auto acxV12 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 8);
              auto acxV13 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 16);
              auto acxV14 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 24);
              auto acxV21 = _mm256_loadu_ps(IntermediateResult + bij2 + kk);
              auto acxV22 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 8);
              auto acxV23 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 16);
              auto acxV24 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 24);
              dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
              dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
              dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
              dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
              dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
              dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
              dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
              dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
            }
            for (; k < Ap[i + 1]; ++k) {
              int bij = Ai[k] * OutputChannelDim;
              auto bxv0 = _mm256_set1_ps(Ax[k]);
              auto cxV11 = _mm256_loadu_ps(IntermediateResult + bij + kk);
              auto cxV12 = _mm256_loadu_ps(IntermediateResult + bij + kk + 8);
              auto cxV13 = _mm256_loadu_ps(IntermediateResult + bij + kk + 16);
              auto cxV14 = _mm256_loadu_ps(IntermediateResult + bij + kk + 24);
              dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
              dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
              dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
              dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
            }
            _mm256_storeu_ps(Output + ip + kk, dxV1);
            _mm256_storeu_ps(Output + ip + kk + 8, dxV2);
            _mm256_storeu_ps(Output + ip + kk + 16, dxV3);
            _mm256_storeu_ps(Output + ip + kk + 24, dxV4);
          }
        }
        TaskFinished[j1] = true;
      }
    }
}


void forwardForOneLayerFusedParallelCSCAtomicSP(
    int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
    int OutputChannelDim, float *Features, float *Weight, float *Output,
    float *IntermediateResult, int NumThreads, int MTile) {
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < M; i += MTile) {
      int tileSize = std::min(MTile, M - i);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize,
                  OutputChannelDim, InputChannelDim, 1.,
                  Features + i * InputChannelDim, InputChannelDim, Weight,
                  OutputChannelDim, 0.,
                  IntermediateResult + i * OutputChannelDim, OutputChannelDim);
      for (int ii = i; ii < i + tileSize; ii++) {
        for (int k = Ap[ii]; k < Ap[ii + 1]; k++) { // for each column of B
          int bij = Ai[k] * OutputChannelDim;
          for (int kk = 0; kk < OutputChannelDim; ++kk) {
            auto tmp = Ax[k] * IntermediateResult[ii * OutputChannelDim + kk];
#pragma omp atomic
            Output[bij + kk] += tmp;
          }
        }
      }
    }
  }
}

void SpMMGeMMInterleavedFusedVectorizedSP(
    int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
    int OutputChannelDim, float *Features, float *Weight, float *Output,
    float *IntermediateResult, int NumThreads, int MTile) {
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < M; i += MTile) {
      int bound = std::min(i + MTile, M);
      for (int ii = i; ii < bound; ii++) {
        int ip = ii * InputChannelDim;
        for (int kk = 0; kk < InputChannelDim; kk += 32) {
          auto dxV1 = _mm256_loadu_ps(IntermediateResult + ip + kk);
          auto dxV2 = _mm256_loadu_ps(IntermediateResult + ip + kk + 8);
          auto dxV3 = _mm256_loadu_ps(IntermediateResult + ip + kk + 16);
          auto dxV4 = _mm256_loadu_ps(IntermediateResult + ip + kk + 24);
          int k = Ap[ii];
          for (; k < Ap[ii + 1] - 1; k += 2) {
            int bij1 = Ai[k] * InputChannelDim;
            int bij2 = Ai[k + 1] * InputChannelDim;
            auto bxV1 = _mm256_set1_ps(Ax[k]);
            auto bxV2 = _mm256_set1_ps(Ax[k + 1]);
            auto acxV11 = _mm256_loadu_ps(Features + bij1 + kk);
            auto acxV12 = _mm256_loadu_ps(Features + bij1 + kk + 8);
            auto acxV13 = _mm256_loadu_ps(Features + bij1 + kk + 16);
            auto acxV14 = _mm256_loadu_ps(Features + bij1 + kk + 24);
            auto acxV21 = _mm256_loadu_ps(Features + bij2 + kk);
            auto acxV22 = _mm256_loadu_ps(Features + bij2 + kk + 8);
            auto acxV23 = _mm256_loadu_ps(Features + bij2 + kk + 16);
            auto acxV24 = _mm256_loadu_ps(Features + bij2 + kk + 24);
            dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
            dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
            dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
            dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
            dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
            dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
            dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
            dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
          }
          for (; k < Ap[ii + 1]; ++k) {
            int bij = Ai[k] * InputChannelDim;
            auto bxv0 = _mm256_set1_ps(Ax[k]);
            auto cxV11 = _mm256_loadu_ps(Features + bij + kk);
            auto cxV12 = _mm256_loadu_ps(Features + bij + kk + 8);
            auto cxV13 = _mm256_loadu_ps(Features + bij + kk + 16);
            auto cxV14 = _mm256_loadu_ps(Features + bij + kk + 24);
            dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
            dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
            dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
            dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
          }
          _mm256_storeu_ps(IntermediateResult + ip + kk, dxV1);
          _mm256_storeu_ps(IntermediateResult + ip + kk + 8, dxV2);
          _mm256_storeu_ps(IntermediateResult + ip + kk + 16, dxV3);
          _mm256_storeu_ps(IntermediateResult + ip + kk + 24, dxV4);
        }
      }
      int tileSize = bound - i;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize,
                  OutputChannelDim, InputChannelDim, 1.,
                  IntermediateResult + i * InputChannelDim, InputChannelDim, Weight,
                  OutputChannelDim, 0.,
                  Output + i * OutputChannelDim, OutputChannelDim);
    }
  }
}

void forwardForOneLayerFusedParallelSeparated(int M, int *Ap, int *Ai, double *Ax,
                                     int InputChannelDim, int OutputChannelDim,
                                     int *Degrees, double *Features,
                                     double *Weight, double *Output, double *IntermediateResult,
                                     int NumThreads, int LevelNo,
                                     const int *LevelPtr, const int *ParPtr, const int* MixPtr,
                                     const int *Partition, const int *ParType) {

  int numKernels = 2;
  for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
        int kBeginL1 = ParPtr[j1];
        int kEndL1 = MixPtr[j1*numKernels];
        int iL1 = Partition[kBeginL1];
        int tileSize = kEndL1 - kBeginL1;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
                    OutputChannelDim, InputChannelDim, 1.,
                    Features + iL1 * InputChannelDim,
                    InputChannelDim, Weight, InputChannelDim, 0., IntermediateResult + iL1*OutputChannelDim,
                    OutputChannelDim);
        int kEndL2 = MixPtr[j1*numKernels + 1];
        for (int k1 = kEndL1; k1 < kEndL2; ++k1) {
          int i = Partition[k1];
          for (int j = Ap[i]; j < Ap[i + 1]; j++) {
            int ip = OutputChannelDim * i;
            for (int k = 0; k < OutputChannelDim; k++) {
              Output[ip + k] += Ax[j] * IntermediateResult[Ai[j] * OutputChannelDim + k];
            }
          }
        }
      }
    }
  }
}

void forwardForOneLayerFusedParallelSeparatedVectorized(int M, int *Ap, int *Ai, double *Ax,
                                              int InputChannelDim, int OutputChannelDim,
                                              int *Degrees, double *Features,
                                              double *Weight, double *Output, double *IntermediateResult,
                                              int NumThreads, int LevelNo,
                                              const int *LevelPtr, const int *ParPtr, const int* MixPtr,
                                              const int *Partition, const int *ParType) {

  int numKernels = 2;
  for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
        int kBeginL1 = ParPtr[j1];
        int kEndL1 = MixPtr[j1*numKernels];
        int iL1 = Partition[kBeginL1];
        int tileSize = kEndL1 - kBeginL1;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize,
                    OutputChannelDim, InputChannelDim, 1.,
                    Features + iL1 * InputChannelDim,
                    InputChannelDim, Weight, OutputChannelDim, 0., IntermediateResult + iL1*OutputChannelDim,
                    OutputChannelDim);
        int kEndL2 = MixPtr[j1*numKernels + 1];
        for (int k1 = kEndL1; k1 < kEndL2; ++k1) {
          int i = Partition[k1];
          int ip = i * OutputChannelDim;
          for (int kk = 0; kk < OutputChannelDim; kk += 16) {
            auto acxV1 = _mm256_loadu_pd(Output + ip + kk);
            auto acxV2 = _mm256_loadu_pd(Output + ip + kk + 4);
            auto acxV3 = _mm256_loadu_pd(Output + ip + kk + 8);
            auto acxV4 = _mm256_loadu_pd(Output + ip + kk + 12);
            int j = Ap[i];
            for (; j < Ap[i + 1]-1; j+=2) {
              int aij1 = Ai[j] * OutputChannelDim;
              int aij2 = Ai[j+1] * OutputChannelDim;
              auto axV1 = _mm256_set1_pd(Ax[j]);
              auto axV2 = _mm256_set1_pd(Ax[j+1]);
              auto cxV11 = _mm256_loadu_pd(IntermediateResult + aij1 + kk);
              auto cxV12 = _mm256_loadu_pd(IntermediateResult + aij1 + kk + 4);
              auto cxV13 = _mm256_loadu_pd(IntermediateResult + aij1 + kk + 8);
              auto cxV14 = _mm256_loadu_pd(IntermediateResult + aij1 + kk + 12);
              auto cxV21 = _mm256_loadu_pd(IntermediateResult + aij2 + kk);
              auto cxV22 = _mm256_loadu_pd(IntermediateResult + aij2 + kk + 4);
              auto cxV23 = _mm256_loadu_pd(IntermediateResult + aij2 + kk + 8);
              auto cxV24 = _mm256_loadu_pd(IntermediateResult + aij2 + kk + 12);
              acxV1 = _mm256_fmadd_pd(axV1, cxV11, acxV1);
              acxV1 = _mm256_fmadd_pd(axV2, cxV21, acxV1);
              acxV2 = _mm256_fmadd_pd(axV1, cxV12, acxV2);
              acxV2 = _mm256_fmadd_pd(axV2, cxV22, acxV2);
              acxV3 = _mm256_fmadd_pd(axV1, cxV13, acxV3);
              acxV3 = _mm256_fmadd_pd(axV2, cxV23, acxV3);
              acxV4 = _mm256_fmadd_pd(axV1, cxV14, acxV4);
              acxV4 = _mm256_fmadd_pd(axV2, cxV24, acxV4);
            }
            for (; j < Ap[i + 1]; ++j) {
              int aij = Ai[j] * OutputChannelDim;
              auto axv0 = _mm256_set1_pd(Ax[j]);
              auto cxV11 = _mm256_loadu_pd(IntermediateResult + aij + kk);
              auto cxV12 = _mm256_loadu_pd(IntermediateResult + aij + kk + 4);
              auto cxV13 = _mm256_loadu_pd(IntermediateResult + aij + kk + 8);
              auto cxV14 = _mm256_loadu_pd(IntermediateResult + aij + kk + 12);
              acxV1 = _mm256_fmadd_pd(axv0, cxV11, acxV1);
              acxV2 = _mm256_fmadd_pd(axv0, cxV12, acxV2);
              acxV3 = _mm256_fmadd_pd(axv0, cxV13, acxV3);
              acxV4 = _mm256_fmadd_pd(axv0, cxV14, acxV4);
            }
            _mm256_storeu_pd(Output + ip + kk, acxV1);
            _mm256_storeu_pd(Output + ip + kk + 4, acxV2);
            _mm256_storeu_pd(Output + ip + kk + 8, acxV3);
            _mm256_storeu_pd(Output + ip + kk + 12, acxV4);
          }
        }
      }
    }
  }
}

void forwardForOneLayerParallel(int M, int *Ap, int *Ai, double *Ax,
                                int InputChannelDim, int OutputChannelDim,
                                int *Degrees, double *Features, double *Weight,
                                double *Output, int NumThreads) {
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < M; i++) {
      double *messages = Output + OutputChannelDim * i;
      for (int j = Ap[i]; j < Ap[i + 1]; j++) {
        int n = Ai[j];
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans, OutputChannelDim, InputChannelDim,
            Ax[j], // alpha
            Weight, InputChannelDim, Features + (n * InputChannelDim), 1,
            1., // beta
            messages, 1);
      }
    }
  }
}

void forwardForFusedLayersParallel(
    int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
    int HiddenChannelDim, int OutputChannelDim, int *Degrees, double *Features,
    double *Layer1Weight, double *Layer2Weight, double *Output,
    double *HiddenOutput, int NumThreads, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType) {
  for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
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
              cblas_dgemv(CblasRowMajor, CblasNoTrans, HiddenChannelDim,
                          InputChannelDim,
                          Ax[j], // alpha
                          Layer1Weight, InputChannelDim,
                          Features + (n * InputChannelDim), 1, 1., // beta
                          messages, 1);
            }
          } else {
            double *messages = Output + OutputChannelDim * i;
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int n = Ai[j];
              cblas_dgemv(CblasRowMajor, CblasNoTrans, OutputChannelDim,
                          HiddenChannelDim,
                          Ax[j], // alpha
                          Layer2Weight, HiddenChannelDim,
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
    int M, int *Ap, int *Ai, double *Ax, int *Bp, int *Bi, double *Bx,
    int InputChannelDim, int HiddenChannelDim, int OutputChannelDim,
    int *Degrees, double *Features, double *Layer1Weight, double *Layer2Weight,
    double *Output, double *HiddenOutput, int NumThreads, int LevelNo,
    const int *LevelPtr, const int *ParPtr, const int *Partition,
    const int *ParType) {
  for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
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
    int M, int *Ap, int *Ai, double *Ax, int *Bp, int *Bi, double *Bx,
    int InputChannelDim, int HiddenChannelDim, int OutputChannelDim,
    int *Degrees, double *Features, double *Layer1Weight, double *Layer2Weight,
    double *Output, double *HiddenOutput, int NumThreads, int LevelNo,
    const int *LevelPtr, const int *ParPtr, const int *Partition,
    const int *ParType) {
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
    int M, int *Ap, int *Ai, double *Ax, int *Bp, int *Bi, double *Bx,
    int InputChannelDim, int HiddenChannelDim, int OutputChannelDim,
    int *Degrees, double *Features, double *Layer1Weight, double *Layer2Weight,
    double *Output, double *HiddenOutput, int TileSize) {
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

void forwardForOneLayerWithMKLGeMMAndMKLSpMM(int NumOfNodes,
                                       sparse_matrix_t AdjMatrix,
                                       double *Features, int FeatDim,
                                       double *Weight, int OutDim,
                                       double *Output, double* IntermediateResult) {
  matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NumOfNodes, OutDim,
              FeatDim, 1., Features, FeatDim, Weight, OutDim, 0.,
              IntermediateResult,
              OutDim);
  mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, AdjMatrix, d,
                  SPARSE_LAYOUT_ROW_MAJOR, IntermediateResult, OutDim, OutDim, 0, Output,
                  OutDim);
}


void forwardForOneLayerWithMKLGeMMAndMKLSpMMSP(int NumOfNodes,
                                             sparse_matrix_t AdjMatrix,
                                             float *Features, int FeatDim,
                                             float *Weight, int OutDim,
                                             float *Output, float *IntermediateResult) {
  matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NumOfNodes, OutDim,
              FeatDim, 1., Features, FeatDim, Weight, OutDim, 0.,
              IntermediateResult,
              OutDim);
  mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, AdjMatrix, d,
                  SPARSE_LAYOUT_ROW_MAJOR, IntermediateResult, OutDim, OutDim, 0, Output,
                  OutDim);
}

void forwardForOneLayerWithMKLSpMMAndMKLGeMMSP(int NumOfNodes,
                                               sparse_matrix_t AdjMatrix,
                                               float *Features, int FeatDim,
                                               float *Weight, int OutDim,
                                               float *Output, float *IntermediateResult) {
  matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, AdjMatrix, d,
                  SPARSE_LAYOUT_ROW_MAJOR, Features, FeatDim, FeatDim, 0, IntermediateResult,
                  FeatDim);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NumOfNodes, OutDim,
              FeatDim, 1., IntermediateResult, FeatDim, Weight, OutDim, 0.,
              Output,
              OutDim);
}

void forwardForOneLayerWithMKLGeMMAndSpMM(int NumOfNodes, int *Ap, int *Ai,
                                          double *Ax, double *Features,
                                          int FeatDim, double *Weight,
                                          int OutDim, double *Output, double *IntermediateResult,int NumThreads) {
  matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, NumOfNodes, OutDim,
              FeatDim, 1., Features, FeatDim, Weight, FeatDim, 0.,
              IntermediateResult,
              OutDim);
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < NumOfNodes; i++) {
      for (int j = Ap[i]; j < Ap[i + 1]; j++) {
        int ip = OutDim * i;
        for (int k = 0; k < OutDim; k++) {
          Output[ip + k] += Ax[j] * IntermediateResult[Ai[j] * OutDim + k];
        }
      }
    }
  }
}

void forwardForOneLayerWithMKLGeMMAndSpMMVectorized(int NumOfNodes, int *Ap, int *Ai,
                                          double *Ax, double *Features,
                                          int FeatDim, double *Weight,
                                          int OutDim, double *Output, double *IntermediateResult,int NumThreads) {
  matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NumOfNodes, OutDim,
              FeatDim, 1., Features, FeatDim, Weight, OutDim, 0.,
              IntermediateResult,
              OutDim);
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < NumOfNodes; i++) {
      int ip = OutDim * i;
      for (int kk = 0; kk < OutDim; kk += 16) {
        auto acxV1 = _mm256_loadu_pd(Output + ip + kk);
        auto acxV2 = _mm256_loadu_pd(Output + ip + kk + 4);
        auto acxV3 = _mm256_loadu_pd(Output + ip + kk + 8);
        auto acxV4 = _mm256_loadu_pd(Output + ip + kk + 12);
        int j = Ap[i];
        for (; j < Ap[i + 1] - 1; j += 2) {
          int aij1 = Ai[j] * OutDim;
          int aij2 = Ai[j + 1] * OutDim;
          auto axV1 = _mm256_set1_pd(Ax[j]);
          auto axV2 = _mm256_set1_pd(Ax[j + 1]);
          auto cxV11 = _mm256_loadu_pd(IntermediateResult + aij1 + kk);
          auto cxV12 = _mm256_loadu_pd(IntermediateResult + aij1 + kk + 4);
          auto cxV13 = _mm256_loadu_pd(IntermediateResult + aij1 + kk + 8);
          auto cxV14 = _mm256_loadu_pd(IntermediateResult + aij1 + kk + 12);
          auto cxV21 = _mm256_loadu_pd(IntermediateResult + aij2 + kk);
          auto cxV22 = _mm256_loadu_pd(IntermediateResult + aij2 + kk + 4);
          auto cxV23 = _mm256_loadu_pd(IntermediateResult + aij2 + kk + 8);
          auto cxV24 = _mm256_loadu_pd(IntermediateResult + aij2 + kk + 12);
          acxV1 = _mm256_fmadd_pd(axV1, cxV11, acxV1);
          acxV1 = _mm256_fmadd_pd(axV2, cxV21, acxV1);
          acxV2 = _mm256_fmadd_pd(axV1, cxV12, acxV2);
          acxV2 = _mm256_fmadd_pd(axV2, cxV22, acxV2);
          acxV3 = _mm256_fmadd_pd(axV1, cxV13, acxV3);
          acxV3 = _mm256_fmadd_pd(axV2, cxV23, acxV3);
          acxV4 = _mm256_fmadd_pd(axV1, cxV14, acxV4);
          acxV4 = _mm256_fmadd_pd(axV2, cxV24, acxV4);
        }
        for (; j < Ap[i + 1]; ++j) {
          int aij = Ai[j] * OutDim;
          auto axv0 = _mm256_set1_pd(Ax[j]);
          auto cxV11 = _mm256_loadu_pd(IntermediateResult + aij + kk);
          auto cxV12 = _mm256_loadu_pd(IntermediateResult + aij + kk + 4);
          auto cxV13 = _mm256_loadu_pd(IntermediateResult + aij + kk + 8);
          auto cxV14 = _mm256_loadu_pd(IntermediateResult + aij + kk + 12);
          acxV1 = _mm256_fmadd_pd(axv0, cxV11, acxV1);
          acxV2 = _mm256_fmadd_pd(axv0, cxV12, acxV2);
          acxV3 = _mm256_fmadd_pd(axv0, cxV13, acxV3);
          acxV4 = _mm256_fmadd_pd(axv0, cxV14, acxV4);
        }
        _mm256_storeu_pd(Output + ip + kk, acxV1);
        _mm256_storeu_pd(Output + ip + kk + 4, acxV2);
        _mm256_storeu_pd(Output + ip + kk + 8, acxV3);
        _mm256_storeu_pd(Output + ip + kk + 12, acxV4);
      }
    }
  }
}

void forwardForOneLayerWithMKLGeMMAndSpMMSPVectorized(int NumOfNodes, int *Ap, int *Ai,
                                            float *Ax, float *Features,
                                            int FeatDim, float *Weight,
                                            int OutDim, float *Output, float *IntermediateResult,int NumThreads) {
  matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NumOfNodes, OutDim,
              FeatDim, 1., Features, FeatDim, Weight, OutDim, 0.,
              IntermediateResult,
              OutDim);
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < NumOfNodes; i++) {
      for (int kk = 0; kk < OutDim; kk += 32) {
        int ip = i * OutDim;
        auto dxV1 = _mm256_loadu_ps(Output + ip + kk);
        auto dxV2 = _mm256_loadu_ps(Output + ip + kk + 8);
        auto dxV3 = _mm256_loadu_ps(Output + ip + kk + 16);
        auto dxV4 = _mm256_loadu_ps(Output + ip + kk + 24);
        int k = Ap[i];
        for (; k < Ap[i + 1] - 1; k += 2) {
          int bij1 = Ai[k] * OutDim;
          int bij2 = Ai[k + 1] * OutDim;
          auto bxV1 = _mm256_set1_ps(Ax[k]);
          auto bxV2 = _mm256_set1_ps(Ax[k + 1]);
          auto acxV11 = _mm256_loadu_ps(IntermediateResult + bij1 + kk);
          auto acxV12 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 8);
          auto acxV13 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 16);
          auto acxV14 = _mm256_loadu_ps(IntermediateResult + bij1 + kk + 24);
          auto acxV21 = _mm256_loadu_ps(IntermediateResult + bij2 + kk);
          auto acxV22 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 8);
          auto acxV23 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 16);
          auto acxV24 = _mm256_loadu_ps(IntermediateResult + bij2 + kk + 24);
          dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
          dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
          dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
          dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
          dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
          dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
          dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
          dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
        }
        for (; k < Ap[i + 1]; ++k) {
          int bij = Ai[k] * OutDim;
          auto bxv0 = _mm256_set1_ps(Ax[k]);
          auto cxV11 = _mm256_loadu_ps(IntermediateResult + bij + kk);
          auto cxV12 = _mm256_loadu_ps(IntermediateResult + bij + kk + 8);
          auto cxV13 = _mm256_loadu_ps(IntermediateResult + bij + kk + 16);
          auto cxV14 = _mm256_loadu_ps(IntermediateResult + bij + kk + 24);
          dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
          dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
          dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
          dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
        }
        _mm256_storeu_ps(Output + ip + kk, dxV1);
        _mm256_storeu_ps(Output + ip + kk + 8, dxV2);
        _mm256_storeu_ps(Output + ip + kk + 16, dxV3);
        _mm256_storeu_ps(Output + ip + kk + 24, dxV4);
      }
    }
  }
}

void forwardForOneLayerWithMKLGeMMAndSpMMSP(int NumOfNodes, int *Ap, int *Ai,
                                          float *Ax, float *Features,
                                          int FeatDim, float *Weight,
                                          int OutDim, float *Output, float *IntermediateResult,int NumThreads) {
  matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, NumOfNodes, OutDim,
              FeatDim, 1., Features, FeatDim, Weight, FeatDim, 0.,
              IntermediateResult,
              OutDim);
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < NumOfNodes; i++) {
      for (int j = Ap[i]; j < Ap[i + 1]; j++) {
        int ip = OutDim * i;
        for (int k = 0; k < OutDim; k++) {
          Output[ip + k] += Ax[j] * IntermediateResult[Ai[j] * OutDim + k];
        }
      }
    }
  }
}


void forwardForOneLayerUnfusedCSC(int NumOfNodes, int *Ap, int *Ai, double *Ax,
                                  double *Features, int FeatDim, double *Weight,
                                  int OutDim, double *Output) {
  double *temp = new double[NumOfNodes * OutDim]{};
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, NumOfNodes, OutDim,
              FeatDim, 1., Features, FeatDim, Weight, FeatDim, 0., temp,
              OutDim);
  for (int i = 0; i < NumOfNodes; i++) {
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      int n = Ai[j];
      for (int k = 0; k < OutDim; k++) {
        Output[n * OutDim + k] += Ax[j] * temp[i * OutDim + k];
      }
    }
  }
  delete[] temp;
}

void forwardForOneLayerFromCSC(int M, int *Ap, int *Ai, double *Ax,
                               int InputChannelDim, int OutputChannelDim,
                               int *Degrees, double *Features, double *Weight,
                               double *Output) {
  double cache[OutputChannelDim];
  for (int i = 0; i < M; i++) {
    std::memset(cache, 0, sizeof(double *) * OutputChannelDim);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, OutputChannelDim, InputChannelDim,
                1, // alpha
                Weight, InputChannelDim, Features + (i * InputChannelDim), 1,
                1., // beta
                cache, 1);
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      for (int k = 0; k < OutputChannelDim; k++) {
        Output[Ai[j] * OutputChannelDim + k] += Ax[j] * cache[k];
      }
    }
  }
}

void forwardForOneLayerFromCSC2(int M, int *Ap, int *Ai, double *Ax,
                                int InputChannelDim, int OutputChannelDim,
                                int *Degrees, double *Features, double *Weight,
                                double *Output) {
  double cache[OutputChannelDim];
  for (int i = 0; i < M; i++) {
    std::memset(cache, 0, sizeof(double *) * OutputChannelDim);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, OutputChannelDim, InputChannelDim,
                1, // alpha
                Weight, InputChannelDim, Features + (i * InputChannelDim), 1,
                1., // beta
                cache, 1);
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      for (int k = 0; k < OutputChannelDim; k++) {
        Output[Ai[j] * OutputChannelDim + k] += Ax[j] * cache[k];
      }
    }
  }
}

void forwardForOneLayerFromCSCParallel(int M, int *Ap, int *Ai, double *Ax,
                                       int InputChannelDim,
                                       int OutputChannelDim, int *Degrees,
                                       double *Features, double *Weight,
                                       double *Output, int NumThreads) {

#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < M; i++) {
      double cache[OutputChannelDim];
      std::memset(cache, 0, sizeof(double *) * OutputChannelDim);
      cblas_dgemv(CblasRowMajor, CblasNoTrans, OutputChannelDim,
                  InputChannelDim,
                  1, // alpha
                  Weight, InputChannelDim, Features + (i * InputChannelDim), 1,
                  1., // beta
                  cache, 1);
      for (int j = Ap[i]; j < Ap[i + 1]; j++) {
        for (int k = 0; k < OutputChannelDim; k++) {
          Output[Ai[j] * OutputChannelDim + k] += Ax[j] * cache[k];
        }
      }
    }
  }
}

// prediction is that it only perform good for reordered graphs
// for now only works on tri-banded
void forwardForOneLayerTiled(int M, int *Ap, int *Ai, double *Ax,
                             int InputChannelDim, int OutputChannelDim,
                             int *Degrees, double *Features, double *Weight,
                             double *Output, int TileSize, int *GeMMLowerBounds,
                             int *GeMMUpperBounds, int MaxGeMMTileSize) {
  double *temp = new double[MaxGeMMTileSize * OutputChannelDim];
  for (int i = 0; i < M; i += TileSize) {
    int geMMTileStartLoc = GeMMLowerBounds[i / TileSize];
    int geMMTileEndLoc = GeMMUpperBounds[i / TileSize];
    int geMMTileSize = geMMTileEndLoc - geMMTileStartLoc;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, geMMTileSize,
                OutputChannelDim, InputChannelDim, 1.,
                Features + geMMTileStartLoc * InputChannelDim, InputChannelDim,
                Weight, InputChannelDim, 0., temp, OutputChannelDim);
    for (int ii = 0; ii < TileSize; ii++) {
      if (i + ii >= M)
        break;
      for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
        int n = Ai[j];
        for (int k = 0; k < OutputChannelDim; k++) {
          Output[(i + ii) * OutputChannelDim + k] +=
              Ax[j] * temp[(n - geMMTileStartLoc) * OutputChannelDim + k];
        }
      }
    }
  }
  delete[] temp;
}

void forwardForOneLayerTiledParallel(int M, int *Ap, int *Ai, double *Ax,
                                     int InputChannelDim, int OutputChannelDim,
                                     int *Degrees, double *Features,
                                     double *Weight, double *Output,
                                     int TileSize, int NumThreads,
                                     int *GeMMLowerBounds, int *GeMMUpperBounds,
                                     int MaxGeMMTileSize) {
  double *temp = new double[NumThreads * MaxGeMMTileSize * OutputChannelDim];
#pragma omp parallel num_threads(NumThreads)
  {
    int threadId = omp_get_thread_num();
#pragma omp for
    for (int i = 0; i < M; i += TileSize) {
      double *ttemp = temp + threadId * MaxGeMMTileSize * OutputChannelDim;
      int geMMTileStartLoc = GeMMLowerBounds[i / TileSize];
      int geMMTileEndLoc = GeMMUpperBounds[i / TileSize];
      int geMMTileSize = geMMTileEndLoc - geMMTileStartLoc;
      //      Timer tgemm;
      //      tgemm.start();
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, geMMTileSize,
                  OutputChannelDim, InputChannelDim, 1.,
                  Features + geMMTileStartLoc * InputChannelDim,
                  InputChannelDim, Weight, OutputChannelDim, 0., ttemp,
                  OutputChannelDim);
      //      tgemm.stop();
      //      std::cout << "GeMM Time: " << tgemm.printTimeCsv(0) << std::endl;
      //      Timer tspmm;
      //      tspmm.start();
      for (int ii = 0; ii < TileSize; ii++) {
        if (i + ii >= M)
          break;
        for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
          int n = Ai[j];
          for (int k = 0; k < OutputChannelDim; k++) {
            Output[(i + ii) * OutputChannelDim + k] +=
                Ax[j] * ttemp[(n - geMMTileStartLoc) * OutputChannelDim + k];
          }
        }
      }
      //      tspmm.stop();
      //      std::cout << "SpMM Time: " << tspmm.printTimeCsv(0) << std::endl;
    }
  }
  delete[] temp;
}
// pick a column tile of adjacency matrix, perform gemm on the corresponding
// feature row tile and weight matrix, spmm on the corresponding adjacency
// matrix tile and the output of gemm
void forwardForOneLayerFromCSCTiled(int M, int *Ap, int *Ai, double *Ax,
                                    int InputChannelDim, int OutputChannelDim,
                                    int *Degrees, double *Features,
                                    double *Weight, double *Output,
                                    int TileSize) {
  double *cache = new double[TileSize * OutputChannelDim];
  int lastTileSize = M % TileSize;
  int lastCompleteTileEnd = M - lastTileSize;
  for (int i = 0; i < lastCompleteTileEnd; i += TileSize) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, TileSize,
                OutputChannelDim, InputChannelDim, 1.,
                Features + i * InputChannelDim, InputChannelDim, Weight,
                InputChannelDim, 0., cache, OutputChannelDim);
    for (int ii = 0; ii < TileSize; ii++) {
      for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
        for (int k = 0; k < OutputChannelDim; k++) {
          Output[Ai[j] * OutputChannelDim + k] +=
              Ax[j] * cache[ii * OutputChannelDim + k];
        }
      }
    }
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, lastTileSize,
              OutputChannelDim, InputChannelDim, 1.,
              Features + lastCompleteTileEnd * InputChannelDim, InputChannelDim,
              Weight, InputChannelDim, 0., cache, OutputChannelDim);
  for (int ii = 0; ii < lastTileSize; ii++) {
    for (int j = Ap[lastCompleteTileEnd + ii];
         j < Ap[lastCompleteTileEnd + ii + 1]; j++) {
      for (int k = 0; k < OutputChannelDim; k++) {
        Output[Ai[j] * OutputChannelDim + k] +=
            Ax[j] * cache[ii * OutputChannelDim + k];
      }
    }
  }
  delete[] cache;
}

void forwardForOneLayerFromCSCTiledParallel(int M, int *Ap, int *Ai, double *Ax,
                                            int InputChannelDim,
                                            int OutputChannelDim, int *Degrees,
                                            double *Features, double *Weight,
                                            double *Output, int TileSize,
                                            int NumThreads) {
  double lastTileSize = M % TileSize;
  int lastCompleteTileEnd = M - lastTileSize;
  double *cache = new double[TileSize * OutputChannelDim * NumThreads];
#pragma omp parallel num_threads(NumThreads)
  {
    int threadId = omp_get_thread_num();
#pragma omp for
    for (int i = 0; i < lastCompleteTileEnd; i += TileSize) {
      double *tcache = cache + threadId * TileSize * OutputChannelDim;
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, TileSize,
                  OutputChannelDim, InputChannelDim, 1.,
                  Features + i * InputChannelDim, InputChannelDim, Weight,
                  InputChannelDim, 0., tcache, OutputChannelDim);
      for (int ii = 0; ii < TileSize; ii++) {
        for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
          for (int k = 0; k < OutputChannelDim; k++) {
#pragma omp atomic
            Output[Ai[j] * OutputChannelDim + k] +=
                Ax[j] * tcache[ii * OutputChannelDim + k];
          }
        }
      }
    }
  }
  double *tcache = cache + 0 * TileSize * OutputChannelDim;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, lastTileSize,
              OutputChannelDim, InputChannelDim, 1.,
              Features + lastCompleteTileEnd * InputChannelDim, InputChannelDim,
              Weight, InputChannelDim, 0., tcache, OutputChannelDim);
  for (int ii = 0; ii < lastTileSize; ii++) {
    for (int j = Ap[lastCompleteTileEnd + ii];
         j < Ap[lastCompleteTileEnd + ii + 1]; j++) {
      for (int k = 0; k < OutputChannelDim; k++) {
        Output[Ai[j] * OutputChannelDim + k] +=
            Ax[j] * tcache[ii * OutputChannelDim + k];
      }
    }
  }
  delete[] cache;
}

void forwardForOneLayerFromCSCTiledParallelV2(
    int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
    int OutputChannelDim, int *Degrees, double *Features, double *Weight,
    double *Output, int MaxTileSize, int NumThreads, int Levels, int *LevelPtr,
    int *Id, int *TileSizes) {
  double *cache = new double[MaxTileSize * OutputChannelDim * NumThreads];
  for (int l = 0; l < Levels; l++) {
#pragma omp parallel num_threads(NumThreads)
    {
      int threadId = omp_get_thread_num();
#pragma omp for
      for (int t = LevelPtr[l]; t < LevelPtr[l + 1]; t++) {
        int id = Id[t];
        int tileSize = TileSizes[id];
        int i = id * MaxTileSize;
        double *tcache = cache + threadId * MaxTileSize * OutputChannelDim;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
                    OutputChannelDim, InputChannelDim, 1.,
                    Features + i * InputChannelDim, InputChannelDim, Weight,
                    InputChannelDim, 0., tcache, OutputChannelDim);
        for (int ii = 0; ii < tileSize; ii++) {
          for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
            for (int k = 0; k < OutputChannelDim; k++) {
              Output[Ai[j] * OutputChannelDim + k] +=
                  Ax[j] * tcache[ii * OutputChannelDim + k];
            }
          }
        }
      }
    }
  }
  delete[] cache;
}

void forwardForOneLayerFromCSCTiledParallelWithKTilingInWaveFronts(
    int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
    int OutputChannelDim, int *Degrees, double *Features, double *Weight,
    double *Output, int MaxTileSize, int NumThreads, int Levels, int *LevelPtr,
    int *Id, int *TileSizes, int *KId,
    int KTileSize) { // Assumption is that KTileSize is dividable by KTileSize
  double *cache = new double[MaxTileSize * KTileSize * NumThreads];

  for (int l = 0; l < Levels; l++) {
#pragma omp parallel num_threads(NumThreads)
    {
      int threadId = omp_get_thread_num();
#pragma omp for
      for (int t = LevelPtr[l]; t < LevelPtr[l + 1]; t++) {
        int id = Id[t];
        int tileSize = TileSizes[id];
        int i = id * MaxTileSize;
        int k = KId[t];
        double *tcache = cache + threadId * MaxTileSize * KTileSize;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
                    KTileSize, InputChannelDim, 1.,
                    Features + i * InputChannelDim, InputChannelDim,
                    Weight + k * InputChannelDim, InputChannelDim, 0., tcache,
                    KTileSize);
        for (int ii = 0; ii < tileSize; ii++) {
          for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
            for (int kk = 0; kk < KTileSize; kk++) {
              Output[Ai[j] * OutputChannelDim + k + kk] +=
                  Ax[j] * tcache[ii * KTileSize + kk];
            }
          }
        }
      }
    }
  }
  delete[] cache;
}

void forwardForOneLayerFromCSCTiledParallelWithKTiling(
    int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
    int OutputChannelDim, int *Degrees, double *Features, double *Weight,
    double *Output, int MaxTileSize, int NumThreads, int Levels, int *LevelPtr,
    int *Id, int *TileSizes,
    int KTileSize) { // Assumption is that KTileSize is dividable by KTileSize
  double *cache = new double[MaxTileSize * KTileSize * NumThreads];

  for (int l = 0; l < Levels; l++) {
#pragma omp parallel num_threads(NumThreads)
    {
      int threadId = omp_get_thread_num();
#pragma omp for
      for (int t = LevelPtr[l]; t < LevelPtr[l + 1]; t++) {
        int id = Id[t];
        int tileSize = TileSizes[id];
        int i = id * MaxTileSize;
        double *tcache = cache + threadId * MaxTileSize * KTileSize;
        for (int k = 0; k < OutputChannelDim; k += KTileSize) {
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
                      KTileSize, InputChannelDim, 1.,
                      Features + i * InputChannelDim, InputChannelDim,
                      Weight + k * InputChannelDim, InputChannelDim, 0., tcache,
                      KTileSize);
          for (int ii = 0; ii < tileSize; ii++) {
            for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
              for (int kk = 0; kk < KTileSize; kk++) {
                Output[Ai[j] * OutputChannelDim + k + kk] +=
                    Ax[j] * tcache[ii * KTileSize + kk];
              }
            }
          }
        }
      }
    }
  }
  delete[] cache;
}

void forwardForOneLayerFromCSCTiledParallelWithSchedulingForKTiling(
    int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
    int OutputChannelDim, int *Degrees, double *Features, double *Weight,
    double *Output, int MaxTileSize, int NumThreads, int Levels, int *LevelPtr,
    int *Id, int *TileSizes,
    int KTileSize) { // Assumption is that KTileSize is dividable by KTileSize
  double *cache = new double[MaxTileSize * KTileSize * NumThreads];
  int numOfKTiles = OutputChannelDim / KTileSize;
  for (int l = 0; l < Levels; l++) {
#pragma omp parallel num_threads(NumThreads)
    {
      int threadId = omp_get_thread_num();
#pragma omp for
      for (int t = LevelPtr[l]; t < LevelPtr[l + 1]; t++) {
        int tile = Id[t];
        int id = tile / numOfKTiles;
        int k = (tile % numOfKTiles) * KTileSize;
        int tileSize = TileSizes[id];
        int i = id * MaxTileSize;
        double *tcache = cache + threadId * MaxTileSize * KTileSize;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
                    KTileSize, InputChannelDim, 1.,
                    Features + i * InputChannelDim, InputChannelDim,
                    Weight + k * InputChannelDim, InputChannelDim, 0., tcache,
                    KTileSize);
        for (int ii = 0; ii < tileSize; ii++) {
          for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
            for (int kk = 0; kk < KTileSize; kk++) {
              Output[Ai[j] * OutputChannelDim + k + kk] +=
                  Ax[j] * tcache[ii * KTileSize + kk];
            }
          }
        }
      }
    }
  }
  delete[] cache;
}

void forwardForOneLayerFromCSCTiledParallelCombined(
    int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
    int OutputChannelDim, int *Degrees, double *Features, double *Weight,
    double *Output, int MinTileSize, int MaxTileSize, int NumThreads,
    int WorkloadsNum, int AggregatedTilesNum, int *WorkloadPtr, int *Id,
    int *TilePtr) {
  double *cache = new double[MinTileSize * OutputChannelDim * NumThreads];
  for (int l = 0; l < WorkloadsNum; l++) {
#pragma omp parallel num_threads(NumThreads)
    {
      int threadId = omp_get_thread_num();
#pragma omp for
      for (int t = WorkloadPtr[l]; t < WorkloadPtr[l + 1]; t++) {
        int id = Id[t];
        int tileSize = TilePtr[id + 1] - TilePtr[id];
        int i = TilePtr[id];
        double *tcache = cache + threadId * MinTileSize * OutputChannelDim;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
                    OutputChannelDim, InputChannelDim, 1.,
                    Features + i * InputChannelDim, InputChannelDim, Weight,
                    InputChannelDim, 0., tcache, OutputChannelDim);
        for (int ii = 0; ii < tileSize; ii++) {
          for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
            for (int k = 0; k < OutputChannelDim; k++) {
              Output[Ai[j] * OutputChannelDim + k] +=
                  Ax[j] * tcache[ii * OutputChannelDim + k];
            }
          }
        }
      }
    }
  }
  delete[] cache;
  mkl_set_num_threads(NumThreads);
  cache = new double[MaxTileSize * OutputChannelDim];
  for (int t = WorkloadPtr[WorkloadsNum];
       t < WorkloadPtr[WorkloadsNum] + AggregatedTilesNum; t++) {
    int id = Id[t];
    int tileSize = TilePtr[id + 1] - TilePtr[id];
    int i = TilePtr[id];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
                OutputChannelDim, InputChannelDim, 1.,
                Features + i * InputChannelDim, InputChannelDim, Weight,
                InputChannelDim, 0., cache, OutputChannelDim);
    for (int ii = 0; ii < tileSize; ii++) {
      for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
        for (int k = 0; k < OutputChannelDim; k++) {
          Output[Ai[j] * OutputChannelDim + k] +=
              Ax[j] * cache[ii * OutputChannelDim + k];
        }
      }
    }
  }
  delete[] cache;
}

void forwardForOneLayerFromCSCTiledParallelCombinedWithKTiling(
    int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
    int OutputChannelDim, int *Degrees, double *Features, double *Weight,
    double *Output, int MinTileSize, int MaxTileSize, int NumThreads,
    int WorkloadsNum, int AggregatedTilesNum, int *WorkloadPtr, int *Id,
    int *TilePtr, int KTileSize) {
  double *cache = new double[MinTileSize * OutputChannelDim * NumThreads];
  int numOfKTiles = OutputChannelDim / KTileSize;
  for (int l = 0; l < WorkloadsNum; l++) {
#pragma omp parallel num_threads(NumThreads)
    {
      int threadId = omp_get_thread_num();
#pragma omp for
      for (int t = WorkloadPtr[l]; t < WorkloadPtr[l + 1]; t++) {
        int id = Id[t];
        int tileSize = TilePtr[id + 1] - TilePtr[id];
        int i = TilePtr[id];
        double *tcache = cache + threadId * MinTileSize * KTileSize;
        int k = (t % numOfKTiles) * KTileSize;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
                    KTileSize, InputChannelDim, 1.,
                    Features + i * InputChannelDim, InputChannelDim,
                    Weight + k * InputChannelDim, InputChannelDim, 0., tcache,
                    KTileSize);
        for (int ii = 0; ii < tileSize; ii++) {
          for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
            for (int kk = 0; kk < KTileSize; kk++) {
              Output[Ai[j] * OutputChannelDim + k + kk] +=
                  Ax[j] * tcache[ii * KTileSize + kk];
            }
          }
        }
      }
    }
  }
  delete[] cache;
  mkl_set_num_threads(NumThreads);
  cache = new double[MaxTileSize * OutputChannelDim];
  for (int t = WorkloadPtr[WorkloadsNum];
       t < WorkloadPtr[WorkloadsNum] + AggregatedTilesNum; t++) {
    int id = Id[t];
    int tileSize = TilePtr[id + 1] - TilePtr[id];
    int i = TilePtr[id];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
                OutputChannelDim, InputChannelDim, 1.,
                Features + i * InputChannelDim, InputChannelDim, Weight,
                InputChannelDim, 0., cache, OutputChannelDim);
    for (int ii = 0; ii < tileSize; ii++) {
      for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
        for (int k = 0; k < OutputChannelDim; k++) {
          Output[Ai[j] * OutputChannelDim + k] +=
              Ax[j] * cache[ii * OutputChannelDim + k];
        }
      }
    }
  }
  delete[] cache;
}

// executer code for fused layers using csc format of the adjacency matrix

// has three types of layers: 0 for first layer, 1 for second layer, 2 for
// last tile of first layer, 3 for last tile of second layer(In case of
// different tile sizes)
void forwardForFusedLayersFromCSCTiled(
    int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
    int HiddenChannelDim, int OutputChannelDim, int *Degrees, double *Features,
    double *Weight1, double *Weight2, double *FirstLayerOutput, double *Output,
    int TileSize, int *ParPtr, int *Partition, int *Type) {
  double *cache1 = new double[TileSize * HiddenChannelDim];
  double *cache2 = new double[TileSize * OutputChannelDim];
  int lastTileSize = M % TileSize;
  int lastCompleteTileEnd = M - lastTileSize;
  for (int it = 0; it < ParPtr[1]; it++) {
    int i = Partition[it];
    int t = Type[it];
    if (t == 0) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, TileSize,
                  HiddenChannelDim, InputChannelDim, 1.,
                  Features + i * InputChannelDim, InputChannelDim, Weight1,
                  InputChannelDim, 0., cache1, HiddenChannelDim);
      for (int ii = 0; ii < TileSize; ii++) {
        for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
          for (int k = 0; k < HiddenChannelDim; k++) {
            FirstLayerOutput[Ai[j] * HiddenChannelDim + k] +=
                Ax[j] * cache1[ii * HiddenChannelDim + k];
          }
        }
      }
    } else if (t == 2) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, lastTileSize,
                  HiddenChannelDim, InputChannelDim, 1.,
                  Features + lastCompleteTileEnd * InputChannelDim,
                  InputChannelDim, Weight1, InputChannelDim, 0., cache1,
                  HiddenChannelDim);
      for (int ii = 0; ii < lastTileSize; ii++) {
        for (int j = Ap[lastCompleteTileEnd + ii];
             j < Ap[lastCompleteTileEnd + ii + 1]; j++) {
          for (int k = 0; k < HiddenChannelDim; k++) {
            FirstLayerOutput[Ai[j] * HiddenChannelDim + k] +=
                Ax[j] * cache1[ii * HiddenChannelDim + k];
          }
        }
      }
    } else if (t == 1) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, TileSize,
                  OutputChannelDim, HiddenChannelDim, 1.,
                  FirstLayerOutput + i * HiddenChannelDim, HiddenChannelDim,
                  Weight2, HiddenChannelDim, 0., cache2, OutputChannelDim);
      for (int ii = 0; ii < TileSize; ii++) {
        for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
          for (int k = 0; k < OutputChannelDim; k++) {
            Output[Ai[j] * OutputChannelDim + k] +=
                Ax[j] * cache2[ii * OutputChannelDim + k];
          }
        }
      }
    } else if (t == 3) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, lastTileSize,
                  OutputChannelDim, HiddenChannelDim, 1.,
                  FirstLayerOutput + lastCompleteTileEnd * HiddenChannelDim,
                  HiddenChannelDim, Weight2, HiddenChannelDim, 0., cache2,
                  OutputChannelDim);
      for (int ii = 0; ii < lastTileSize; ii++) {
        for (int j = Ap[lastCompleteTileEnd + ii];
             j < Ap[lastCompleteTileEnd + ii + 1]; j++) {
          for (int k = 0; k < OutputChannelDim; k++) {
            Output[Ai[j] * OutputChannelDim + k] +=
                Ax[j] * cache2[ii * OutputChannelDim + k];
          }
        }
      }
    }
  }
  delete[] cache1;
  delete[] cache2;
}

#ifdef __AVX2__
void forwardForOneLayerFromCSCVectorized(int M, int *Ap, int *Ai, double *Ax,
                                         int InputChannelDim,
                                         int OutputChannelDim, int *Degrees,
                                         double *Features, double *Weight,
                                         double *Output, int NumThreads) {
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < M; i++) {
      double cache[OutputChannelDim];
      std::memset(cache, 0, sizeof(double *) * OutputChannelDim);
      cblas_dgemv(CblasRowMajor, CblasNoTrans, OutputChannelDim,
                  InputChannelDim,
                  1, // alpha
                  Weight, InputChannelDim, Features + (i * InputChannelDim), 1,
                  1., // beta
                  cache, 1);
      for (int j = Ap[i]; j < Ap[i + 1]; j++) {
        __m256d Axj = _mm256_set1_pd(Ax[j]);
        for (int k = 0; k < OutputChannelDim; k += 4) {
          __m256d cachek = _mm256_loadu_pd(cache + k);
          __m256d outputk =
              _mm256_loadu_pd(Output + Ai[j] * OutputChannelDim + k);
          __m256d result = _mm256_fmadd_pd(Axj, cachek, outputk);
          _mm256_storeu_pd(Output + Ai[j] * OutputChannelDim + k, result);
        }
      }
    }
  }
}

void forwardForOneLayerFromCSCTiledVectorized(int M, int *Ap, int *Ai,
                                              double *Ax, int InputChannelDim,
                                              int OutputChannelDim,
                                              int *Degrees, double *Features,
                                              double *Weight, double *Output,
                                              int TileSize) {
  double cache[TileSize * OutputChannelDim];
  int lastTileSize = M % TileSize;
  int lastCompleteTileEnd = M - lastTileSize;
  for (int i = 0; i < lastCompleteTileEnd; i += TileSize) {
    std::memset(cache, 0, sizeof(double *) * TileSize * OutputChannelDim);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, TileSize,
                OutputChannelDim, InputChannelDim, 1.,
                Features + i * InputChannelDim, InputChannelDim, Weight,
                InputChannelDim, 0., cache, OutputChannelDim);
    for (int ii = 0; ii < TileSize; ii++) {
      int nnzNum = Ap[i + ii + 1] - Ap[i + ii];
      int unrollingEnd = Ap[i + ii + 1] - nnzNum % 3;
      for (int j = Ap[i + ii]; j < unrollingEnd; j += 3) {
        __m256d axj1 = _mm256_set1_pd(Ax[j]);
        __m256d axj2 = _mm256_set1_pd(Ax[j + 1]);
        __m256d axj3 = _mm256_set1_pd(Ax[j + 2]);
        for (int k = 0; k < OutputChannelDim; k += 4) {
          __m256d cachek = _mm256_loadu_pd(cache + ii * OutputChannelDim + k);
          __m256d output1k =
              _mm256_loadu_pd(Output + Ai[j] * OutputChannelDim + k);
          __m256d output2k =
              _mm256_loadu_pd(Output + Ai[j + 1] * OutputChannelDim + k);
          __m256d output3k =
              _mm256_loadu_pd(Output + Ai[j + 2] * OutputChannelDim + k);
          __m256d result1 = _mm256_fmadd_pd(axj1, cachek, output1k);
          __m256d result2 = _mm256_fmadd_pd(axj2, cachek, output2k);
          __m256d result3 = _mm256_fmadd_pd(axj3, cachek, output3k);
          _mm256_storeu_pd(Output + Ai[j] * OutputChannelDim + k, result1);
          _mm256_storeu_pd(Output + Ai[j + 1] * OutputChannelDim + k, result2);
          _mm256_storeu_pd(Output + Ai[j + 2] * OutputChannelDim + k, result3);
        }
      }
      for (int j = unrollingEnd; j < Ap[i + ii + 1]; j++) {
        __m256d axj = _mm256_set1_pd(Ax[j]);
        for (int k = 0; k < OutputChannelDim; k += 4) {
          __m256d cachek = _mm256_loadu_pd(cache + ii * OutputChannelDim + k);
          __m256d outputk =
              _mm256_loadu_pd(Output + Ai[j] * OutputChannelDim + k);
          __m256d result = _mm256_fmadd_pd(axj, cachek, outputk);
          _mm256_storeu_pd(Output + Ai[j] * OutputChannelDim + k, result);
        }
      }
    }
  }
  std::memset(cache, 0, sizeof(double *) * TileSize * OutputChannelDim);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, lastTileSize,
              OutputChannelDim, InputChannelDim, 1.,
              Features + lastCompleteTileEnd * InputChannelDim, InputChannelDim,
              Weight, InputChannelDim, 0., cache, OutputChannelDim);
  for (int ii = 0; ii < lastTileSize; ii++) {
    int nnzNum =
        Ap[lastCompleteTileEnd + ii + 1] - Ap[lastCompleteTileEnd + ii];
    int unrollingEnd = Ap[lastCompleteTileEnd + ii + 1] - nnzNum % 3;
    for (int j = Ap[lastCompleteTileEnd + ii]; j < unrollingEnd; j += 3) {
      __m256d axj1 = _mm256_set1_pd(Ax[j]);
      __m256d axj2 = _mm256_set1_pd(Ax[j + 1]);
      __m256d axj3 = _mm256_set1_pd(Ax[j + 2]);
      for (int k = 0; k < OutputChannelDim; k += 4) {
        __m256d cachek = _mm256_loadu_pd(cache + ii * OutputChannelDim + k);
        __m256d output1k =
            _mm256_loadu_pd(Output + Ai[j] * OutputChannelDim + k);
        __m256d output2k =
            _mm256_loadu_pd(Output + Ai[j + 1] * OutputChannelDim + k);
        __m256d output3k =
            _mm256_loadu_pd(Output + Ai[j + 2] * OutputChannelDim + k);
        __m256d result1 = _mm256_fmadd_pd(axj1, cachek, output1k);
        __m256d result2 = _mm256_fmadd_pd(axj2, cachek, output2k);
        __m256d result3 = _mm256_fmadd_pd(axj3, cachek, output3k);
        _mm256_storeu_pd(Output + Ai[j] * OutputChannelDim + k, result1);
        _mm256_storeu_pd(Output + Ai[j + 1] * OutputChannelDim + k, result2);
        _mm256_storeu_pd(Output + Ai[j + 2] * OutputChannelDim + k, result3);
      }
    }
    for (int j = unrollingEnd; j < Ap[lastCompleteTileEnd + ii + 1]; j++) {
      __m256d axj = _mm256_set1_pd(Ax[j]);
      for (int k = 0; k < OutputChannelDim; k += 4) {
        __m256d cachek = _mm256_loadu_pd(cache + ii * OutputChannelDim + k);
        __m256d outputk =
            _mm256_loadu_pd(Output + Ai[j] * OutputChannelDim + k);
        __m256d result = _mm256_fmadd_pd(axj, cachek, outputk);
        _mm256_storeu_pd(Output + Ai[j] * OutputChannelDim + k, result);
      }
    }
  }
}

void forwardForOneLayerFromCSCTiledParallelCombinedVectorized(
    int M, int *Ap, int *Ai, double *Ax, int InputChannelDim,
    int OutputChannelDim, int *Degrees, double *Features, double *Weight,
    double *Output, int MinTileSize, int MaxTileSize, int NumThreads,
    int WorkloadsNum, int AggregatedTilesNum, int *WorkloadPtr, int *Id,
    int *TilePtr) {
  double *cache = new double[MinTileSize * OutputChannelDim * NumThreads];
  for (int l = 0; l < WorkloadsNum; l++) {
#pragma omp parallel num_threads(NumThreads)
    {
      int threadId = omp_get_thread_num();
#pragma omp for
      for (int t = WorkloadPtr[l]; t < WorkloadPtr[l + 1]; t++) {
        int id = Id[t];
        int tileSize = TilePtr[id + 1] - TilePtr[id];
        int i = TilePtr[id];
        double *tcache = cache + threadId * MinTileSize * OutputChannelDim;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
                    OutputChannelDim, InputChannelDim, 1.,
                    Features + i * InputChannelDim, InputChannelDim, Weight,
                    InputChannelDim, 0., tcache, OutputChannelDim);
        for (int ii = 0; ii < tileSize; ii++) {
          int nnzNum = Ap[i + ii + 1] - Ap[i + ii];
          int unrollingEnd = Ap[i + ii + 1] - nnzNum % 3;
          for (int j = Ap[i + ii]; j < unrollingEnd; j += 3) {
            __m256d axV1 = _mm256_set1_pd(Ax[j]);
            __m256d axV2 = _mm256_set1_pd(Ax[j + 1]);
            __m256d axV3 = _mm256_set1_pd(Ax[j + 2]);
            for (int k = 0; k < OutputChannelDim; k += 4) {
              __m256d outV1 =
                  _mm256_loadu_pd(Output + (Ai[j] * OutputChannelDim + k));
              __m256d outV2 =
                  _mm256_loadu_pd(Output + (Ai[j + 1] * OutputChannelDim + k));
              __m256d outV3 =
                  _mm256_loadu_pd(Output + (Ai[j + 2] * OutputChannelDim + k));
              __m256d cacheV =
                  _mm256_loadu_pd(tcache + (ii * OutputChannelDim) + k);
              outV1 = _mm256_fmadd_pd(axV1, cacheV, outV1);
              outV2 = _mm256_fmadd_pd(axV2, cacheV, outV2);
              outV3 = _mm256_fmadd_pd(axV3, cacheV, outV3);
              _mm256_storeu_pd(Output + (Ai[j] * OutputChannelDim + k), outV1);
              _mm256_storeu_pd(Output + (Ai[j + 1] * OutputChannelDim + k),
                               outV2);
              _mm256_storeu_pd(Output + (Ai[j + 2] * OutputChannelDim + k),
                               outV3);
            }
          }
          for (int j = unrollingEnd; j < Ap[i + ii + 1]; j++) {
            __m256d axj = _mm256_set1_pd(Ax[j]);
            for (int k = 0; k < OutputChannelDim; k += 4) {
              __m256d cacheV =
                  _mm256_loadu_pd(tcache + ii * OutputChannelDim + k);
              __m256d outputV =
                  _mm256_loadu_pd(Output + Ai[j] * OutputChannelDim + k);
              outputV = _mm256_fmadd_pd(axj, cacheV, outputV);
              _mm256_storeu_pd(Output + Ai[j] * OutputChannelDim + k, outputV);
            }
          }
        }
      }
    }
  }
  delete[] cache;
  mkl_set_num_threads(NumThreads);
  cache = new double[MaxTileSize * OutputChannelDim];
  for (int t = WorkloadPtr[WorkloadsNum];
       t < WorkloadPtr[WorkloadsNum] + AggregatedTilesNum; t++) {
    int id = Id[t];
    int tileSize = TilePtr[id + 1] - TilePtr[id];
    int i = TilePtr[id];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
                OutputChannelDim, InputChannelDim, 1.,
                Features + i * InputChannelDim, InputChannelDim, Weight,
                InputChannelDim, 0., cache, OutputChannelDim);
    for (int ii = 0; ii < tileSize; ii++) {
      int nnzNum = Ap[i + ii + 1] - Ap[i + ii];
      int unrollingEnd = Ap[i + ii + 1] - nnzNum % 3;
      for (int j = Ap[i + ii]; j < unrollingEnd; j += 3) {
        __m256d axV1 = _mm256_set1_pd(Ax[j]);
        __m256d axV2 = _mm256_set1_pd(Ax[j + 1]);
        __m256d axV3 = _mm256_set1_pd(Ax[j + 2]);
        for (int k = 0; k < OutputChannelDim; k += 4) {
          __m256d outV1 =
              _mm256_loadu_pd(Output + (Ai[j] * OutputChannelDim + k));
          __m256d outV2 =
              _mm256_loadu_pd(Output + (Ai[j + 1] * OutputChannelDim + k));
          __m256d outV3 =
              _mm256_loadu_pd(Output + (Ai[j + 2] * OutputChannelDim + k));
          __m256d cacheV = _mm256_loadu_pd(cache + (ii * OutputChannelDim) + k);
          outV1 = _mm256_fmadd_pd(axV1, cacheV, outV1);
          outV2 = _mm256_fmadd_pd(axV2, cacheV, outV2);
          outV3 = _mm256_fmadd_pd(axV3, cacheV, outV3);
          _mm256_storeu_pd(Output + (Ai[j] * OutputChannelDim + k), outV1);
          _mm256_storeu_pd(Output + (Ai[j + 1] * OutputChannelDim + k), outV2);
          _mm256_storeu_pd(Output + (Ai[j + 2] * OutputChannelDim + k), outV3);
        }
      }
      for (int j = unrollingEnd; j < Ap[i + ii + 1]; j++) {
        __m256d axj = _mm256_set1_pd(Ax[j]);
        for (int k = 0; k < OutputChannelDim; k += 4) {
          __m256d cacheV = _mm256_loadu_pd(cache + ii * OutputChannelDim + k);
          __m256d outputV =
              _mm256_loadu_pd(Output + Ai[j] * OutputChannelDim + k);
          outputV = _mm256_fmadd_pd(axj, cacheV, outputV);
          _mm256_storeu_pd(Output + Ai[j] * OutputChannelDim + k, outputV);
        }
      }
    }
  }
  delete[] cache;
}

#endif

#endif // SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H

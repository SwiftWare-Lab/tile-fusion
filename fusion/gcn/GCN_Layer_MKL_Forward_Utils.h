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
      cblas_dgemv(CblasRowMajor, CblasTrans, InputChannelDim, OutputChannelDim,
                  Ax[j], // alpha
                  Weight, OutputChannelDim, Features + (n * InputChannelDim), 1,
                  1., // beta
                  messages, 1);
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
            CblasRowMajor, CblasTrans, InputChannelDim, OutputChannelDim,
            Ax[j], // alpha
            Weight, OutputChannelDim, Features + (n * InputChannelDim), 1,
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

void forwardForOneLayerWithGeMMAndSpMM(int NumOfNodes,
                                       sparse_matrix_t AdjMatrix,
                                       double *Features, int FeatDim,
                                       double *Weight, int OutDim,
                                       double *Output) {
  double *temp = new double[NumOfNodes * OutDim]{};
  matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NumOfNodes, OutDim,
              FeatDim, 1., Features, FeatDim, Weight, OutDim, 0., temp, OutDim);
  mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, AdjMatrix, d,
                  SPARSE_LAYOUT_ROW_MAJOR, temp, OutDim, OutDim, 0, Output,
                  OutDim);
  delete[] temp;
}

void forwardForOneLayerFromCSC(int M, int *Ap, int *Ai, double *Ax,
                               int InputChannelDim, int OutputChannelDim,
                               int *Degrees, double *Features, double *Weight,
                               double *Output) {
  double cache[OutputChannelDim];
  for (int i = 0; i < M; i++) {
    std::memset(cache, 0, sizeof(double *) * OutputChannelDim);
    cblas_dgemv(CblasRowMajor, CblasTrans, InputChannelDim, OutputChannelDim,
                1, // alpha
                Weight, OutputChannelDim, Features + (i * InputChannelDim), 1,
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
    cblas_dgemv(CblasRowMajor, CblasTrans, InputChannelDim, OutputChannelDim,
                1, // alpha
                Weight, OutputChannelDim, Features + (i * InputChannelDim), 1,
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
      cblas_dgemv(CblasRowMajor, CblasTrans, InputChannelDim, OutputChannelDim,
                  1, // alpha
                  Weight, OutputChannelDim, Features + (i * InputChannelDim), 1,
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
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, geMMTileSize,
                OutputChannelDim, InputChannelDim, 1.,
                Features + geMMTileStartLoc * InputChannelDim, InputChannelDim,
                Weight, OutputChannelDim, 0., temp, OutputChannelDim);
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
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, geMMTileSize,
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
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, TileSize,
                OutputChannelDim, InputChannelDim, 1.,
                Features + i * InputChannelDim, InputChannelDim, Weight,
                OutputChannelDim, 0., cache, OutputChannelDim);
    for (int ii = 0; ii < TileSize; ii++) {
      for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
        for (int k = 0; k < OutputChannelDim; k++) {
          Output[Ai[j] * OutputChannelDim + k] +=
              Ax[j] * cache[ii * OutputChannelDim + k];
        }
      }
    }
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lastTileSize,
              OutputChannelDim, InputChannelDim, 1.,
              Features + lastCompleteTileEnd * InputChannelDim, InputChannelDim,
              Weight, OutputChannelDim, 0., cache, OutputChannelDim);
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
    for (int i = 0; i < lastCompleteTileEnd; i += 2 * TileSize) {
      double *tcache = cache + threadId * TileSize * OutputChannelDim;
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, TileSize,
                  OutputChannelDim, InputChannelDim, 1.,
                  Features + i * InputChannelDim, InputChannelDim, Weight,
                  OutputChannelDim, 0., tcache, OutputChannelDim);
      for (int ii = 0; ii < TileSize; ii++) {
        for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
          for (int k = 0; k < OutputChannelDim; k++) {
            Output[Ai[j] * OutputChannelDim + k] +=
                Ax[j] * tcache[ii * OutputChannelDim + k];
          }
        }
      }
    }
  }
#pragma omp parallel num_threads(NumThreads)
  {
    int threadId = omp_get_thread_num();
#pragma omp for
    for (int i = TileSize; i < lastCompleteTileEnd; i += 2 * TileSize) {
      double *tcache = cache + threadId * TileSize * OutputChannelDim;
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, TileSize,
                  OutputChannelDim, InputChannelDim, 1.,
                  Features + i * InputChannelDim, InputChannelDim, Weight,
                  OutputChannelDim, 0., tcache, OutputChannelDim);
      for (int ii = 0; ii < TileSize; ii++) {
        for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
          for (int k = 0; k < OutputChannelDim; k++) {
            Output[Ai[j] * OutputChannelDim + k] +=
                Ax[j] * tcache[ii * OutputChannelDim + k];
          }
        }
      }
    }
  }
  double *tcache = cache + 0 * TileSize * OutputChannelDim;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lastTileSize,
              OutputChannelDim, InputChannelDim, 1.,
              Features + lastCompleteTileEnd * InputChannelDim, InputChannelDim,
              Weight, OutputChannelDim, 0., tcache, OutputChannelDim);
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
    double *Output, int MaxTileSize,int NumThreads, int Levels, int *LevelPtr,
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
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize,
                    OutputChannelDim, InputChannelDim, 1.,
                    Features + i * InputChannelDim, InputChannelDim, Weight,
                    OutputChannelDim, 0., tcache, OutputChannelDim);
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
// executer code for fused layers using csc format of the adjacency matrix

// has three types of layers: 0 for first layer, 1 for second layer, 2 for last
// tile of first layer, 3 for last tile of second layer(In case of different
// tile sizes)
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
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, TileSize,
                  HiddenChannelDim, InputChannelDim, 1.,
                  Features + i * InputChannelDim, InputChannelDim, Weight1,
                  HiddenChannelDim, 0., cache1, HiddenChannelDim);
      for (int ii = 0; ii < TileSize; ii++) {
        for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
          for (int k = 0; k < HiddenChannelDim; k++) {
            FirstLayerOutput[Ai[j] * HiddenChannelDim + k] +=
                Ax[j] * cache1[ii * HiddenChannelDim + k];
          }
        }
      }
    } else if (t == 2) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lastTileSize,
                  HiddenChannelDim, InputChannelDim, 1.,
                  Features + lastCompleteTileEnd * InputChannelDim,
                  InputChannelDim, Weight1, HiddenChannelDim, 0., cache1,
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
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, TileSize,
                  OutputChannelDim, HiddenChannelDim, 1.,
                  FirstLayerOutput + i * HiddenChannelDim, HiddenChannelDim,
                  Weight2, OutputChannelDim, 0., cache2, OutputChannelDim);
      for (int ii = 0; ii < TileSize; ii++) {
        for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; j++) {
          for (int k = 0; k < OutputChannelDim; k++) {
            Output[Ai[j] * OutputChannelDim + k] +=
                Ax[j] * cache2[ii * OutputChannelDim + k];
          }
        }
      }
    } else if (t == 3) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lastTileSize,
                  OutputChannelDim, HiddenChannelDim, 1.,
                  FirstLayerOutput + lastCompleteTileEnd * HiddenChannelDim,
                  HiddenChannelDim, Weight2, OutputChannelDim, 0., cache2,
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
      cblas_dgemv(CblasRowMajor, CblasTrans, InputChannelDim, OutputChannelDim,
                  1, // alpha
                  Weight, OutputChannelDim, Features + (i * InputChannelDim), 1,
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
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, TileSize,
                OutputChannelDim, InputChannelDim, 1.,
                Features + i * InputChannelDim, InputChannelDim, Weight,
                OutputChannelDim, 0., cache, OutputChannelDim);
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
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lastTileSize,
              OutputChannelDim, InputChannelDim, 1.,
              Features + lastCompleteTileEnd * InputChannelDim, InputChannelDim,
              Weight, OutputChannelDim, 0., cache, OutputChannelDim);
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
#endif

#endif // SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H

//
// Created by salehm32 on 29/11/23.
//

#include <torch/torch.h>
#include "Backend_Utils.h"
#include <mkl.h>
#include <omp.h>

#ifndef SPARSE_FUSION_TORCH_GCN_LAYER_UTILS_H
#define SPARSE_FUSION_TORCH_GCN_LAYER_UTILS_H

void forwardForOneLayerFromCSCTiledParallelCombined(
    int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
    int OutputChannelDim, float *Features, float *Weight, float *Output,
    int MinTileSize, int MaxTileSize, int NumThreads, int WorkloadsNum,
    int AggregatedTilesNum, int *WorkloadPtr, int *Id, int *TilePtr) {
  float *cache = new float[MinTileSize * OutputChannelDim * NumThreads];
  for (int l = 0; l < WorkloadsNum; l++) {
#pragma omp parallel num_threads(NumThreads)
    {
      int threadId = omp_get_thread_num();
#pragma omp for
      for (int t = WorkloadPtr[l]; t < WorkloadPtr[l + 1]; t++) {
        int id = Id[t];
        int tileSize = TilePtr[id + 1] - TilePtr[id];
        int i = TilePtr[id];
        float *tcache = cache + threadId * MinTileSize * OutputChannelDim;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
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
  cache = new float[MaxTileSize * OutputChannelDim];
  for (int t = WorkloadPtr[WorkloadsNum];
       t < WorkloadPtr[WorkloadsNum] + AggregatedTilesNum; t++) {
    int id = Id[t];
    int tileSize = TilePtr[id + 1] - TilePtr[id];
    int i = TilePtr[id];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tileSize,
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


void forwardForTiledFusedCSCGCN(torch::Tensor X, torch::Tensor Adj,
                                torch::Tensor Weight, torch::Tensor WorkloadPtr,
                                torch::Tensor Ids, torch::Tensor TilePtr,
                                int MaxTileSize, int MinTileSize,
                                int NumThreads, int NumWorkloads,
                                int NumAggregatedTiles) {
  float *out = new float[X.size(0) * Weight.size(0)];
  forwardForOneLayerFromCSCTiledParallelCombined(
      X.size(0), Adj.ccol_indices().data_ptr<int>(), Adj.row_indices().data_ptr<int>(), Adj.values().data_ptr<float>(), X.size(1),
      Weight.size(0), X.data_ptr<float>(), Weight.data_ptr<float>(), out, MinTileSize, MaxTileSize,
      NumThreads, NumWorkloads, NumAggregatedTiles, WorkloadPtr.data_ptr<int>(), Ids.data_ptr<int>(), TilePtr.data_ptr<int>());
}

void createScheduleFor(torch::Tensor Adj){

}


#endif // SPARSE_FUSION_TORCH_GCN_LAYER_UTILS_H

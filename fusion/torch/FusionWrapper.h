//
// Created by salehm32 on 05/01/24.
//

// #include "../gcn/Inspection/Fusion_Inspector.h"
#include "aggregation/sparse_utilities.h"
#include "inspection/GraphColoring.h"
#include "inspection/Inspection.h"
#include "sparse-fusion/SparseFusion.h"
// #include "sparse-fusion/MultiDimensionalSet.h"
// #include "Backend_Utils.h"
#ifdef MKL
#include <mkl.h>
#endif
#include <omp.h>
#include <torch/extension.h>
#ifndef FUSED_GCN_FUSIONWRAPPER_H
#define FUSED_GCN_FUSIONWRAPPER_H

void forwardForOneLayerFromCSCTiledParallelCombined(
    int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
    int OutputChannelDim, float *Features, float *Weight, float *Output,
    int MinTileSize, int MaxTileSize, int NumThreads, int WorkloadsNum,
    int AggregatedTilesNum, int *WorkloadPtr, int *Id, int *TilePtr);

std::vector<torch::Tensor>
createScheduleForCSC(torch::Tensor &Adj, int TileSize, int MinWorkloadSize);

void forwardForOneLayerFusedParallelSeparated(
    int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
    int OutputChannelDim, float *Features, float *Weight, float *Output,
    int NumThreads, int LevelNo, const int *LevelPtr, const int *ParPtr,
    const int *MixPtr, const int *Partition);

std::vector<torch::Tensor>
createScheduleForCSR(sym_lib::MultiDimensionalSet *FusedCompSet);

void forwardForOneLayerFromCSCTiledParallelCombined(
    int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
    int OutputChannelDim, float *Features, float *Weight, float *Output,
    int MinTileSize, int MaxTileSize, int NumThreads, int WorkloadsNum,
    int AggregatedTilesNum, int *WorkloadPtr, int *Id, int *TilePtr) {
  float *cache = new float[MinTileSize * OutputChannelDim * NumThreads];
  mkl_set_num_threads(1);
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

std::vector<torch::Tensor>
createScheduleForCSC(torch::Tensor &Adj, int TileSize, int MinWorkloadSize) {
  DsaturColoringForConflictGraph dsaturColoring;
  MultiDimensionalSet *fusedCompSet;
  int *colPtr = Adj.ccol_indices().data_ptr<int>();
  std::cout << colPtr << std::endl;
  std::map<int, std::vector<int>> colorToTiles =
      dsaturColoring.generateGraphColoringForConflictGraphOf(
          Adj.size(0), Adj.ccol_indices().data_ptr<int>(),
          Adj.row_indices().data_ptr<int>(), TileSize);
  InspectorForSingleLayerTiledFusedCSCCombined inspector;
  fusedCompSet = inspector.generateScheduleBasedOnConflictGraphColoring(
      colorToTiles, Adj.size(0), TileSize, MinWorkloadSize);
  int numOfTiles = fusedCompSet->ptr1_[fusedCompSet->n1_] + fusedCompSet->n2_;
  torch::Tensor workloadPtr = torch::from_blob(
      fusedCompSet->ptr1_, {fusedCompSet->n1_ + 1}, torch::kInt32);
  torch::Tensor ids =
      torch::from_blob(fusedCompSet->id_, {numOfTiles}, torch::kInt32);
  torch::Tensor tilePtr =
      torch::from_blob(fusedCompSet->type_, {numOfTiles}, torch::kInt32);
  torch::Tensor numericalParameters = torch::empty({3}, torch::kInt32);
  numericalParameters[0] = fusedCompSet->n1_;
  numericalParameters[1] = fusedCompSet->n2_;
  numericalParameters[2] = fusedCompSet->n3_;
  return {workloadPtr, ids, tilePtr, numericalParameters};
}

void forwardForOneLayerFusedParallelSeparated(
    int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
    int OutputChannelDim, float *Features, float *Weight, float *Output,
    int NumThreads, int LevelNo, const int *LevelPtr, const int *ParPtr,
    const int *MixPtr, const int *Partition) {
  float *intermediateResult = new float[M * OutputChannelDim];
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
            intermediateResult + iL1 * OutputChannelDim, OutputChannelDim);
        int kEndL2 = MixPtr[j1 * numKernels + 1];
        for (int k1 = kEndL1; k1 < kEndL2; ++k1) {
          int i = Partition[k1];
          for (int j = Ap[i]; j < Ap[i + 1]; j++) {
            int ip = OutputChannelDim * i;
            for (int k = 0; k < OutputChannelDim; k++) {
              Output[ip + k] +=
                  Ax[j] * intermediateResult[Ai[j] * OutputChannelDim + k];
            }
          }
        }
      }
    }
  }
}

sym_lib::MultiDimensionalSet *
generateFusedScheduleForCSRFused(sym_lib::CSR *adj,
                                 sym_lib::ScheduleParameters Sp) {
  Sp._num_w_partition =
      std::max<int>(adj->m / Sp.IterPerPartition, 2 * Sp._num_threads);
  auto *sf01 = new sym_lib::SparseFusion(&Sp, 2);
  auto *mvDAG = sym_lib::diagonal(adj->m, 1.0);
  auto *tmpCSCCSR =
      new sym_lib::CSC(adj->m, adj->n, adj->nnz, adj->p, adj->i, adj->x);
  // sym_lib::print_csc(1, "Di", 6, Di->p, Di->i, Di->x);
  sf01->fuse(0, mvDAG, tmpCSCCSR);

  // sf01->print_final_list();
  sf01->fuse(1, mvDAG, tmpCSCCSR);
  // sf01->print_final_list();
  sym_lib::MultiDimensionalSet *fusedCompSet =
      sf01->getFusedCompressed(sym_lib::Separated);
  //    FusedCompSet->print_3d();
  delete sf01;
  delete mvDAG;
  delete tmpCSCCSR;
  return fusedCompSet;
}

std::vector<torch::Tensor>
createScheduleForCSR(sym_lib::MultiDimensionalSet *FusedCompSet) {
  int levelNum = FusedCompSet->n1_;
  int *levelPtr = FusedCompSet->ptr1_;
  int *parPtr = FusedCompSet->ptr2_;
  int *partition = FusedCompSet->id_;
  int *mixPtr = FusedCompSet->ker_begin_;
  torch::Tensor levelPtrTensor =
      torch::from_blob(levelPtr, {levelNum + 1}, torch::kInt32);
  torch::Tensor parPtrTensor =
      torch::from_blob(parPtr, {levelPtr[levelNum]}, torch::kInt32);
  torch::Tensor mixPtrTensor =
      torch::from_blob(mixPtr, {levelPtr[levelNum] * 2}, torch::kInt32);
  torch::Tensor partitionTensor =
      torch::from_blob(partition, {parPtr[levelPtr[levelNum]]}, torch::kInt32);
  torch::Tensor numericalParameters = torch::empty({1}, torch::kInt32);
  numericalParameters[0] = levelNum;
  return {levelPtrTensor, parPtrTensor, partitionTensor, mixPtrTensor,
          numericalParameters};
}
#endif // FUSED_GCN_FUSIONWRAPPER_H
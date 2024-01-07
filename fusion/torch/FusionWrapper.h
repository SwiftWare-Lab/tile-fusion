//
// Created by salehm32 on 05/01/24.
//

//#include "../gcn/Inspection/Fusion_Inspector.h"
#include "inspection/GraphColoring.h"
#include "inspection/Inspection.h"
//#include "sparse-fusion/MultiDimensionalSet.h"
//#include "Backend_Utils.h"
#include <mkl.h>
#include <omp.h>
#include <torch/extension.h>
#ifndef FUSED_GCN_FUSIONWRAPPER_H
#define FUSED_GCN_FUSIONWRAPPER_H

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

torch::Tensor forwardForTiledFusedCSCGCN(torch::Tensor X, torch::Tensor Adj,
                                         torch::Tensor Weight, torch::Tensor WorkloadPtr,
                                         torch::Tensor Ids, torch::Tensor TilePtr,
                                         int MaxTileSize, int MinTileSize,
                                         int NumThreads, int NumWorkloads,
                                         int NumAggregatedTiles) {
    int outputSize = X.size(0) * Weight.size(0);
    auto *out = new float[outputSize];
    forwardForOneLayerFromCSCTiledParallelCombined(
            X.size(0), Adj.ccol_indices().data_ptr<int>(),
            Adj.row_indices().data_ptr<int>(), Adj.values().data_ptr<float>(),
            X.size(1), Weight.size(0), X.data_ptr<float>(), Weight.data_ptr<float>(),
            out, MinTileSize, MaxTileSize, NumThreads, NumWorkloads,
            NumAggregatedTiles, WorkloadPtr.data_ptr<int>(), Ids.data_ptr<int>(),
            TilePtr.data_ptr<int>());
    return torch::from_blob(out, {outputSize}, torch::kFloat32);
}

std::vector<torch::Tensor> createScheduleFor(torch::Tensor &Adj, int TileSize, int MinWorkloadSize) {
    DsaturColoringForConflictGraph dsaturColoring;
    MultiDimensionalSet *fusedCompSet;
    int *colPtr = Adj.ccol_indices().data_ptr<int>();
    std::cout << colPtr << std::endl;
    std::map<int, std::vector<int>> colorToTiles =
            dsaturColoring.generateGraphColoringForConflictGraphOf(
                    Adj.size(0), Adj.ccol_indices().data_ptr<int>(),
                    Adj.row_indices().data_ptr<int>(), TileSize);
    InspectorForSingleLayerTiledFusedCSCCombined inspector;
    fusedCompSet =
            inspector.generateScheduleBasedOnConflictGraphColoring(
                    colorToTiles, Adj.size(0), TileSize,
                    MinWorkloadSize);
    int numOfTiles = fusedCompSet->ptr1_[fusedCompSet->n1_] + fusedCompSet->n2_;
    torch::Tensor workloadPtr = torch::from_blob(fusedCompSet->ptr1_,{fusedCompSet->n1_ + 1}, torch::kInt32);
    torch::Tensor ids = torch::from_blob(fusedCompSet->id_, {numOfTiles}, torch::kInt32);
    torch::Tensor tilePtr = torch::from_blob(fusedCompSet->type_, {numOfTiles}, torch::kInt32);
    torch::Tensor numericalParameters = torch::empty({3}, torch::kInt32);
    numericalParameters[0] = fusedCompSet->n1_;
    numericalParameters[1] = fusedCompSet->n2_;
    numericalParameters[2] = fusedCompSet->n3_;
    return {workloadPtr, ids, tilePtr, numericalParameters};
}

torch::Tensor gcnBackward(torch::Tensor Adj, torch::Tensor X){
    return Adj;
}
#endif //FUSED_GCN_FUSIONWRAPPER_H

//
// Created by salehm32 on 05/01/24.
//
#include <torch/torch.h>
#include "FusedGCNLayerModule.h"
#include "FusionWrapper.h"
#ifndef FUSED_GCN_GCNMODULE_H
#define FUSED_GCN_GCNMODULE_H
struct GCN : torch::nn::Module {
    FusedGCNLayer layer1;
    FusedGCNLayer layer2;
    GCN(torch::Tensor Adj, torch::Tensor features, int embedDim, int tileSize, int minWorkloadSize, int numThreads){
        auto scheduleTensors = createScheduleFor(Adj, tileSize, minWorkloadSize);
        auto workloadPtr = scheduleTensors[0];
        auto ids = scheduleTensors[1];
        auto tilePtr = scheduleTensors[2];
        auto numericalTensor = scheduleTensors[3];
        auto numWorkloads = numericalTensor[0].item<int>();
        auto numAggregatedTiles = numericalTensor[1].item<int>();
        auto maxTileSize = numericalTensor[2].item<int>();
        this->layer1 = FusedGCNLayer(features.size(1), embedDim, tileSize, maxTileSize, numThreads, numWorkloads, numAggregatedTiles, workloadPtr, ids, tilePtr);
        this->layer2 = FusedGCNLayer(embedDim, embedDim, tileSize, maxTileSize, numThreads, numWorkloads, numAggregatedTiles, workloadPtr, ids, tilePtr);
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor adj){
        x = torch::relu(this->layer1.forward(x, adj));
        x = this->layer2.forward(x, adj);
        std::cout << "Forward is done..." << std::endl;
        return x;
    }
};
#endif //FUSED_GCN_GCNMODULE_H

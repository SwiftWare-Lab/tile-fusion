//
// Created by salehm32 on 05/01/24.
//
#include <torch/torch.h>
#include "FusedGCNForward.h"
#ifndef FUSED_GCN_FUSEDGCNMODULE_H
#define FUSED_GCN_FUSEDGCNMODULE_H

struct FusedGCNLayer : torch::nn::Module {
    FusedGCNLayer(int inputChannelDim, int outputChannelDim, int minTileSize, int maxTileSize, int numThreads,
                  int numWorkloads, int numAggregatedTiles, torch::Tensor workloadPtr, torch::Tensor ids, torch::Tensor tilePtr) {
        this->minTileSize = minTileSize;
        this->maxTileSize = maxTileSize;
        this->numThreads = numThreads;
        this->numWorkloads = numWorkloads;
        this->numAggregatedTiles = numAggregatedTiles;
        this->weight = register_parameter("weight", torch::randn({outputChannelDim, inputChannelDim}).requires_grad_(true));
        this->workloadPtr = workloadPtr;
        this->ids = ids;
        this->tilePtr = tilePtr;
    }
    FusedGCNLayer(){
        this->minTileSize = 0;
        this->maxTileSize = 0;
        this->numThreads = 0;
        this->numWorkloads = 0;
        this->numAggregatedTiles = 0;
        this->weight = register_parameter("weight", torch::randn({0, 0}).requires_grad_(true));
        this->workloadPtr = torch::zeros({0});
        this->ids = torch::zeros({0});
        this->tilePtr = torch::zeros({0});
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
        return FusedGCNForwardFunction::apply(x, adj, weight, workloadPtr.detach(), ids.detach(), tilePtr.detach(), maxTileSize, minTileSize,
                                              numThreads, numWorkloads, numAggregatedTiles);
    }

    int minTileSize;
    int maxTileSize;
    int numThreads;
    int numWorkloads;
    int numAggregatedTiles;
    torch::Tensor weight;
    torch::Tensor workloadPtr;
    torch::Tensor ids;
    torch::Tensor tilePtr;
};

#endif //FUSED_GCN_FUSEDGCNMODULE_H

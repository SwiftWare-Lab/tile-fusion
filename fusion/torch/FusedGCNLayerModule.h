//
// Created by salehm32 on 05/01/24.
//
#include "FusedGCNForward.h"
#include <torch/torch.h>
#ifndef FUSED_GCN_FUSEDGCNMODULE_H
#define FUSED_GCN_FUSEDGCNMODULE_H

struct CSRFusedGCNLayer : torch::nn::Module {
  CSRFusedGCNLayer(int inputChannelDim, int outputChannelDim, int levelNum,
                   int numThreads, int spMMGeMMTileSize, torch::Tensor levelPtr,
                   torch::Tensor parPtr, torch::Tensor partition, torch::Tensor mixPtr) {
    this->LevelNum = levelNum;
    this->NumThreads = numThreads;
    this->weight = register_parameter(
        "weight",
        torch::randn({outputChannelDim, inputChannelDim}).requires_grad_(true));
    this->LevelPtr = levelPtr;
    this->ParPtr = parPtr;
    this->Partition = partition;
    this->MixPtr = mixPtr;
    auto numThreadsTensor = torch::tensor({numThreads});
    auto levelNumTensor = torch::tensor({levelNum});
    auto spMMGeMMTileSizeTensor = torch::tensor({spMMGeMMTileSize});
    this->scheduleData =
        {this->LevelPtr, this->ParPtr, this->Partition,
         this->MixPtr, levelNumTensor, numThreadsTensor,
         spMMGeMMTileSizeTensor};
  }
  CSRFusedGCNLayer() {
    this->NumThreads = 0;
    this->LevelNum = 0;
    this->weight =
        register_parameter("weight", torch::randn({0, 0}).requires_grad_(true));
    this->LevelPtr = torch::zeros({0});
    this->ParPtr = torch::zeros({0});
    this->Partition = torch::zeros({0});
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
    return CSRFusedGCNForwardFunctionWithFusedBackward::apply(x, adj, weight, scheduleData);
  }

  int NumThreads;
  int LevelNum;
  torch::Tensor weight;
  torch::Tensor LevelPtr;
  torch::Tensor ParPtr;
  torch::Tensor Partition;
  torch::Tensor MixPtr;
  std::vector<torch::Tensor> scheduleData;
};

struct GCNFirstLayer : torch::nn::Module {
  GCNFirstLayer(int inputChannelDim, int outputChannelDim, int levelNum,
                   int numThreads, torch::Tensor levelPtr, torch::Tensor parPtr,
                   torch::Tensor partition, torch::Tensor mixPtr) {
    this->LevelNum = levelNum;
    this->NumThreads = numThreads;
    this->weight = register_parameter(
        "weight",
        torch::randn({outputChannelDim, inputChannelDim}).requires_grad_(true));
    this->LevelPtr = levelPtr;
    this->ParPtr = parPtr;
    this->Partition = partition;
    this->MixPtr = mixPtr;
    auto numThreadsTensor = torch::tensor({numThreads});
    auto levelNumTensor = torch::tensor({levelNum});
    this->scheduleData =
        {this->LevelPtr, this->ParPtr, this->Partition,
         this->MixPtr, levelNumTensor, numThreadsTensor};
  }
  GCNFirstLayer() {
    this->NumThreads = 0;
    this->LevelNum = 0;
    this->weight =
        register_parameter("weight", torch::randn({0, 0}).requires_grad_(true));
    this->LevelPtr = torch::zeros({0});
    this->ParPtr = torch::zeros({0});
    this->Partition = torch::zeros({0});
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
    return GCNForwardFirstLayer::apply(x, adj, weight, scheduleData);
  }

  int NumThreads;
  int LevelNum;
  torch::Tensor weight;
  torch::Tensor LevelPtr;
  torch::Tensor ParPtr;
  torch::Tensor Partition;
  torch::Tensor MixPtr;
  std::vector<torch::Tensor> scheduleData;
};


struct MKLGCNLayer : torch::nn::Module {
  MKLGCNLayer(int inputChannelDim, int outputChannelDim, int numThreads) {
    this->NumThreads = numThreads;
    this->weight = register_parameter(
        "weight",
        torch::randn({outputChannelDim, inputChannelDim}).requires_grad_(true));
  }
  MKLGCNLayer() {
    this->NumThreads = 0;
    this->weight =
        register_parameter("weight", torch::randn({0, 0}).requires_grad_(true));
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
    return GCNForwardFunctionMKL::apply(x, adj, weight, NumThreads);
  }

  int NumThreads;
  torch::Tensor weight;
};

struct CSCFusedGCNLayer : torch::nn::Module {
  CSCFusedGCNLayer(int inputChannelDim, int outputChannelDim, int minTileSize,
                   int maxTileSize, int numThreads, int numWorkloads,
                   int numAggregatedTiles, torch::Tensor workloadPtr,
                   torch::Tensor ids, torch::Tensor tilePtr) {
    this->minTileSize = minTileSize;
    this->maxTileSize = maxTileSize;
    this->numThreads = numThreads;
    this->numWorkloads = numWorkloads;
    this->numAggregatedTiles = numAggregatedTiles;
    this->weight = register_parameter(
        "weight",
        torch::randn({outputChannelDim, inputChannelDim}).requires_grad_(true));
    this->workloadPtr = workloadPtr;
    this->ids = ids;
    this->tilePtr = tilePtr;
  }
  CSCFusedGCNLayer() {
    this->minTileSize = 0;
    this->maxTileSize = 0;
    this->numThreads = 0;
    this->numWorkloads = 0;
    this->numAggregatedTiles = 0;
    this->weight =
        register_parameter("weight", torch::randn({0, 0}).requires_grad_(true));
    this->workloadPtr = torch::zeros({0});
    this->ids = torch::zeros({0});
    this->tilePtr = torch::zeros({0});
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
    return CSCFusedGCNForwardFunction::apply(
        x, adj, weight, workloadPtr.detach(), ids.detach(), tilePtr.detach(),
        maxTileSize, minTileSize, numThreads, numWorkloads, numAggregatedTiles);
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

#endif // FUSED_GCN_FUSEDGCNMODULE_H

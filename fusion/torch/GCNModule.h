//
// Created by salehm32 on 05/01/24.
//
#include <torch/torch.h>
#include "FusedGCNLayerModule.h"
#include "FusionWrapper.h"
#ifndef FUSED_GCN_GCNMODULE_H
#define FUSED_GCN_GCNMODULE_H
struct GCN : torch::nn::Module {
    std::shared_ptr<CSCFusedGCNLayer> Layer1;
    std::shared_ptr<CSCFusedGCNLayer> Layer2;
    GCN(torch::Tensor Adj, torch::Tensor features, int embedDim, int tileSize, int minWorkloadSize, int numThreads, int numClasses){
        auto scheduleTensors =
          createScheduleForCSC(Adj, tileSize, minWorkloadSize);
        auto workloadPtr = scheduleTensors[0];
        auto ids = scheduleTensors[1];
        auto tilePtr = scheduleTensors[2];
        auto numericalTensor = scheduleTensors[3];
        auto numWorkloads = numericalTensor[0].item<int>();
        auto numAggregatedTiles = numericalTensor[1].item<int>();
        auto maxTileSize = numericalTensor[2].item<int>();
        Layer1 = register_module<CSCFusedGCNLayer>("layer1", std::make_shared<CSCFusedGCNLayer>(features.size(1), embedDim, tileSize, maxTileSize, numThreads, numWorkloads, numAggregatedTiles, workloadPtr, ids, tilePtr));
        Layer2 = register_module<CSCFusedGCNLayer>("layer2", std::make_shared<CSCFusedGCNLayer>(embedDim, numClasses, tileSize, maxTileSize, numThreads, numWorkloads, numAggregatedTiles, workloadPtr, ids, tilePtr));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor adj){
        x = this->Layer1->forward(x, adj);
//        std::cout << x << std::endl;
        x = torch::relu(x);
        x = this->Layer2->forward(x, adj);
        return x;
    }
};

struct CSRFusedGCN : torch::nn::Module {
    std::shared_ptr<CSRFusedGCNLayer> Layer1;
    std::shared_ptr<CSRFusedGCNLayer> Layer2;
    CSRFusedGCN(torch::Tensor Adj, torch::Tensor Features, int EmbedDim, int NumClasses, int NumThreads, sym_lib::MultiDimensionalSet *FusedCompSet){
        auto scheduleTensors =
            createScheduleForCSR(FusedCompSet);
        auto levelPtr = scheduleTensors[0];
        auto parPtr = scheduleTensors[1];
        auto partition = scheduleTensors[2];
        auto mixPtr = scheduleTensors[3];
        auto numericalTensor = scheduleTensors[4];
        auto levelNum = numericalTensor[0].item<int>();
        Layer1 = register_module<CSRFusedGCNLayer>("layer1", std::make_shared<CSRFusedGCNLayer>(Features.size(1), EmbedDim, levelNum, NumThreads, levelPtr, parPtr, partition, mixPtr));
        Layer2 = register_module<CSRFusedGCNLayer>("layer2", std::make_shared<CSRFusedGCNLayer>(EmbedDim, NumClasses, levelNum, NumThreads, levelPtr, parPtr, partition, mixPtr));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor adj){
        x = this->Layer1->forward(x, adj);
        //        std::cout << x << std::endl;
        x = torch::relu(x);
        x = this->Layer2->forward(x, adj);
        return x;
    }
};
#endif //FUSED_GCN_GCNMODULE_H


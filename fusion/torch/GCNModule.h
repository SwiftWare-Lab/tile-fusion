//
// Created by salehm32 on 05/01/24.
//
#include <torch/torch.h>
#include "FusedGCNLayerModule.h"
#include "FusionWrapper.h"
#ifndef FUSED_GCN_GCNMODULE_H
#define FUSED_GCN_GCNMODULE_H

struct GCN : torch::nn::Module {
//    std::shared_ptr<CSCFusedGCNLayer> Layer1;
//    std::shared_ptr<CSCFusedGCNLayer> Layer2;
//    GCN(torch::Tensor Adj, torch::Tensor features, int embedDim, int tileSize, int minWorkloadSize, int numThreads, int numClasses){
//        auto scheduleTensors =
//          createScheduleForCSC(Adj, tileSize, minWorkloadSize);
//        auto workloadPtr = scheduleTensors[0];
//        auto ids = scheduleTensors[1];
//        auto tilePtr = scheduleTensors[2];
//        auto numericalTensor = scheduleTensors[3];
//        auto numWorkloads = numericalTensor[0].item<int>();
//        auto numAggregatedTiles = numericalTensor[1].item<int>();
//        auto maxTileSize = numericalTensor[2].item<int>();
//        Layer1 = register_module<CSCFusedGCNLayer>("layer1", std::make_shared<CSCFusedGCNLayer>(features.size(1), embedDim, tileSize, maxTileSize, numThreads, numWorkloads, numAggregatedTiles, workloadPtr, ids, tilePtr));
//        Layer2 = register_module<CSCFusedGCNLayer>("layer2", std::make_shared<CSCFusedGCNLayer>(embedDim, numClasses, tileSize, maxTileSize, numThreads, numWorkloads, numAggregatedTiles, workloadPtr, ids, tilePtr));
//    }

    virtual torch::Tensor forward(torch::Tensor x, torch::Tensor adj, torch::Tensor FirstLayerInput){}
};

struct CSRFusedGCN : GCN {
    std::shared_ptr<GCNFirstLayer> Layer1;
    std::shared_ptr<CSRFusedGCNLayer> Layer2;
    CSRFusedGCN(torch::Tensor Adj, torch::Tensor Features, int EmbedDim, int NumClasses, int NumThreads, sym_lib::MultiDimensionalSet *FusedCompSetL1, sym_lib::MultiDimensionalSet *FusedCompSetL2, int SpMMGeMMTileSize){
        auto scheduleTensors =
            createScheduleForCSR(FusedCompSetL1);
        auto levelPtr = scheduleTensors[0];
        auto parPtr = scheduleTensors[1];
        auto partition = scheduleTensors[2];
        auto mixPtr = scheduleTensors[3];
        auto numericalTensor = scheduleTensors[4];
        auto levelNum = numericalTensor[0].item<int>();
        Layer1 = register_module<GCNFirstLayer>("layer1", std::make_shared<GCNFirstLayer>(Features.size(1), EmbedDim, levelNum, NumThreads, levelPtr, parPtr, partition, mixPtr));
        scheduleTensors =
            createScheduleForCSR(FusedCompSetL2);
        auto levelPtr2 = scheduleTensors[0];
        auto parPtr2 = scheduleTensors[1];
        auto partition2 = scheduleTensors[2];
        auto mixPtr2 = scheduleTensors[3];
        auto numericalTensor2 = scheduleTensors[4];
        auto levelNum2 = numericalTensor[0].item<int>();
        Layer2 = register_module<CSRFusedGCNLayer>("layer2", std::make_shared<CSRFusedGCNLayer>(EmbedDim, NumClasses, levelNum2, NumThreads, SpMMGeMMTileSize, levelPtr2, parPtr2, partition2, mixPtr2));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor adj, torch::Tensor FirstLayerTensor) override{
        x = this->Layer1->forward(FirstLayerTensor, adj);
        //        std::cout << x << std::endl;
        x = torch::relu(x);
        x = this->Layer2->forward(x, adj);
        return x;
    }
};

struct MKLGCN : GCN {
    std::shared_ptr<GCNFirstLayer> Layer1;
    std::shared_ptr<MKLGCNLayer> Layer2;
    MKLGCN(torch::Tensor Adj, torch::Tensor Features, int EmbedDim, int NumClasses, int NumThreads, sym_lib::MultiDimensionalSet *FusedCompSet){
        auto scheduleTensors =
            createScheduleForCSR(FusedCompSet);
        auto levelPtr = scheduleTensors[0];
        auto parPtr = scheduleTensors[1];
        auto partition = scheduleTensors[2];
        auto mixPtr = scheduleTensors[3];
        auto numericalTensor = scheduleTensors[4];
        auto levelNum = numericalTensor[0].item<int>();
        Layer1 = register_module<GCNFirstLayer>("layer1", std::make_shared<GCNFirstLayer>(Features.size(1), EmbedDim, levelNum, NumThreads, levelPtr, parPtr, partition, mixPtr));
        Layer2 = register_module<MKLGCNLayer>("layer2", std::make_shared<MKLGCNLayer>(EmbedDim, NumClasses, NumThreads));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor adj, torch::Tensor FirstLayerInputTensor) override{
        x = this->Layer1->forward(FirstLayerInputTensor, adj);
        //        std::cout << x << std::endl;
        x = torch::relu(x);
        x = this->Layer2->forward(x, adj);
        return x;
    }
};
#endif //FUSED_GCN_GCNMODULE_H


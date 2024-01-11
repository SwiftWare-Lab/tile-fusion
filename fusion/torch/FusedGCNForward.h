//
// Created by salehm32 on 05/01/24.
//
#include <torch/torch.h>
#include "FusionWrapper.h"
#ifndef FUSED_GCN_FUSEDGCNFORWARD_H
#define FUSED_GCN_FUSEDGCNFORWARD_H

class CSCFusedGCNForwardFunction : public torch::autograd::Function<CSCFusedGCNForwardFunction> {
public:

    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor X, torch::Tensor Adj,
                                 torch::Tensor Weight, torch::Tensor WorkloadPtr,
                                 torch::Tensor Ids, torch::Tensor TilePtr,
                                 int MaxTileSize, int MinTileSize,
                                 int NumThreads, int NumWorkloads,
                                 int NumAggregatedTiles) {
        ctx->mark_non_differentiable({WorkloadPtr, Ids, TilePtr});
        int outputSize = X.size(0) * Weight.size(0);
        ctx->save_for_backward({X, Adj, Weight});
        float *out = new float[outputSize];
        memset(out, 0, outputSize * sizeof(float));
        mkl_set_num_threads(1);
        forwardForOneLayerFromCSCTiledParallelCombined(
                X.size(0), Adj.ccol_indices().data_ptr<int>(),
                Adj.row_indices().data_ptr<int>(), Adj.values().data_ptr<float>(),
                X.size(1), Weight.size(0), X.data_ptr<float>(), Weight.data_ptr<float>(),
                out, MinTileSize, MaxTileSize, NumThreads, NumWorkloads,
                NumAggregatedTiles, WorkloadPtr.data_ptr<int>(), Ids.data_ptr<int>(),
                TilePtr.data_ptr<int>());
        return torch::from_blob(out, {X.size(0), Weight.size(0)}, torch::kFloat32);
    }

    static torch::autograd::tensor_list
    backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto adj = saved[1];
        auto weight = saved[2];
        auto grad_output = grad_outputs[0];
        auto grad_input = adj.mm(grad_output.mm(weight));
        auto grad_pure_weight = adj.mm(input);
        auto grad_weight = grad_output.t().mm(grad_pure_weight);
        at::Tensor undef;
        return {grad_input, undef, grad_weight, undef, undef, undef, undef, undef, undef, undef,undef};
    }
};

class CSRFusedGCNForwardFunction : public torch::autograd::Function<CSRFusedGCNForwardFunction> {
  public:

    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor X, torch::Tensor Adj,
                                 torch::Tensor Weight, torch::Tensor LevelPtr, torch::Tensor ParPtr,
                                 torch::Tensor Partition, torch::Tensor MixPtr,
                                 int NumThreads, int LevelNum) {

//        ctx->mark_non_differentiable({WorkloadPtr, Ids, TilePtr});
        mkl_set_num_threads(1);
        int outputSize = X.size(0) * Weight.size(0);
        ctx->save_for_backward({X, Adj, Weight});
        float *out = new float[outputSize];
        memset(out, 0, outputSize * sizeof(float));
        forwardForOneLayerFusedParallelSeparated(
            X.size(0), Adj.crow_indices().data_ptr<int>(),
            Adj.col_indices().data_ptr<int>(), Adj.values().data_ptr<float>(),
            X.size(1), Weight.size(0), X.data_ptr<float>(), Weight.data_ptr<float>(), out,
            NumThreads, LevelNum, LevelPtr.data_ptr<int>(), ParPtr.data_ptr<int>(), MixPtr.data_ptr<int>(), Partition.data_ptr<int>()
            );
        return torch::from_blob(out, {X.size(0), Weight.size(0)}, torch::kFloat32);
    }

    static torch::autograd::tensor_list
    backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto adj = saved[1];
        auto weight = saved[2];
        auto grad_output = grad_outputs[0];
        auto grad_input = adj.mm(grad_output.mm(weight));
        auto grad_pure_weight = adj.mm(input);
        auto grad_weight = grad_output.t().mm(grad_pure_weight);
        at::Tensor undef;
        return {grad_input, undef, grad_weight, undef, undef, undef, undef, undef, undef};
    }
};

class CSRUnFusedGCNForwardFunction : public torch::autograd::Function<CSRUnFusedGCNForwardFunction> {
  public:

    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor X, torch::Tensor Adj,
                                 torch::Tensor Weight) {
        int outputSize = X.size(0) * Weight.size(0);
        ctx->save_for_backward({X, Adj, Weight});
        auto out = Adj.mm(X.mm(Weight.t()));
//        for (int i = 0; i < X.size(0); i++){
//          for (int j = 0; j < Weight.size(0); j++){
//            std::cout << out[i * Weight.size(0) + j] << " ";
//          }
//          std::cout << std::endl;
//        }
        return out;
    }

    static torch::autograd::tensor_list
    backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto adj = saved[1];
        auto weight = saved[2];
        auto grad_output = grad_outputs[0];
        auto grad_input = adj.mm(grad_output.mm(weight));
        auto grad_weight = grad_output.t().mm(adj.mm(input));
        at::Tensor undef;
        return {grad_input, undef, grad_weight};
    }
};

#endif //FUSED_GCN_FUSEDGCNFORWARD_H

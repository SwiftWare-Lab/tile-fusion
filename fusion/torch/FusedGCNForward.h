//
// Created by salehm32 on 05/01/24.
//
#include <torch/torch.h>
#include "FusionWrapper.h"
#ifndef FUSED_GCN_FUSEDGCNFORWARD_H
#define FUSED_GCN_FUSEDGCNFORWARD_H

class FusedGCNForwardFunction : public torch::autograd::Function<FusedGCNForwardFunction> {
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

#endif //FUSED_GCN_FUSEDGCNFORWARD_H

//
// Created by salehm32 on 05/01/24.
//
#include "FusionWrapper.h"
#include <mkl.h>
#include <torch/torch.h>
#ifndef FUSED_GCN_FUSEDGCNFORWARD_H
#define FUSED_GCN_FUSEDGCNFORWARD_H

class CSCFusedGCNForwardFunction
    : public torch::autograd::Function<CSCFusedGCNForwardFunction> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor X, torch::Tensor Adj,
                               torch::Tensor Weight, torch::Tensor WorkloadPtr,
                               torch::Tensor Ids, torch::Tensor TilePtr,
                               int MaxTileSize, int MinTileSize, int NumThreads,
                               int NumWorkloads, int NumAggregatedTiles) {
    ctx->mark_non_differentiable({WorkloadPtr, Ids, TilePtr});
    int outputSize = X.size(0) * Weight.size(0);
    ctx->save_for_backward({X, Adj, Weight});
    float *out = new float[outputSize];
    memset(out, 0, outputSize * sizeof(float));
    mkl_set_num_threads(1);
    forwardForOneLayerFromCSCTiledParallelCombined(
        X.size(0), Adj.ccol_indices().data_ptr<int>(),
        Adj.row_indices().data_ptr<int>(), Adj.values().data_ptr<float>(),
        X.size(1), Weight.size(0), X.data_ptr<float>(),
        Weight.data_ptr<float>(), out, MinTileSize, MaxTileSize, NumThreads,
        NumWorkloads, NumAggregatedTiles, WorkloadPtr.data_ptr<int>(),
        Ids.data_ptr<int>(), TilePtr.data_ptr<int>());
    return torch::from_blob(out, {X.size(0), Weight.size(0)}, torch::kFloat32);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto adj = saved[1];
    auto weight = saved[2];
    auto grad_output = grad_outputs[0];
    auto grad_input = adj.mm(grad_output.mm(weight));
    auto grad_pure_weight = adj.mm(input);
    auto grad_weight = grad_output.t().mm(grad_pure_weight);
    at::Tensor undef;
    return {grad_input, undef, grad_weight, undef, undef, undef,
            undef,      undef, undef,       undef, undef};
  }
};

class CSRFusedGCNForwardFunction
    : public torch::autograd::Function<CSRFusedGCNForwardFunction> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor X, torch::Tensor Adj,
                               torch::Tensor Weight, torch::Tensor LevelPtr,
                               torch::Tensor ParPtr, torch::Tensor Partition,
                               torch::Tensor MixPtr, int NumThreads,
                               int LevelNum) {

    //        ctx->mark_non_differentiable({WorkloadPtr, Ids, TilePtr});
    mkl_set_num_threads(1);
    int outputSize = X.size(0) * Weight.size(0);
    ctx->save_for_backward({X, Adj, Weight});
    float *out = new float[outputSize];
    memset(out, 0, outputSize * sizeof(float));
    forwardForOneLayerFusedParallelSeparatedVectorizedSP(
        X.size(0), Adj.crow_indices().data_ptr<int>(),
        Adj.col_indices().data_ptr<int>(), Adj.values().data_ptr<float>(),
        X.size(1), Weight.size(0), X.data_ptr<float>(),
        Weight.data_ptr<float>(), out,  NumThreads, LevelNum,
        LevelPtr.data_ptr<int>(), ParPtr.data_ptr<int>(),
        MixPtr.data_ptr<int>(), Partition.data_ptr<int>());
    mkl_set_num_threads(NumThreads);
    return torch::from_blob(
        out, {X.size(0), Weight.size(0)},
        [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    sparse_matrix_t MKLAdj;
    matrix_descr d;
    d.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto adj = saved[1];
    auto weight = saved[2];
    int *adjPtr = adj.crow_indices().data_ptr<int>();
    int *adjIndex = adj.col_indices().data_ptr<int>();
    mkl_sparse_s_create_csr(&MKLAdj, SPARSE_INDEX_BASE_ZERO, adj.size(0),
                            adj.size(1), adjPtr, adjPtr + 1, adjIndex,
                            adj.values().data_ptr<float>());
    auto grad_output = grad_outputs[0];
    float *grad_output_raw = grad_output.data_ptr<float>();
    float *grad_intermediate = new float[adj.size(0) * grad_output.size(1)];
    mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, MKLAdj, d,
                    SPARSE_LAYOUT_ROW_MAJOR, grad_output_raw,
                    grad_output.size(1), grad_output.size(1), 0,
                    grad_intermediate, grad_output.size(1));
    torch::Tensor grad_input;
    if (ctx->needs_input_grad(0)) {
      float *weight_raw = weight.data_ptr<float>();
      float *grad_input_raw = new float[adj.size(0) * weight.size(1)];
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, adj.size(0),
                  weight.size(1), weight.size(0), 1., grad_intermediate,
                  grad_output.size(1), weight_raw, weight.size(1), 0.,
                  grad_input_raw, weight.size(1));
      grad_input = torch::from_blob(
          grad_input_raw, {(long)grad_output.size(0), (long)weight.size(1)},
          [](void *ptr) { delete[] static_cast<float *>(ptr); },
          torch::kFloat32);
    }
    float *grad_weight_raw = new float[grad_output.size(1) * input.size(1)];
    float *input_raw = input.data_ptr<float>();
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, grad_output.size(1),
                input.size(1), adj.size(0), 1., grad_intermediate,
                grad_output.size(1), input_raw, input.size(1), 0.,
                grad_weight_raw, input.size(1));
    mkl_free(MKLAdj);
    delete[] grad_intermediate;
    auto grad_weight = torch::from_blob(
        grad_weight_raw, {grad_output.size(1), input.size(1)},
        [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
    at::Tensor undef;
    return {grad_input, undef, grad_weight, undef, undef,
            undef,      undef, undef,       undef};
  }
};

class CSRFusedGCNForwardFunctionWithFusedBackward
    : public torch::autograd::Function<
          CSRFusedGCNForwardFunctionWithFusedBackward> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor X, torch::Tensor Adj,
                               torch::Tensor Weight, torch::Tensor ScheduleData,
                               torch::Tensor LevelPtr, torch::Tensor ParPtr,
                               torch::Tensor Partition, torch::Tensor MixPtr,
                               int NumThreads, int LevelNum) {

    //        ctx->mark_non_differentiable({WorkloadPtr, Ids, TilePtr});
    mkl_set_num_threads(1);
    int outputSize = X.size(0) * Weight.size(0);
    ctx->save_for_backward({X, Adj, Weight, ScheduleData});
    float *out = new float[outputSize];
    memset(out, 0, outputSize * sizeof(float));
    forwardForOneLayerFusedParallelSeparated(
        X.size(0), Adj.crow_indices().data_ptr<int>(),
        Adj.col_indices().data_ptr<int>(), Adj.values().data_ptr<float>(),
        X.size(1), Weight.size(0), X.data_ptr<float>(),
        Weight.data_ptr<float>(), out, NumThreads, LevelNum,
        LevelPtr.data_ptr<int>(), ParPtr.data_ptr<int>(),
        MixPtr.data_ptr<int>(), Partition.data_ptr<int>());
    mkl_set_num_threads(NumThreads);
    return torch::from_blob(out, {X.size(0), Weight.size(0)}, torch::kFloat32);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    sparse_matrix_t MKLAdj;
    matrix_descr d;
    d.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto adj = saved[1];
    auto weight = saved[2];
    auto scheduleData = saved[3];
    int *LevelPtr = scheduleData[0].data_ptr<int>();
    int *ParPtr = scheduleData[1].data_ptr<int>();
    int *Partition = scheduleData[2].data_ptr<int>();
    int *MixPtr = scheduleData[3].data_ptr<int>();
    int LevelNum = scheduleData[4][0].item<int>();
    int ThreadNum = scheduleData[5][0].item<int>();
    int *adjPtr = adj.crow_indices().data_ptr<int>();
    int *adjIndex = adj.col_indices().data_ptr<int>();
    mkl_sparse_s_create_csr(&MKLAdj, SPARSE_INDEX_BASE_ZERO, adj.size(0),
                            adj.size(1), adjPtr, adjPtr + 1, adjIndex,
                            adj.values().data_ptr<float>());
    auto grad_output = grad_outputs[0];
    float *grad_output_raw = grad_output.data_ptr<float>();
    float *weight_raw = weight.data_ptr<float>();
    float *grad_input_intermediate =
        new float[grad_output.size(0) * weight.size(1)];
    float *grad_input_raw = new float[adj.size(0) * weight.size(1)];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, adj.size(0),
                weight.size(1), weight.size(0), 1., grad_output_raw,
                grad_output.size(1), weight_raw, weight.size(1), 0.,
                grad_input_intermediate, weight.size(1));
    mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, MKLAdj, d,
                    SPARSE_LAYOUT_ROW_MAJOR, grad_input_intermediate,
                    weight.size(1), weight.size(1), 0, grad_input_raw,
                    weight.size(1));
    delete[] grad_input_intermediate;
    auto grad_input = torch::from_blob(
        grad_input_raw, {(long)grad_output.size(0), (long)weight.size(1)},
        [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
    float *grad_weight_intermediate = new float[adj.size(0) * input.size(1)];
    float *grad_weight_raw = new float[grad_output.size(1) * input.size(1)];
    float *input_raw = input.data_ptr<float>();
    mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, MKLAdj, d,
                    SPARSE_LAYOUT_ROW_MAJOR, input_raw, input.size(1),
                    input.size(1), 0, grad_weight_intermediate, input.size(1));
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, grad_output.size(1),
                input.size(1), grad_output.size(0), 1., grad_output_raw,
                grad_output.size(1), grad_weight_intermediate, input.size(1),
                0., grad_weight_raw, input.size(1));
    delete[] grad_weight_intermediate;
    mkl_free(MKLAdj);
    //    auto grad_pure_weight = adj.mm(input);
    //    auto grad_weight = grad_output.t().mm(grad_pure_weight);
    auto grad_weight = torch::from_blob(
        grad_weight_raw, {grad_output.size(1), input.size(1)},
        [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
    at::Tensor undef;
    return {grad_input, undef, grad_weight, undef, undef,
            undef,      undef, undef,       undef};
  }
};

class CSRUnFusedGCNForwardFunction
    : public torch::autograd::Function<CSRUnFusedGCNForwardFunction> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor X, torch::Tensor Adj,
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
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto adj = saved[1];
    auto weight = saved[2];
    auto grad_output = grad_outputs[0];
    auto test = adj.mm(grad_output);
    auto grad_input = adj.mm(grad_output.mm(weight));
    auto grad_weight = grad_output.t().mm(adj.mm(input));
    at::Tensor undef;
    return {grad_input, undef, grad_weight};
  }
};

#endif // FUSED_GCN_FUSEDGCNFORWARD_H

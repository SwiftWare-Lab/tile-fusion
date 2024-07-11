//
// Created by salehm32 on 03/07/24.
//

#ifndef SPARSE_FUSION_FUNCTIONS_H
#define SPARSE_FUSION_FUNCTIONS_H
#include <torch/torch.h>
#include <vector>
#include <mkl.h>
#include <immintrin.h>

#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(a, b) ((a < b) ? a : b)
//inputs: NumThreads, MTile, CSR Matrix
//outputs: LevelPtr, MixPtr, Partition(Id)

std::vector<int*> createSchedule(int32_t* Ap, int32_t* Ai, int64_t ARows, int64_t MTileSize);

void fusedMKLGeMMSpMM(
        int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
        int OutputChannelDim, float *Features, float *Weight, float *Output,
        int NumThreads, int LevelNo, const int *LevelPtr, const int *MixPtr, const int *Partition);

void fusedMKLGeMMSpMMTransposedWeight(
        int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
        int OutputChannelDim, float *Features, float *Weight, float *Output,
        int NumThreads, int LevelNo, const int *LevelPtr, const int *MixPtr, const int *Partition);

class FusedGeMMSpMM: public torch::autograd::Function<FusedGeMMSpMM>{
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *Ctx,
                                 torch::Tensor Adj, torch::Tensor Feature, torch::Tensor Weight,
                                 torch::Tensor LevelPtr, torch::Tensor MixPtr, torch::Tensor Partition,
                                 int64_t NumThreads);

    static torch::autograd::tensor_list
    backward(torch::autograd::AutogradContext *Ctx,
             torch::autograd::tensor_list GradOutputs);
};
#endif // SPARSE_FUSION_FUNCTIONS_H

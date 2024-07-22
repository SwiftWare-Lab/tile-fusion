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


struct VariableTile{
    int Start;
    int End;
    std::vector<int> FusedIters;
    VariableTile* Next;
    VariableTile(int Start, int End){
        this->Start = Start;
        this->End = End;
        this->Next = nullptr;
    }
};

int** generateVariableTileSizeScheduleGeMMSpMM(int M, int* Ap, int* Ai, int BCol, int CCol, int CacheSize,
                                                           int DataSize= 4);

void createReorderedAdj(int M, int NNZ, int* Ap, int* Ai, float* Ax, int *LevelPtr, int *MixPtr, int *Id,
                        int* UFAp, int* UFAi, float* UFAx, int* NewMixPtr,  int* L2Ptr);

int findInitialTileSize(int BCol, int CCol, int MaxWSSize, int DataSize = 4);
int calculateWorkingSetSize(int Nnz, int UniqueColsNum, int BCol, int TileSize, int FusedIters, int DataSize = 4);


std::vector<int*> createSchedule(int32_t* Ap, int32_t* Ai, int64_t ARows, int64_t MTileSize);

void fusedMKLGeMMSpMM(
        int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
        int OutputChannelDim, float *Features, float *Weight, float *Output,
        int NumThreads, int LevelNo, const int *LevelPtr, const int *MixPtr, const int *Partition);

void fusedMKLGeMMSpMMTransposedWeight(
        int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
        int OutputChannelDim, float *Features, float *Weight, float *Output,
        int NumThreads, int LevelNo, const int *LevelPtr, const int *MixPtr, const int *Partition);
void
registerReuseVectorizedSpMM(const int *Ap, const int *Ai, const float *Ax, const int OutputChannelDim, float *Output,
                            const int *L2Ptr, int residueStart, const float *intermediateResult, int kBeginL2,
                            int kEndL2);

void
perfectSpatialLocalitySpMM(const int *Ap, const int *Ai, const float *Ax, const int OutputChannelDim, float *Output,
                           const int *L2Ptr, int residueStart, const float *intermediateResult, int kBeginL2,
                           int kEndL2);

class FusedGeMMSpMMROAdj: public torch::autograd::Function<FusedGeMMSpMMROAdj>{
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *Ctx,
                                 torch::Tensor Adj, torch::Tensor ROAdj, torch::Tensor Feature, torch::Tensor Weight,
                                 torch::Tensor LevelPtr, torch::Tensor MixPtr, torch::Tensor Partition,
                                 int64_t NumThreads);

    static torch::autograd::tensor_list
    backward(torch::autograd::AutogradContext *Ctx,
             torch::autograd::tensor_list GradOutputs);
};

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

class FusedGeMMSpMMROAdjCaching: public torch::autograd::Function<FusedGeMMSpMMROAdj>{
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *Ctx,
                                 torch::Tensor Adj, torch::Tensor ROAdj, torch::Tensor Feature, torch::Tensor Weight,
                                 torch::Tensor LevelPtr, torch::Tensor MixPtr, torch::Tensor Partition,
                                 int64_t NumThreads);

    static torch::autograd::tensor_list
    backward(torch::autograd::AutogradContext *Ctx,
             torch::autograd::tensor_list GradOutputs);
};
#endif // SPARSE_FUSION_FUNCTIONS_H

//
// Created by salehm32 on 2026-05-08.
//

#ifndef SPARSE_FUSION_FUNCTIONS_H
#define SPARSE_FUSION_FUNCTIONS_H
// #include <torch/torch.h>
#include <vector>
#include <mkl.h>
#include <immintrin.h>
#include <set>

#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(a, b) ((a < b) ? a : b)
namespace swiftware{
  namespace fusion{
    int** generateVariableTileSizeScheduleGeMMSpMM(int M, int* Ap, int* Ai, int BCol, int CCol, int CacheSize, int NumThreads,int DataSize);
    int** generateVariableTileSizeScheduleSpMMSpMM(int M, int* Ap, int* Ai, int FeatureDim, int CacheSize, int NumThreads, int DataSize);
    int** createSpMMSpMMFixedSchedule(int M, int MTileSize);
    std::vector<int*> createSchedule(int32_t* Ap, int32_t* Ai, int64_t ARows, int64_t MTileSize);
    int calculateWorkingSetSizeForGCNLayer(int Nnz, int UniqueColNnz, int BCol, int CCol, int TileSize, int DataSize);
    int calculateWorkingSetSizeForSpMMSpMM(int Nnz, int UniqueColNnz, int FeatureDim, int TileSize, int FusedTileSize, int DataSize);
    void fusedMKLGeMMSpMM(
        int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
        int OutputChannelDim, float *Features, float *Weight, float *Output,
        int NumThreads, int LevelNo, const int *LevelPtr, const int *MixPtr, const int *Partition);
    void fusedMKLGeMMSpMMTransposedWeight(
        int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
        int OutputChannelDim, float *Features, float *Weight, float *Output,
        int NumThreads, int LevelNo, const int *LevelPtr, const int *MixPtr, const int *Partition);
    void fusedMKLGeMMSpMMTransposedWeightVT(
      const int M, const int *__restrict__ Ap, const int *__restrict__ Ai,
      const float *__restrict__ Ax, const int InputChannelDim,
      const int OutDim1, const float * Features,
      const float * Weight1, float * Output,
      const int NumThreads, const int LevelNo, const int *LevelPtr,
      const int *MixPtr, const int *L2Ptr);
    void fusedMKLGeMMSpMMVT(
      const int M, const int *__restrict__ Ap, const int *__restrict__ Ai,
      const float *__restrict__ Ax, const int InputChannelDim,
      const int OutDim1, const float * Features,
      const float * Weight1, float * Output,
      const int NumThreads, const int LevelNo, const int *LevelPtr,
      const int *MixPtr, const int *L2Ptr);
    void fusedSpMMSpMMVectorizedAvx256SP(
        int M, int FeatureDim, const int *Ap, const int *Ai, const float *Ax,
        const int *Bp, const int *Bi, const float *Bx, const float *Features,
        float *Output, float *Intermediate, int LevelNo, const int *LevelPtr,
        const int *ParPtr, const int *Partition, const int *ParType,
        int NumThreads);
    inline void rowWiseSpMMKernel(const int *Ap, const int *Ai, const float *Ax, const int OutputChannelDim, float *Output,
                                const int *L2Ptr, int residueStart, const float *intermediateResult, int kBeginL2,
                                int kEndL2);
  }
}


#endif // SPARSE_FUSION_FUNCTIONS_H

//
// Created by salehm32 on 03/07/24.
//

#include "Functions.h"

std::vector<int*> createSchedule(int32_t* Ap, int32_t* Ai, int64_t ARows, int64_t MTileSize){
    int *levelPtr;
    int *mixPtr;
    int *partition;
    int wf1NumTiles = CEIL(ARows, MTileSize);
    int maxKernelPerWF = 2;
    std::vector<int> ufRows;
    std::vector<std::vector<int>> fRows(wf1NumTiles);
    int rowTile = MTileSize;
    int nnzCount = 0;
    for (int i = 0; i < ARows; i+=rowTile) {
        int t = i / rowTile;
        int end = MIN(i + rowTile, ARows);
        for (int ii = i; ii < end; ii++){
            bool isUnfused = false;
            for (int j = Ap[ii]; j < Ap[ii + 1]; j++){
                if (Ai[j] < i || Ai[j] >= end){
                    ufRows.push_back(ii);
                    isUnfused = true;
                    break;
                }
            }
            if (!isUnfused){
                fRows[t].push_back(ii);
                nnzCount += Ap[ii + 1] - Ap[ii];
            }
        }
    }
    int ufCount = ufRows.size();
    int wf2NumTiles = CEIL(ufCount, MTileSize);
//    int fIdCount = ARows - ufCount;
//    int fPtrCount = wf1NumTiles + 1;
    int totalPartNum = wf1NumTiles + wf2NumTiles;
    levelPtr = new int[maxKernelPerWF + 1];
    levelPtr[0] = 0;
    levelPtr[1] = wf1NumTiles;
    levelPtr[2] = totalPartNum;
    mixPtr = new int[maxKernelPerWF*(totalPartNum) + 1];
    partition = new int[ARows*2];
    mixPtr[0] = 0;
    for (int i = 0; i < wf1NumTiles; i++){
        int start1 = mixPtr[maxKernelPerWF*i];
        int start2;
        if (i < wf1NumTiles-1 || (ARows % MTileSize) == 0){
            start2 = mixPtr[maxKernelPerWF*i] + MTileSize;
        }
        else{
            start2 = mixPtr[maxKernelPerWF*(i)] + (ARows % MTileSize);
        }
        mixPtr[maxKernelPerWF*(i) + 1] = start2;
        for (int j = start1; j < start2; j++){
            partition[j] = i * MTileSize + j - start1;
        }
        int end2 = mixPtr[maxKernelPerWF*i + 1] + fRows[i].size();
        mixPtr[maxKernelPerWF*(i+1)] = end2;
        for (int j = start2; j < end2; j++){
            partition[j] = fRows[i][j-start2];
        }
    }
    int wf2Start = mixPtr[maxKernelPerWF*wf1NumTiles];
    for (int i = 0; i < wf2NumTiles; i+=1){
        int tileUfCount = i == (wf2NumTiles - 1) && ((ufCount % MTileSize) != 0) ? ufCount % MTileSize : MTileSize;
        int t = i + wf1NumTiles;
        int start1 = mixPtr[maxKernelPerWF*t];
        int start2 = start1;
        int end2 = start1 + tileUfCount;
        for (int ii = start2; ii < end2; ii++){
            partition[ii] = ufRows[ii-wf2Start];
        }
        mixPtr[maxKernelPerWF*(t) + 1] = start2;
        mixPtr[maxKernelPerWF*(t+1)] = end2;
    }
    int* numericalValues = new int[2];
    numericalValues[0] = maxKernelPerWF*(totalPartNum) + 1;
    numericalValues[1] = ARows*2;
    return {levelPtr, mixPtr, partition, numericalValues};
}

void fusedMKLGeMMSpMM(
        int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
        int OutputChannelDim, float *Features, float *Weight, float *Output,
        int NumThreads, int LevelNo, const int *LevelPtr, const int *MixPtr, const int *Partition) {
    int numKernels = 2;
    int residueStart = OutputChannelDim - OutputChannelDim%32;
    float *intermediateResult = new float[M * OutputChannelDim];
    for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
        {
#pragma omp for
            for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
                int kBeginL1 = MixPtr[j1 * numKernels];
                int kEndL1 = MixPtr[j1 * numKernels + 1];
                int iL1 = Partition[kBeginL1];
                int tileSize = kEndL1 - kBeginL1;
                cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize, OutputChannelDim,
                        InputChannelDim, 1., Features + iL1 * InputChannelDim,
                        InputChannelDim, Weight, OutputChannelDim, 0.,
                        intermediateResult + iL1 * OutputChannelDim, OutputChannelDim);
                int kEndL2 = MixPtr[(j1+1) * numKernels];
                for (int k1 = kEndL1; k1 < kEndL2; ++k1) {
                    int i = Partition[k1];
                    for (int kk = 0; kk < residueStart; kk += 32) {
                        int ip = i * OutputChannelDim;
                        auto dxV1 = _mm256_loadu_ps(Output + ip + kk);
                        auto dxV2 = _mm256_loadu_ps(Output + ip + kk + 8);
                        auto dxV3 = _mm256_loadu_ps(Output + ip + kk + 16);
                        auto dxV4 = _mm256_loadu_ps(Output + ip + kk + 24);
                        int k = Ap[i];
                        for (; k < Ap[i + 1]-1; k+=2) {
                            int bij1 = Ai[k] * OutputChannelDim;
                            int bij2 = Ai[k+1] * OutputChannelDim;
                            auto bxV1 = _mm256_set1_ps(Ax[k]);
                            auto bxV2 = _mm256_set1_ps(Ax[k+1]);
                            auto acxV11 = _mm256_loadu_ps(intermediateResult + bij1 + kk);
                            auto acxV12 = _mm256_loadu_ps(intermediateResult + bij1 + kk + 8);
                            auto acxV13 = _mm256_loadu_ps(intermediateResult + bij1 + kk + 16);
                            auto acxV14 = _mm256_loadu_ps(intermediateResult + bij1 + kk + 24);
                            auto acxV21 = _mm256_loadu_ps(intermediateResult + bij2 + kk);
                            auto acxV22 = _mm256_loadu_ps(intermediateResult + bij2 + kk + 8);
                            auto acxV23 = _mm256_loadu_ps(intermediateResult + bij2 + kk + 16);
                            auto acxV24 = _mm256_loadu_ps(intermediateResult + bij2 + kk + 24);
                            dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
                            dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
                            dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
                            dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
                            dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
                            dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
                            dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
                            dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
                        }
                        for (; k < Ap[i + 1]; ++k) {
                            int bij = Ai[k] * OutputChannelDim;
                            auto bxv0 = _mm256_set1_ps(Ax[k]);
                            auto cxV11 = _mm256_loadu_ps(intermediateResult + bij + kk);
                            auto cxV12 = _mm256_loadu_ps(intermediateResult + bij + kk + 8);
                            auto cxV13 = _mm256_loadu_ps(intermediateResult + bij + kk + 16);
                            auto cxV14 = _mm256_loadu_ps(intermediateResult + bij + kk + 24);
                            dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
                            dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
                            dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
                            dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
                        }
                        _mm256_storeu_ps(Output + ip + kk, dxV1);
                        _mm256_storeu_ps(Output + ip + kk + 8, dxV2);
                        _mm256_storeu_ps(Output + ip + kk + 16, dxV3);
                        _mm256_storeu_ps(Output + ip + kk + 24, dxV4);
                    }
                    for (int k = Ap[i]; k < Ap[i + 1]; k++) {
                        int ip = OutputChannelDim * i;
                        for (int kk = residueStart; kk < OutputChannelDim; kk++) {
                            Output[ip + kk] +=
                                    Ax[k] * intermediateResult[Ai[k] * OutputChannelDim + kk];
                        }
                    }
                }
            }
        }
    }
    delete[] intermediateResult;
}
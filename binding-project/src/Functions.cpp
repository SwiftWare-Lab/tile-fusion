//
// Created by salehm32 on 03/07/24.
//

#include "Functions.h"

void spmmKernel(const int *Ap, const int *Ai, const float *Ax, int N, const float *B, float *Output, int residueStart,
                int row);

int**
generateVariableTileSizeScheduleGeMMSpMM(int M, int* Ap, int* Ai, int BCol, int CCol, int CacheSize, int NumThreads,int DataSize){
    //create initial tiles
    int MIN_STRIDE = 16;
    int variableStride = MIN_STRIDE;
    int maxStride = M / NumThreads;
    std::set<int> uniqueColumns;
    int nnzNum = 0;
    int uft = 0;
    int ufTileSize = 0;
    std::vector<int> partPtr;
    partPtr.push_back(0);
    while (uft < M){
        for (int ii = uft; ii < std::min(M, uft + variableStride); ii++){
            int row = ii;
            nnzNum += Ap[row + 1] - Ap[row];
            uniqueColumns.insert(Ai + Ap[row], Ai + Ap[row + 1]);
            ufTileSize += 1;
        }
        int workingSet = calculateWorkingSetSizeForGCNLayer(nnzNum, uniqueColumns.size(), BCol, CCol, ufTileSize, DataSize);
        if(((workingSet < CacheSize) || (ufTileSize <= MIN_STRIDE)) && ufTileSize < maxStride){
            uft += variableStride;
        }
        else{
            nnzNum = 0;
            uniqueColumns.clear();
            if (ufTileSize <= variableStride){
                variableStride = variableStride / 2;
            }
            else{
                partPtr.push_back(uft);
            }
            if (ufTileSize >= 3*variableStride){
                variableStride = variableStride * 2;
            }
            ufTileSize = 0;
        }
    }
    partPtr.push_back(M);
    int partPerWF = partPtr.size() - 1;
    int* ptr1 = new int[3];
    ptr1[0] = 0;
    ptr1[1] = partPerWF;
    ptr1[2] = partPerWF*2;
    int* kerBegin = new int[(partPerWF*2)*2+2];
    int* id = new int[M];
    int cnt = 0;
    int pCounter = 0;
    kerBegin[0] = 0;
    kerBegin[1] = 0;
    std::vector<std::vector<int>> ufTiles(partPerWF);
    for (int i = 0; i < partPtr.size()-1; i++){
        int start = partPtr[i];
        int end = partPtr[i+1];
        kerBegin[(i+1)*2] = end;
        for (int j = start; j < end; j++){
            int fused = true;
            for (int k = Ap[j]; k < Ap[j+1]; k++){
                if(Ai[k] >= end || Ai[k] < start){
                    fused=false;
                    break;
                }
            }
            if (fused){
                id[cnt] = j;
                cnt+=1;
            }
            else{
                ufTiles[i].push_back(j);
            }
        }

        kerBegin[(i+1)*2 + 1] = cnt;
    }

    for (int i = 0; i < partPtr.size()-1; i++){
        int p = i + partPerWF;
        for (int j = 0; j < ufTiles[i].size(); j++){
            id[cnt] = ufTiles[i][j];
            cnt++;
        }
        kerBegin[(p+1)*2] =  kerBegin[p*2];
        kerBegin[(p+1)*2 + 1] = cnt;
    }
    int** out = new int*[3];
    out[0] = ptr1;
    out[1] = kerBegin;
    out[2] = id;
    return out;
}


int calculateWorkingSetSize(int Nnz, int UniqueColsNum, int BCol, int TileSize, int FusedIters, int DataSize){
    return (Nnz + UniqueColsNum * BCol + TileSize * BCol + FusedIters * BCol) * DataSize + Nnz*4;
}

int calculateWorkingSetSizeForGeMM(int FusedNnz, int BCol, int CCol, int TileSize, int FusedIters, int DataSize){
    return (TileSize * CCol + CCol * BCol + TileSize * BCol + FusedIters * CCol + FusedNnz) * DataSize + FusedNnz * 4;
}

int calculateWorkingSetSizeForGCNLayer(int Nnz, int UniqueColNnz, int BCol, int CCol, int TileSize, int DataSize){
    return (CCol * BCol * 2 + TileSize * CCol + TileSize * BCol + UniqueColNnz * BCol + UniqueColNnz * CCol) * DataSize + Nnz * 4;
}

int findInitialTileSize(int BCol, int CCol, int MaxWSSize, int DataSize){
    return (MaxWSSize/DataSize - BCol*CCol) / (BCol + CCol);
}

void createReorderedAdj(int M, int NNZ, int* Ap, int* Ai, float* Ax, int *LevelPtr, int *MixPtr, int *Id,
                        int* UFAp, int* UFAi, float* UFAx, int* NewMixPtr, int* L2Ptr){
    int numKernels = 2;
    int nnzCount = 0;
    int rowCountL2 = 0;
    UFAp[0] = 0;
    NewMixPtr[0] = 0;
    NewMixPtr[1] = 0;
    for (int l1 = 0; l1 < numKernels; l1++){
        for (int p1 = LevelPtr[l1]; p1 < LevelPtr[l1+1]; p1++){
            int kBeginL2 = MixPtr[p1 * numKernels];
            NewMixPtr[(p1 + 1) * numKernels] = kBeginL2 - rowCountL2;
            int kEndL2 = MixPtr[p1 * numKernels + 1];
            for (int i1=kBeginL2; i1 < kEndL2; i1++) {
                int oldRow = Id[i1];
                L2Ptr[rowCountL2] = oldRow;
                for (int j = Ap[oldRow]; j < Ap[oldRow + 1]; j++){
                    UFAx[nnzCount] = Ax[j];
                    UFAi[nnzCount] = Ai[j];
                    nnzCount += 1;
                }
                rowCountL2 += 1;
                UFAp[rowCountL2] = nnzCount;
            }
            NewMixPtr[(p1 + 1) * numKernels + 1] = rowCountL2;
        }
    }
}

void fusedGeMMSpMMReorderedAdjVectorizedTransposedWeight(
        const int M, const int *__restrict__ Ap, const int *__restrict__ Ai,
        const float *__restrict__ Ax, const int InputChannelDim,
        const int OutputChannelDim, const float * Features,
        const float *Weight, float * Output,
        const int NumThreads, const int LevelNo, const int *LevelPtr,
        const int * MixPtr, const int * L2Ptr) {
    int numKernels = 2;
    int k1Counter = 0;
    float *intermediateResult = new float[M * OutputChannelDim]{};
    int residueStart = OutputChannelDim - OutputChannelDim%32;
    for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
        {
#pragma omp for
            for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
                int kBeginL1 = MixPtr[j1 * numKernels];
                int kEndL1 = MixPtr[(j1+1) * numKernels];
                int tileSize = kEndL1 - kBeginL1;
                cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans, tileSize, OutputChannelDim,
                        InputChannelDim, 1., Features + kBeginL1 * InputChannelDim,
                        InputChannelDim, Weight, InputChannelDim, 0.,
                        intermediateResult + kBeginL1 * OutputChannelDim, OutputChannelDim);
                int kBeginL2 = MixPtr[j1 * numKernels + 1];
                int kEndL2 = MixPtr[(j1+1) * numKernels + 1];
#ifdef AVX512
                perfectSpatialLocalitySpMMAVX512OutDim64(Ap, Ai, Ax, OutputChannelDim, Output, L2Ptr, residueStart,
                                           intermediateResult, kBeginL2, kEndL2);
#else
                perfectSpatialLocalitySpMM(Ap, Ai, Ax, OutputChannelDim, Output, L2Ptr, residueStart,
                                           intermediateResult, kBeginL2, kEndL2);
#endif
            }
        }
    }
    delete[] intermediateResult;
}

void fusedGeMMSpMMReorderedAdjVectorized(
        const int M, const int *__restrict__ Ap, const int *__restrict__ Ai,
        const float *__restrict__ Ax, const int InputChannelDim,
        const int OutputChannelDim, const float * Features,
        const float * Weight, float * Output,
        const int NumThreads, const int LevelNo, const int *LevelPtr,
        const int *MixPtr, const int *L2Ptr) {
    int numKernels = 2;
    int k1Counter = 0;
    int residueStart = OutputChannelDim - OutputChannelDim%16;
    float *intermediateResult = new float[M * OutputChannelDim]{};
    for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
        {
#pragma omp for
            for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
                int kBeginL1 = MixPtr[j1 * numKernels];
                int kEndL1 = MixPtr[(j1+1) * numKernels];
                int tileSize = kEndL1 - kBeginL1;
                k1Counter += tileSize;
                cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize, OutputChannelDim,
                        InputChannelDim, 1., Features + kBeginL1 * InputChannelDim,
                        InputChannelDim, Weight, OutputChannelDim, 0.,
                        intermediateResult + kBeginL1 * OutputChannelDim, OutputChannelDim);
                int kBeginL2 = MixPtr[j1 * numKernels + 1];
                int kEndL2 = MixPtr[(j1+1) * numKernels + 1];

#ifdef AVX512
                PerfectSpatialLocalitySpMMAVX512OutDim32(Ap, Ai, Ax, OutputChannelDim, Output, L2Ptr, residueStart,
                                           intermediateResult, kBeginL2, kEndL2);
#else
                perfectSpatialLocalitySpMM(Ap, Ai, Ax, OutputChannelDim, Output, L2Ptr, residueStart,
                                           intermediateResult, kBeginL2, kEndL2);
#endif
            }
        }
    }
    delete[] intermediateResult;
}

inline void
perfectSpatialLocalitySpMM(const int *Ap, const int *Ai, const float *Ax, const int OutputChannelDim, float *Output,
                            const int *L2Ptr, int residueStart, const float *intermediateResult, int kBeginL2,
                            int kEndL2) {
    for (int k1 = kBeginL2; k1 < kEndL2; ++k1) {
        int i = L2Ptr[k1];
        int ip = i * OutputChannelDim;
        int row = i;
        int k = Ap[row];
        for (; k < Ap[row + 1]-1; k+=2) {
            auto bxV1 = _mm256_set1_ps(Ax[k]);
            auto bxV2 = _mm256_set1_ps(Ax[k + 1]);
            int bij1 = Ai[k] * OutputChannelDim;
            int bij2 = Ai[k + 1] * OutputChannelDim;
            for (int kk = 0; kk < residueStart; kk += 16) {
                auto dxV1 = _mm256_loadu_ps(Output + ip + kk);
                auto dxV2 = _mm256_loadu_ps(Output + ip + kk + 8);
                auto acxV11 = _mm256_loadu_ps(intermediateResult + bij1 + kk);
                auto acxV12 = _mm256_loadu_ps(intermediateResult + bij1 + kk + 8);
                auto acxV21 = _mm256_loadu_ps(intermediateResult + bij2 + kk);
                auto acxV22 = _mm256_loadu_ps(intermediateResult + bij2 + kk + 8);
                dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
                dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
                dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
                dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
                _mm256_storeu_ps(Output + ip + kk, dxV1);
                _mm256_storeu_ps(Output + ip + kk + 8, dxV2);
            }
            for (int kk = residueStart; kk < OutputChannelDim; kk++) {
                Output[ip + kk] +=
                        Ax[k] * intermediateResult[bij1 + kk];
                Output[ip + kk] +=
                        Ax[k+1] * intermediateResult[bij2 + kk];
            }
        }
        for (; k < Ap[row + 1]; k+=1) {
            auto bxV1 = _mm256_set1_ps(Ax[k]);
            int bij1 = Ai[k] * OutputChannelDim;
            for (int kk = 0; kk < residueStart; kk += 16) {
                auto dxV1 = _mm256_loadu_ps(Output + ip + kk);
                auto dxV2 = _mm256_loadu_ps(Output + ip + kk + 8);
                auto acxV11 = _mm256_loadu_ps(intermediateResult + bij1 + kk);
                auto acxV12 = _mm256_loadu_ps(intermediateResult + bij1 + kk + 8);
                dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
                dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
                _mm256_storeu_ps(Output + ip + kk, dxV1);
                _mm256_storeu_ps(Output + ip + kk + 8, dxV2);
            }
            for (int kk = residueStart; kk < OutputChannelDim; kk++) {
                Output[ip + kk] +=
                        Ax[k] * intermediateResult[bij1 + kk];
            }
        }
    }
}

void
registerReuseVectorizedSpMM(const int *Ap, const int *Ai, const float *Ax, const int OutputChannelDim, float *Output,
                            const int *L2Ptr, int residueStart, const float *intermediateResult, int kBeginL2,
                            int kEndL2) {
    for (int k1 = kBeginL2; k1 < kEndL2; ++k1) {
        int i = L2Ptr[k1];
        int row = i;
        for (int kk = 0; kk < residueStart; kk += 32) {
            int ip = i * OutputChannelDim;
            auto dxV1 = _mm256_loadu_ps(Output + ip + kk);
            auto dxV2 = _mm256_loadu_ps(Output + ip + kk + 8);
            auto dxV3 = _mm256_loadu_ps(Output + ip + kk + 16);
            auto dxV4 = _mm256_loadu_ps(Output + ip + kk + 24);
            int k = Ap[row];
            for (; k < Ap[row + 1]-1; k+=2) {
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
            for (; k < Ap[row + 1]; ++k) {
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
        for (int k = Ap[row]; k < Ap[row + 1]; k++) {
            int ip = OutputChannelDim * i;
            for (int kk = residueStart; kk < OutputChannelDim; kk++) {
                Output[ip + kk] +=
                        Ax[k] * intermediateResult[Ai[k] * OutputChannelDim + kk];
            }
        }
    }
}

#ifdef AVX512

inline void
PerfectSpatialLocalitySpMMAVX512OutDim32(const int *Ap, const int *Ai, const float *Ax, const int OutputChannelDim, float *Output,
                           const int *L2Ptr, const float *intermediateResult, int kBeginL2,
                           int kEndL2) {
    int residueStart = OutputChannelDim % 32;
    for (int k1 = kBeginL2; k1 < kEndL2; ++k1) {
        int i = L2Ptr[k1];
        int ip = i * OutputChannelDim;
        int row = i;
        int k = Ap[row];

        for (; k < Ap[row + 1]-1; k+=2) {
            auto bxV1 = _mm512_set1_ps(Ax[k]);
            auto bxV2 = _mm512_set1_ps(Ax[k + 1]);
            int bij1 = Ai[k] * OutputChannelDim;
            int bij2 = Ai[k + 1] * OutputChannelDim;
            for (int kk = 0; kk < residueStart; kk += 32) {
                auto dxV1 = _mm512_loadu_ps(Output + ip + kk);
                auto dxV2 = _mm512_loadu_ps(Output + ip + kk + 16);
                auto acxV11 = _mm512_loadu_ps(intermediateResult + bij1 + kk);
                auto acxV12 = _mm512_loadu_ps(intermediateResult + bij1 + kk + 16);
                auto acxV21 = _mm512_loadu_ps(intermediateResult + bij2 + kk);
                auto acxV22 = _mm512_loadu_ps(intermediateResult + bij2 + kk + 16);
                dxV1 = _mm512_fmadd_ps(bxV1, acxV11, dxV1);
                dxV1 = _mm512_fmadd_ps(bxV2, acxV21, dxV1);
                dxV2 = _mm512_fmadd_ps(bxV1, acxV12, dxV2);
                dxV2 = _mm512_fmadd_ps(bxV2, acxV22, dxV2);
                _mm512_storeu_ps(Output + ip + kk, dxV1);
                _mm512_storeu_ps(Output + ip + kk + 16, dxV2);
            }
            for (int kk = residueStart; kk < OutputChannelDim; kk++) {
                Output[ip + kk] +=
                        Ax[k] * intermediateResult[bij1 + kk];
                Output[ip + kk] +=
                        Ax[k+1] * intermediateResult[bij2 + kk];
            }
        }
        for (; k < Ap[row + 1]; k+=1) {
            auto bxV1 = _mm512_set1_ps(Ax[k]);
            int bij1 = Ai[k] * OutputChannelDim;
            for (int kk = 0; kk < residueStart; kk += 32) {
                auto dxV1 = _mm512_loadu_ps(Output + ip + kk);
                auto dxV2 = _mm512_loadu_ps(Output + ip + kk + 16);
                auto acxV11 = _mm512_loadu_ps(intermediateResult + bij1 + kk);
                auto acxV12 = _mm512_loadu_ps(intermediateResult + bij1 + kk + 16);
                dxV1 = _mm512_fmadd_ps(bxV1, acxV11, dxV1);
                dxV2 = _mm512_fmadd_ps(bxV1, acxV12, dxV2);
                _mm512_storeu_ps(Output + ip + kk, dxV1);
                _mm512_storeu_ps(Output + ip + kk + 16, dxV2);
            }
            for (int kk = residueStart; kk < OutputChannelDim; kk++) {
                Output[ip + kk] +=
                        Ax[k] * intermediateResult[bij1 + kk];
            }
        }
    }
}

inline void
perfectSpatialLocalitySpMMAVX512OutDim64(const int *Ap, const int *Ai, const float *Ax, const int OutputChannelDim, float *Output,
                                         const int *L2Ptr, const float *intermediateResult, int kBeginL2,
                                         int kEndL2) {
    int residueStart = OutputChannelDim % 64;
    for (int k1 = kBeginL2; k1 < kEndL2; ++k1) {
        int i = L2Ptr[k1];
        int ip = i * OutputChannelDim;
        int row = i;
        int k = Ap[row];

        for (; k < Ap[row + 1]-1; k+=2) {
            auto bxV1 = _mm512_set1_ps(Ax[k]);
            auto bxV2 = _mm512_set1_ps(Ax[k + 1]);
            int bij1 = Ai[k] * OutputChannelDim;
            int bij2 = Ai[k + 1] * OutputChannelDim;
            for (int kk = 0; kk < residueStart; kk += 64) {
                auto dxV1 = _mm512_loadu_ps(Output + ip + kk);
                auto dxV2 = _mm512_loadu_ps(Output + ip + kk + 16);
                auto dxV3 = _mm512_loadu_ps(Output + ip + kk + 32);
                auto dxV4 = _mm512_loadu_ps(Output + ip + kk + 48);
                auto acxV11 = _mm512_loadu_ps(intermediateResult + bij1 + kk);
                auto acxV12 = _mm512_loadu_ps(intermediateResult + bij1 + kk + 16);
                auto acxV13 = _mm512_loadu_ps(intermediateResult + bij1 + kk + 32);
                auto acxV14 = _mm512_loadu_ps(intermediateResult + bij1 + kk + 48);
                auto acxV21 = _mm512_loadu_ps(intermediateResult + bij2 + kk);
                auto acxV22 = _mm512_loadu_ps(intermediateResult + bij2 + kk + 16);
                auto acxV23 = _mm512_loadu_ps(intermediateResult + bij2 + kk + 32);
                auto acxV24 = _mm512_loadu_ps(intermediateResult + bij2 + kk + 48);
                dxV1 = _mm512_fmadd_ps(bxV1, acxV11, dxV1);
                dxV1 = _mm512_fmadd_ps(bxV2, acxV21, dxV1);
                dxV2 = _mm512_fmadd_ps(bxV1, acxV12, dxV2);
                dxV2 = _mm512_fmadd_ps(bxV2, acxV22, dxV2);
                dxV3 = _mm512_fmadd_ps(bxV1, acxV13, dxV3);
                dxV3 = _mm512_fmadd_ps(bxV2, acxV23, dxV3);
                dxV4 = _mm512_fmadd_ps(bxV1, acxV14, dxV4);
                dxV4 = _mm512_fmadd_ps(bxV2, acxV24, dxV4);
                _mm512_storeu_ps(Output + ip + kk, dxV1);
                _mm512_storeu_ps(Output + ip + kk + 16, dxV2);
                _mm512_storeu_ps(Output + ip + kk + 32, dxV3);
                _mm512_storeu_ps(Output + ip + kk + 48, dxV4);
            }
            for (int kk = residueStart; kk < OutputChannelDim; kk++) {
                Output[ip + kk] +=
                        Ax[k] * intermediateResult[bij1 + kk];
                Output[ip + kk] +=
                        Ax[k+1] * intermediateResult[bij2 + kk];
            }
        }
        for (; k < Ap[row + 1]; k+=1) {
            auto bxV1 = _mm512_set1_ps(Ax[k]);
            int bij1 = Ai[k] * OutputChannelDim;
            for (int kk = 0; kk < residueStart; kk += 64) {
                auto dxV1 = _mm512_loadu_ps(Output + ip + kk);
                auto dxV2 = _mm512_loadu_ps(Output + ip + kk + 16);
                auto dxV3 = _mm512_loadu_ps(Output + ip + kk + 32);
                auto dxV4 = _mm512_loadu_ps(Output + ip + kk + 48);
                auto acxV11 = _mm512_loadu_ps(intermediateResult + bij1 + kk);
                auto acxV12 = _mm512_loadu_ps(intermediateResult + bij1 + kk + 16);
                auto acxV13 = _mm512_loadu_ps(intermediateResult + bij1 + kk + 32);
                auto acxV14 = _mm512_loadu_ps(intermediateResult + bij1 + kk + 48);
                dxV1 = _mm512_fmadd_ps(bxV1, acxV11, dxV1);
                dxV2 = _mm512_fmadd_ps(bxV1, acxV12, dxV2);
                dxV3 = _mm512_fmadd_ps(bxV1, acxV13, dxV3);
                dxV4 = _mm512_fmadd_ps(bxV1, acxV14, dxV4);
                _mm512_storeu_ps(Output + ip + kk, dxV1);
                _mm512_storeu_ps(Output + ip + kk + 16, dxV2);
                _mm512_storeu_ps(Output + ip + kk + 32, dxV3);
                _mm512_storeu_ps(Output + ip + kk + 48, dxV4);
            }
            for (int kk = residueStart; kk < OutputChannelDim; kk++) {
                Output[ip + kk] +=
                        Ax[k] * intermediateResult[bij1 + kk];
            }
        }
    }
}
#endif

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

void fusedMKLGeMMSpMMTransposedWeight(
        int M, int *Ap, int *Ai, float *Ax, int InputChannelDim,
        int OutputChannelDim, float *Features, float *Weight, float *Output,
        int NumThreads, int LevelNo, const int *LevelPtr, const int *MixPtr, const int *Partition) {
    int numKernels = 2;
    int residueStart = OutputChannelDim - OutputChannelDim%32;
    float *intermediateResult = new float[M * OutputChannelDim]{};
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
                        CblasRowMajor, CblasNoTrans, CblasTrans, tileSize, OutputChannelDim,
                        InputChannelDim, 1., Features + iL1 * InputChannelDim,
                        InputChannelDim, Weight, InputChannelDim, 0.,
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

void spMMTiled(int M, int *Ap, int *Ai, float *Ax, int N,
               float *B, float *Output,
               int NumThreads, const int *LevelPtr, const int *MixPtr) {

    int residueStart = N - N % 16;
#pragma omp parallel num_threads(NumThreads)
    {
#pragma omp for
        for (int i1 = 0; i1 < LevelPtr[1]; i1++) {
            int tBegin = MixPtr[i1 * 2];
            int tEnd = MixPtr[(i1 + 1) * 2];
            for (int i = tBegin; i < tEnd; i++) {
                int ip = i * N;
                int row = i;
//                std::cout << "1: " <<  k << std::endl;
                spmmKernel(Ap, Ai, Ax, N, B, Output + ip, residueStart, row);
            }
        }
    }
}


void calculateWeightGradAndInputGradFused(int M, int *Ap, int *Ai, float *Ax, int N, int Of, int Ow,
               float *B, float* F, float* W, float* OutF, float* OutW, int MaxTileSize,
               int NumThreads, const int *LevelPtr, const int *MixPtr) {
    float* outWTemp = new float[N * Ow * NumThreads]{};
    float* intermediate = new float[MaxTileSize * N * NumThreads]{};
    int residueStart = N - N % 16;
#pragma omp parallel num_threads(NumThreads)
    {
        int tid = omp_get_thread_num();
#pragma omp for
        for (int i1 = 0; i1 < LevelPtr[1]; i1++) {
            int tBegin = MixPtr[i1 * 2];
            int tEnd = MixPtr[(i1 + 1) * 2];
            float* interMediateCache = intermediate + tid * MaxTileSize * N;
            memset(interMediateCache, 0, sizeof (float) * MaxTileSize * N);
            int tileSize = tEnd - tBegin;
            for (int i = tBegin; i < tEnd; i++) {
                int ip = i * N;
                int row = i;
//                std::cout << "1: " <<  k << std::endl;
                spmmKernel(Ap, Ai, Ax, N, B, interMediateCache, residueStart, row);
            }
            cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize, Ow,
                    N, 1., interMediateCache,
                    N, W, Ow, 0.,
                    OutF + tBegin * N, Ow);
            float* outWCache = outWTemp + tid * Ow * N;
            memset(outWCache, 0, sizeof (float) * Ow * N);
            cblas_sgemm(
                    CblasRowMajor, CblasTrans, CblasNoTrans, N, Of,
                    tileSize, 1., interMediateCache,
                    N, F + tBegin * Of, Of, 1.,
                    outWCache, Of);
//            for (int i = 0; i < N * Ow; i++){
//#pragma omp atomic
//                OutW[i] += outWCache[i];
//            }
        }
    }
//    int rowPerThread = std::max(N / NumThreads, 1);
int prevT;
for (int t = NumThreads/2; t > 1; t = t/2){
#pragma omp parallel num_threads(NumThreads)
        {
            int tid = omp_get_thread_num();
            int tidCorr = tid+t;
            float* outWCache1 = outWTemp + tid * Ow * N;
            float* outWCache2 = outWTemp + tidCorr * Ow * N;
            mkl_somatadd(MKL_ROW_MAJOR, CblasNoTrans, CblasNoTrans, N, Ow, 1.,
                         outWCache1, Ow, 1., outWCache2, Ow, outWCache1, Ow);
            if (tidCorr == prevT - 2){
                int tidCorr2 = tidCorr + 1;
                float* outWCache3 = outWTemp + tidCorr2 * Ow * N;
                mkl_somatadd(MKL_ROW_MAJOR, CblasNoTrans, CblasNoTrans, N, Ow, 1.,
                             outWCache1, Ow, 1., outWCache3, Ow, outWCache1, Ow);
            }
            prevT = t;
//        for (int t = 0; t < NumThreads; t++){
//            float* outWCache = outWTemp + t * Ow * N;
//            for (int i = 0; i < N * Ow; i++){
//#pragma omp atomic
//                OutW[i] += outWCache[i];
//            }
//        }
//        for (int i = 0; i < NumThreads; i += 1) {
//            int rowStart = i * rowPerThread;
//            int rowEnd = std::min(rowStart + rowPerThread, N);
//            for (int r = rowStart; r < rowEnd; r++) {
//                for (int t = 0; t < NumThreads; t++) {
//                    int ip = r * Ow;
//                    float *wPart = outWTemp + t * N * Ow + ip;
//                    for (int j = 0; j < Ow; j++) {
//                        OutW[ip + j] += wPart[j];
//                    }
//                }
//            }
//        }

        }
    }
    delete[] outWTemp;
    delete[] intermediate;
}

void calculateWGradFused(int M, int *Ap, int *Ai, float *Ax, int N, int Ow,
                                              float *B, float* F, float* OutW, int MaxTileSize,
                                              int NumThreads, const int *LevelPtr, const int *MixPtr) {
    float* outWTemp = new float[N * Ow * NumThreads]{};
    float* intermediate = new float[MaxTileSize * N * NumThreads]{};
    int residueStart = N - N % 16;
#pragma omp parallel num_threads(NumThreads)
    {
        int tid = omp_get_thread_num();
#pragma omp for
        for (int i1 = 0; i1 < LevelPtr[1]; i1++) {
            int tBegin = MixPtr[i1 * 2];
            int tEnd = MixPtr[(i1 + 1) * 2];
            float* interMediateCache = intermediate + tid * MaxTileSize * N;
            memset(interMediateCache, 0, sizeof (float) * MaxTileSize * N);
            int tileSize = tEnd - tBegin;
            for (int i = tBegin; i < tEnd; i++) {
                int ip = i * N;
                int row = i;
//                std::cout << "1: " <<  k << std::endl;
                spmmKernel(Ap, Ai, Ax, N, B, interMediateCache, residueStart, row);
            }
            float* outWCache = outWTemp + tid * Ow * N;
            memset(outWCache, 0, sizeof (float) * Ow * N);
            cblas_sgemm(
                    CblasRowMajor, CblasTrans, CblasNoTrans, N, Ow,
                    tileSize, 1., interMediateCache,
                    N, F + tBegin * Ow, Ow, 1.,
                    outWCache, Ow);
//            for (int i = 0; i < N * Ow; i++){
//#pragma omp atomic
//                OutW[i] += outWCache[i];
//            }
        }
    }
//    int rowPerThread = std::max(N / NumThreads, 1);
    int prevT;
    for (int t = NumThreads/2; t > 1; t = t/2){
#pragma omp parallel num_threads(NumThreads)
        {
            int tid = omp_get_thread_num();
            int tidCorr = tid+t;
            float* outWCache1 = outWTemp + tid * Ow * N;
            float* outWCache2 = outWTemp + tidCorr * Ow * N;
            mkl_somatadd(MKL_ROW_MAJOR, CblasNoTrans, CblasNoTrans, N, Ow, 1.,
                         outWCache1, Ow, 1., outWCache2, Ow, outWCache1, Ow);
            if (tidCorr == prevT - 2){
                int tidCorr2 = tidCorr + 1;
                float* outWCache3 = outWTemp + tidCorr2 * Ow * N;
                mkl_somatadd(MKL_ROW_MAJOR, CblasNoTrans, CblasNoTrans, N, Ow, 1.,
                             outWCache1, Ow, 1., outWCache3, Ow, outWCache1, Ow);
            }
            prevT = t;
//        for (int t = 0; t < NumThreads; t++){
//            float* outWCache = outWTemp + t * Ow * N;
//            for (int i = 0; i < N * Ow; i++){
//#pragma omp atomic
//                OutW[i] += outWCache[i];
//            }
//        }
//        for (int i = 0; i < NumThreads; i += 1) {
//            int rowStart = i * rowPerThread;
//            int rowEnd = std::min(rowStart + rowPerThread, N);
//            for (int r = rowStart; r < rowEnd; r++) {
//                for (int t = 0; t < NumThreads; t++) {
//                    int ip = r * Ow;
//                    float *wPart = outWTemp + t * N * Ow + ip;
//                    for (int j = 0; j < Ow; j++) {
//                        OutW[ip + j] += wPart[j];
//                    }
//                }
//            }
//        }

        }
    }
    delete[] outWTemp;
    delete[] intermediate;
}

inline void spmmKernel(const int *Ap, const int *Ai, const float *Ax, int N, const float *B, float *Output, int residueStart, int row) {
    int k = Ap[row];
    for (; k < Ap[row + 1] - 1; k += 2) {
        auto bxV1 = _mm256_set1_ps(Ax[k]);
        auto bxV2 = _mm256_set1_ps(Ax[k + 1]);
        int bij1 = Ai[k] * N;
        int bij2 = Ai[k + 1] * N;
        for (int kk = 0; kk < residueStart; kk += 16) {
            auto dxV1 = _mm256_loadu_ps(Output + kk);
            auto dxV2 = _mm256_loadu_ps(Output + kk + 8);
            auto acxV11 = _mm256_loadu_ps(B + bij1 + kk);
            auto acxV12 = _mm256_loadu_ps(B + bij1 + kk + 8);
            auto acxV21 = _mm256_loadu_ps(B + bij2 + kk);
            auto acxV22 = _mm256_loadu_ps(B + bij2 + kk + 8);
            dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
            dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
            dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
            dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
            _mm256_storeu_ps(Output + kk, dxV1);
            _mm256_storeu_ps(Output + kk + 8, dxV2);
        }
        for (int kk = residueStart; kk < N; kk++) {
            Output[kk] +=
                    Ax[k] * B[bij1 + kk];
            Output[kk] +=
                    Ax[k + 1] * B[bij2 + kk];
        }
    }
//                std::cout << "2: " <<  k << std::endl;
    for (; k < Ap[row + 1]; k += 1) {
        auto bxV1 = _mm256_set1_ps(Ax[k]);
        int bij1 = Ai[k] * N;
        for (int kk = 0; kk < residueStart; kk += 16) {
            auto dxV1 = _mm256_loadu_ps(Output + kk);
            auto dxV2 = _mm256_loadu_ps(Output + kk + 8);
            auto acxV11 = _mm256_loadu_ps(B + bij1 + kk);
            auto acxV12 = _mm256_loadu_ps(B + bij1 + kk + 8);
            dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
            dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
            _mm256_storeu_ps(Output + kk, dxV1);
            _mm256_storeu_ps(Output + kk + 8, dxV2);
        }
        for (int kk = residueStart; kk < N; kk++) {
            Output[kk] +=
                    Ax[k] * B[bij1 + kk];
        }
    }
}

torch::Tensor FusedGeMMSpMMROAdj::forward(torch::autograd::AutogradContext *Ctx,
                                     torch::Tensor Adj, torch::Tensor Feature, torch::Tensor Weight,
                                     torch::Tensor LevelPtr, torch::Tensor MixPtr, torch::Tensor L2Ptr,
                                     int64_t NumThreads, int64_t MaxTileSize) {
    float *out = new float[Adj.size(0) * Weight.size(0)]{};
//    mkl_set_num_threads(1);
    fusedGeMMSpMMReorderedAdjVectorizedTransposedWeight(Adj.size(0),
                                     Adj.crow_indices().data_ptr<int32_t>(),
                                     Adj.col_indices().data_ptr<int32_t>(),
                                     Adj.values().data_ptr<float>(),
                                     Feature.size(1),
                                     Weight.size(0), Feature.data_ptr<float>(),
                                     Weight.data_ptr<float>(), out, NumThreads, 2, LevelPtr.data_ptr<int32_t>(),
                                     MixPtr.data_ptr<int32_t>(),
                                     L2Ptr.data_ptr<int32_t>());
    Ctx->save_for_backward({Adj, Feature, Weight, LevelPtr, MixPtr});
    Ctx->saved_data["num_threads"] = NumThreads;
    Ctx->saved_data["max_tile_size"] = MaxTileSize;
    return torch::from_blob(
            out, {Adj.size(0), Weight.size(0)},
            [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
}

//
//torch::autograd::tensor_list
//FusedGeMMSpMMROAdj::backward(torch::autograd::AutogradContext *Ctx,
//                        torch::autograd::tensor_list GradOutputs) {
//    matrix_descr d;
//    d.type = SPARSE_MATRIX_TYPE_GENERAL;
//    auto saved = Ctx->get_saved_variables();
//    auto input = saved[1];
//    auto adj = saved[0];
//    auto weight = saved[2];
//    auto *levelPtr = saved[3].data_ptr<int>();
//    auto *mixPtr = saved[4].data_ptr<int>();
//    int threadNum = Ctx->saved_data["num_threads"].toInt();
//    int *adjPtr = adj.crow_indices().data_ptr<int>();
//    int *adjIndex = adj.col_indices().data_ptr<int>();
//    auto grad_output = GradOutputs[0];
//    float *grad_output_raw = grad_output.data_ptr<float>();
//    float *inputRaw = input.data_ptr<float>();
//
////    mkl_sparse_s_create_csr(&MKLAdj, SPARSE_INDEX_BASE_ZERO, adj.size(0),
////                            adj.size(1), adjPtr,
////                            adjPtr + 1,
////                            adjIndex,
////                            adj.values().data_ptr<float>());
//    float *adjTGradRes = new float[adj.size(0) * grad_output.size(1)]{};
//    spMMTiled(adj.size(0), adjPtr, adjIndex, adj.values().data_ptr<float>(),
//              grad_output.size(1), grad_output_raw, adjTGradRes,
//              threadNum, levelPtr, mixPtr);
//    torch::Tensor grad_weight;
//    if (Ctx->needs_input_grad(2)){
//        float *grad_weight_raw = new float[grad_output.size(1) * input.size(1)]{};
//        //      swiftware::benchmark::Timer t1;
//        //      t1.start();
////        mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, MKLAdj, d,
////                        SPARSE_LAYOUT_ROW_MAJOR, inputRaw,
////                        input.size(1), input.size(1), 0,
////                        grad_intermediate, input.size(1));
////        mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, MKLAdj, d,
////                        SPARSE_LAYOUT_ROW_MAJOR, grad_output_raw,
////                        grad_output.size(1), grad_output.size(1), 0,
////                        adjTGradRes, grad_output.size(1));
//        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, grad_output.size(1),
//                    weight.size(1), adj.size(0), 1., adjTGradRes,
//                    grad_output.size(1), inputRaw, input.size(1), 0.,
//                    grad_weight_raw, input.size(1));
//        //      t1.stop();
//        //      std::cout <<  "GeMMSpMM_BWW_TiledFused" << "," << "mat_name" << "," << t1.printTimeCsv(0) << std::endl;
////        mkl_free(MKLAdj);
//        grad_weight = torch::from_blob(
//                grad_weight_raw, {grad_output.size(1), input.size(1)},
//                [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
//    }
//    torch::Tensor grad_input;
//    if (Ctx->needs_input_grad(1)) {
//        float *weight_raw = weight.data_ptr<float>();
//        float *grad_input_raw = new float[adj.size(0) * weight.size(1)]{};
//
//        //      swiftware::benchmark::Timer t1;
//        //      t1.start();
//        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, adj.size(0),
//                    weight.size(1), grad_output.size(1), 1., adjTGradRes,
//                    grad_output.size(1), weight_raw, weight.size(1), 0.,
//                    grad_input_raw, weight.size(1));
//        //      t1.stop();
//        //      std::cout <<  "GeMMSpMM_BWI_TiledFused" << "," << "mat_name" << "," << t1.printTimeCsv(0) << std::endl;
//        grad_input = torch::from_blob(
//                grad_input_raw, {grad_output.size(0), weight.size(1)},
//                [](void *ptr) { delete[] static_cast<float *>(ptr); },
//                torch::kFloat32);
//    }
//    delete[] adjTGradRes;
//    at::Tensor undef;
//    return {undef, grad_input, grad_weight, undef, undef, undef, undef, undef};
//}

torch::autograd::tensor_list
FusedGeMMSpMMROAdj::backward(torch::autograd::AutogradContext *Ctx,
                             torch::autograd::tensor_list GradOutputs) {
    matrix_descr d;
    d.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto saved = Ctx->get_saved_variables();
    auto input = saved[1];
    auto adj = saved[0];
    auto weight = saved[2];
    auto *levelPtr = saved[3].data_ptr<int>();
    auto *mixPtr = saved[4].data_ptr<int>();
    int numThreads = Ctx->saved_data["num_threads"].toInt();
    int maxTileSize = Ctx->saved_data["max_tile_size"].toInt();
    int *adjPtr = adj.crow_indices().data_ptr<int>();
    int *adjIndex = adj.col_indices().data_ptr<int>();
    auto grad_output = GradOutputs[0];
    float *grad_output_raw = grad_output.data_ptr<float>();
    float *inputRaw = input.data_ptr<float>();
    float *weightRaw = input.data_ptr<float>();
//    mkl_sparse_s_create_csr(&MKLAdj, SPARSE_INDEX_BASE_ZERO, adj.size(0),
//                            adj.size(1), adjPtr,
//                            adjPtr + 1,
//                            adjIndex,
//                            adj.values().data_ptr<float>());
    float *grad_weight_raw = new float[grad_output.size(1) * input.size(1)]{};
    torch::Tensor grad_input;
    if (Ctx->needs_input_grad(1)) {
    float *grad_input_raw = new float[adj.size(0) * weight.size(1)]{};
        calculateWeightGradAndInputGradFused(adj.size(0), adjPtr, adjIndex, adj.values().data_ptr<float>(),
                     grad_output.size(1), input.size(1), weight.size(1), grad_output_raw, inputRaw, weightRaw,
                     grad_input_raw, grad_weight_raw, maxTileSize,
                     numThreads, levelPtr, mixPtr);

        grad_input = torch::from_blob(
                grad_input_raw, {grad_output.size(0), weight.size(1)},
                [](void *ptr) { delete[] static_cast<float *>(ptr); },
                torch::kFloat32);
    }
    else{
        calculateWGradFused(adj.size(0), adjPtr, adjIndex, adj.values().data_ptr<float>(),
                                             grad_output.size(1), weight.size(1),
                                             grad_output_raw, inputRaw, grad_weight_raw, maxTileSize,
                                             numThreads, levelPtr, mixPtr);
    }
    torch::Tensor grad_weight;
        grad_weight = torch::from_blob(
                grad_weight_raw, {grad_output.size(1), input.size(1)},
                [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
//    }

    at::Tensor undef;
    return {undef, grad_input, grad_weight, undef, undef, undef, undef, undef};
}



torch::Tensor FusedGeMMSpMM::forward(torch::autograd::AutogradContext *Ctx,
                                     torch::Tensor Adj, torch::Tensor Feature, torch::Tensor Weight,
                                     torch::Tensor LevelPtr, torch::Tensor MixPtr, torch::Tensor Partition,
                                     int64_t NumThreads) {
    float *out = new float[Adj.size(0) * Weight.size(0)]{};
    mkl_set_num_threads(1);
    fusedMKLGeMMSpMMTransposedWeight(Adj.size(0),
                                     Adj.crow_indices().data_ptr<int32_t>(),
                                     Adj.col_indices().data_ptr<int32_t>(),
                                     Adj.values().data_ptr<float>(),
                                     Feature.size(1),
                                     Weight.size(0), Feature.data_ptr<float>(),
                                     Weight.data_ptr<float>(), out, NumThreads, 2, LevelPtr.data_ptr<int32_t>(),
                                     MixPtr.data_ptr<int32_t>(),
                                     Partition.data_ptr<int32_t>());
    Ctx->save_for_backward({Adj, Feature, Weight, LevelPtr, MixPtr, Partition});
    Ctx->saved_data["num_threads"] = NumThreads;
    return torch::from_blob(
            out, {Adj.size(0), Weight.size(0)},
            [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
}

torch::autograd::tensor_list
FusedGeMMSpMM::backward(torch::autograd::AutogradContext *Ctx,
         torch::autograd::tensor_list GradOutputs) {
    sparse_matrix_t MKLAdj;
    matrix_descr d;
    d.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto saved = Ctx->get_saved_variables();
    auto input = saved[1];
    auto adj = saved[0];
    auto weight = saved[2];
    int threadNum = Ctx->saved_data["num_threads"].toInt();
    int* levelPtr = saved[3].data_ptr<int>();
    int* mixPtr = saved[4].data_ptr<int>();
    int* partition = saved[5].data_ptr<int>();
    int *adjPtr = adj.crow_indices().data_ptr<int>();
    int *adjIndex = adj.col_indices().data_ptr<int>();
    auto grad_output = GradOutputs[0];
    float *grad_output_raw = grad_output.data_ptr<float>();
    float *inputRaw = input.data_ptr<float>();
    torch::Tensor grad_input;
    if (Ctx->needs_input_grad(1)) {
        mkl_set_num_threads(1);
        float *weight_raw = weight.data_ptr<float>();
        float *grad_input_raw = new float[adj.size(0) * weight.size(1)]{};
        //      swiftware::benchmark::Timer t1;
        //      t1.start();
        fusedMKLGeMMSpMM(
                grad_output.size(0), adjPtr, adjIndex, adj.values().data_ptr<float>(),
                grad_output.size(1), weight.size(1), grad_output_raw,
                weight_raw, grad_input_raw, threadNum,2, levelPtr,
                mixPtr, partition);
        //      t1.stop();
        //      std::cout <<  "GeMMSpMM_BWI_TiledFused" << "," << "mat_name" << "," << t1.printTimeCsv(0) << std::endl;
        grad_input = torch::from_blob(
                grad_input_raw, {(long)grad_output.size(0), (long)weight.size(1)},
                [](void *ptr) { delete[] static_cast<float *>(ptr); },
                torch::kFloat32);
    }
    torch::Tensor grad_weight;
    if (Ctx->needs_input_grad(2)){
        mkl_set_num_threads(threadNum);
        mkl_sparse_s_create_csr(&MKLAdj, SPARSE_INDEX_BASE_ZERO, adj.size(0),
                                adj.size(1), adjPtr,
                                adjPtr + 1,
                                adjIndex,
                                adj.values().data_ptr<float>());
        float *grad_intermediate = new float[adj.size(0) * input.size(1)]{};
        float *grad_weight_raw = new float[grad_output.size(1) * input.size(1)]{};
        //      swiftware::benchmark::Timer t1;
        //      t1.start();
        mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, MKLAdj, d,
                        SPARSE_LAYOUT_ROW_MAJOR, inputRaw,
                        input.size(1), input.size(1), 0,
                        grad_intermediate, input.size(1));
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, grad_output.size(1),
                    input.size(1), adj.size(0), 1., grad_output_raw,
                    grad_output.size(1), grad_intermediate, input.size(1), 0.,
                    grad_weight_raw, input.size(1));
        //      t1.stop();
        //      std::cout <<  "GeMMSpMM_BWW_TiledFused" << "," << "mat_name" << "," << t1.printTimeCsv(0) << std::endl;
        mkl_free(MKLAdj);
        delete[] grad_intermediate;
        grad_weight = torch::from_blob(
                grad_weight_raw, {grad_output.size(1), input.size(1)},
                [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
    }
    at::Tensor undef;
    return {undef, grad_input, grad_weight, undef, undef, undef, undef};
}

torch::Tensor SGForwardFusedGSBackward::forward(torch::autograd::AutogradContext *Ctx,
                                                torch::Tensor Adj, torch::Tensor Feature, torch::Tensor Weight,
                                                torch::Tensor LevelPtr, torch::Tensor MixPtr, torch::Tensor L2Ptr,
                                                int64_t NumThreads) {
    mkl_set_num_threads(NumThreads);
    sparse_matrix_t MKLAdj;
    matrix_descr d;
    d.type = SPARSE_MATRIX_TYPE_GENERAL;
    int *adjPtr = Adj.crow_indices().data_ptr<int>();
    int *adjIndex = Adj.col_indices().data_ptr<int>();
    float* featRaw = Feature.data_ptr<float>();
    float* weightRaw = Weight.data_ptr<float>();
    mkl_sparse_s_create_csr(&MKLAdj, SPARSE_INDEX_BASE_ZERO, Adj.size(0),
                            Adj.size(1), adjPtr,
                            adjPtr + 1,
                            adjIndex,
                            Adj.values().data_ptr<float>());
    float *outIntermediate = new float[Adj.size(0) * Feature.size(1)]{};
    float *out = new float[Adj.size(0) * Weight.size(0)]{};
    mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, MKLAdj, d,
                    SPARSE_LAYOUT_ROW_MAJOR, featRaw,
                    Feature.size(1), Feature.size(1), 0,
                    outIntermediate, Feature.size(1));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Adj.size(0),
                Weight.size(0), Feature.size(1), 1., outIntermediate,
                Feature.size(1), weightRaw, Weight.size(1), 0.,
                out, Weight.size(0));
    auto intermediateTensor = torch::from_blob(
            outIntermediate, {Adj.size(0), Feature.size(1)},
            [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
    mkl_free(MKLAdj);
    Ctx->save_for_backward({intermediateTensor, Adj, Weight, LevelPtr, MixPtr, L2Ptr});
    Ctx->saved_data["num_threads"] = NumThreads;
    return torch::from_blob(
            out, {Adj.size(0), Weight.size(0)},
            [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
}

torch::autograd::tensor_list
SGForwardFusedGSBackward::backward(torch::autograd::AutogradContext *Ctx,
                                   torch::autograd::tensor_list GradOutputs) {
    sparse_matrix_t MKLAdj;
    matrix_descr d;
    d.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto saved = Ctx->get_saved_variables();
    auto intermediateRes = saved[0];
    auto weight = saved[2];
    int threadNum = Ctx->saved_data["num_threads"].toInt();
    int* levelPtr = saved[3].data_ptr<int>();
    int* mixPtr = saved[4].data_ptr<int>();
    int* partition = saved[5].data_ptr<int>();
    auto roAdj = saved[1];
//    int *adjPtr = adj.crow_indices().data_ptr<int>();
//    int *adjIndex = adj.col_indices().data_ptr<int>();
    int *roAdjPtr = roAdj.crow_indices().data_ptr<int>();
    int *roAdjIndex = roAdj.col_indices().data_ptr<int>();
    auto grad_output = GradOutputs[0];
    float *grad_output_raw = grad_output.data_ptr<float>();
    torch::Tensor grad_input;
    if (Ctx->needs_input_grad(1)) {
//        mkl_set_num_threads(1);
        float *weight_raw = weight.data_ptr<float>();
        float *grad_input_raw = new float[roAdj.size(0) * weight.size(1)]{};
        //      swiftware::benchmark::Timer t1;
        //      t1.start();
        fusedGeMMSpMMReorderedAdjVectorized(
                grad_output.size(0), roAdjPtr, roAdjIndex, roAdj.values().data_ptr<float>(),
                grad_output.size(1), weight.size(1), grad_output_raw,
                weight_raw, grad_input_raw, threadNum,2, levelPtr,
                mixPtr, partition);
        //      t1.stop();
        //      std::cout <<  "GeMMSpMM_BWI_TiledFused" << "," << "mat_name" << "," << t1.printTimeCsv(0) << std::endl;
        grad_input = torch::from_blob(
                grad_input_raw, {(long)grad_output.size(0), (long)weight.size(1)},
                [](void *ptr) { delete[] static_cast<float *>(ptr); },
                torch::kFloat32);
    }
    torch::Tensor grad_weight;
    if (Ctx->needs_input_grad(2)){
        float *grad_weight_raw = new float[grad_output.size(1) * intermediateRes.size(1)]{};
//        mkl_set_num_threads(threadNum);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, grad_output.size(1),
                    intermediateRes.size(1), intermediateRes.size(0), 1., grad_output_raw,
                    grad_output.size(1), intermediateRes.data_ptr<float>(), intermediateRes.size(1), 0.,
                    grad_weight_raw, intermediateRes.size(1));
        //      t1.stop();
        //      std::cout <<  "GeMMSpMM_BWW_TiledFused" << "," << "mat_name" << "," << t1.printTimeCsv(0) << std::endl;
//        mkl_free(MKLAdj);
        grad_weight = torch::from_blob(
                grad_weight_raw, {grad_output.size(1), intermediateRes.size(1)},
                [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
    }
    at::Tensor undef;
    return {undef, grad_input, grad_weight, undef, undef, undef, undef};
}


torch::Tensor ForwardCachingAF::forward(torch::autograd::AutogradContext *Ctx,
                                                torch::Tensor AF,torch::Tensor Weight,
                                                int64_t NumThreads) {
//    mkl_set_num_threads(NumThreads);
    float* weightRaw = Weight.data_ptr<float>();
    float* afRaw = AF.data_ptr<float>();
    float *out = new float[AF.size(0) * Weight.size(0)]{};
//    mkl_set_num_threads(NumThreads);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, AF.size(0),
                Weight.size(0), AF.size(1), 1., afRaw,
                AF.size(1), weightRaw, Weight.size(1), 0.,
                out, Weight.size(0));
    Ctx->save_for_backward({AF});
    Ctx->saved_data["num_threads"] = NumThreads;
    return torch::from_blob(
            out, {AF.size(0), Weight.size(0)},
            [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
}

torch::autograd::tensor_list
ForwardCachingAF::backward(torch::autograd::AutogradContext *Ctx,
                                   torch::autograd::tensor_list GradOutputs) {
    auto saved = Ctx->get_saved_variables();
    int threadNum = Ctx->saved_data["num_threads"].toInt();
    auto afRes = saved[0];
    auto grad_output = GradOutputs[0];
    float *grad_output_raw = grad_output.data_ptr<float>();
    float *af_raw = afRes.data_ptr<float>();
    torch::Tensor grad_weight;
    if (Ctx->needs_input_grad(1)){
        float *grad_weight_raw = new float[grad_output.size(1) * afRes.size(1)]{};
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, grad_output.size(1),
                    afRes.size(1), afRes.size(0), 1., grad_output_raw,
                    grad_output.size(1), af_raw, afRes.size(1), 0.,
                    grad_weight_raw, afRes.size(1));
        //      t1.stop();
        //      std::cout <<  "GeMMSpMM_BWW_TiledFused" << "," << "mat_name" << "," << t1.printTimeCsv(0) << std::endl;
        grad_weight = torch::from_blob(
                grad_weight_raw, {grad_output.size(1), afRes.size(1)},
                [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32);
    }
    at::Tensor undef;
    return {undef, grad_weight, undef};
}
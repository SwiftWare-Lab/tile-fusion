//
// Created by salehm32 on 2026-05-08.
//

#include "Functions.h"

#include <algorithm>
#include <cmath>
#include <set>


int**
swiftware::fusion::generateVariableTileSizeScheduleGeMMSpMM(int M, int* Ap, int* Ai, int BCol, int CCol, int CacheSize, int NumThreads,int DataSize){
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

int swiftware::fusion::calculateWorkingSetSizeForSpMMSpMM(
    int Nnz, int UniqueColNnz, int FeatureDim, int TileSize,
    int FusedTileSize, int DataSize) {
  return (TileSize * FeatureDim + UniqueColNnz * FeatureDim +
          FusedTileSize * FeatureDim) *
             DataSize +
         Nnz * static_cast<int>(sizeof(int));
}

int** swiftware::fusion::generateVariableTileSizeScheduleSpMMSpMM(
    int M, int* Ap, int* Ai, int FeatureDim, int CacheSize, int NumThreads,
    int DataSize) {
  struct VariableTile {
    int start;
    int end;
    std::vector<int> fusedIters;
    VariableTile* next;
    VariableTile(int start_, int end_) : start(start_), end(end_), next(nullptr) {}
  };

  int minMaxTileSize = std::max(1, M / std::max(1, 2 * NumThreads));
  int initialTileSize = std::min(4096, minMaxTileSize);
  int extraIters = M % initialTileSize;
  int extraRemoved = 0;
  int numOfTiles = M / initialTileSize;
  int extraIterPerTile = numOfTiles == 0 ? 0 : static_cast<int>(std::ceil(extraIters / static_cast<double>(numOfTiles)));
  std::vector<int> unfusedIters;

  VariableTile* head = new VariableTile(0, 0);
  VariableTile* curr = head;
  for (int i = 0; i < numOfTiles; i++) {
    int start = initialTileSize * i + extraRemoved;
    int end = start + initialTileSize;
    if (extraIters > extraRemoved) {
      int ext = std::min(extraIters - extraRemoved, extraIterPerTile);
      end += ext;
      extraRemoved += ext;
    }
    curr->next = new VariableTile(start, end);
    curr = curr->next;
  }

  curr = head;
  while (curr->next != nullptr) {
    curr = curr->next;
    for (int i = curr->start; i < curr->end; i++) {
      if (Ap[i] < Ap[i + 1] && Ai[Ap[i]] >= curr->start &&
          Ai[Ap[i + 1] - 1] < curr->end) {
        curr->fusedIters.push_back(i);
      } else {
        unfusedIters.push_back(i);
      }
    }
  }

  curr = head;
  while (curr->next != nullptr) {
    VariableTile* prev = curr;
    curr = curr->next;
    int tileSize = curr->end - curr->start;
    int* firstColPtr = Ai + Ap[curr->start];
    int* lastColPtr = Ai + Ap[curr->end];
    int nnzNum = Ap[curr->end] - Ap[curr->start];
    std::set<int> uniqueColumns(firstColPtr, lastColPtr);
    int workingSet = calculateWorkingSetSizeForSpMMSpMM(
        nnzNum, uniqueColumns.size(), FeatureDim, tileSize,
        curr->fusedIters.size(), DataSize);
    if (workingSet > CacheSize && tileSize > 1) {
      int separator = tileSize / 2 + curr->start;
      VariableTile* vt1 = new VariableTile(curr->start, separator);
      VariableTile* vt2 = new VariableTile(separator, curr->end);
      for (int fi : curr->fusedIters) {
        if (Ap[fi] < Ap[fi + 1] && Ai[Ap[fi + 1] - 1] < separator) {
          vt1->fusedIters.push_back(fi);
        } else if (Ap[fi] < Ap[fi + 1] && Ai[Ap[fi]] >= separator) {
          vt2->fusedIters.push_back(fi);
        } else {
          unfusedIters.push_back(fi);
        }
      }
      vt1->next = vt2;
      vt2->next = curr->next;
      prev->next = vt1;
      numOfTiles += 1;
      delete curr;
      curr = prev;
    }
  }

  std::sort(unfusedIters.begin(), unfusedIters.end());
  std::vector<int> ufPartPtr;
  int minStride = 16;
  std::set<int> uniqueColumns;
  int nnzNum = 0;
  int uft = 0;
  int ufTileSize = 0;
  ufPartPtr.push_back(0);
  while (uft < static_cast<int>(unfusedIters.size())) {
    int strideEnd = std::min(static_cast<int>(unfusedIters.size()), uft + minStride);
    for (int ii = uft; ii < strideEnd; ii++) {
      int row = unfusedIters[ii];
      uniqueColumns.insert(Ai + Ap[row], Ai + Ap[row + 1]);
      nnzNum += Ap[row + 1] - Ap[row];
      ufTileSize += 1;
    }
    int workingSet = calculateWorkingSetSizeForSpMMSpMM(
        nnzNum, uniqueColumns.size(), FeatureDim, ufTileSize, 0, DataSize);
    if (((workingSet < CacheSize) || (ufTileSize == 1)) &&
        (ufTileSize + minStride < minMaxTileSize)) {
      uft += minStride;
    } else {
      ufPartPtr.push_back(uft);
      nnzNum = 0;
      uniqueColumns.clear();
      if (ufTileSize <= minStride) {
        minStride = std::max(1, minStride / 2);
      }
      if (ufTileSize >= 3 * minStride) {
        minStride *= 2;
      }
      ufTileSize = 0;
    }
  }
  ufPartPtr.push_back(unfusedIters.size());
  int numUfTiles = static_cast<int>(ufPartPtr.size()) - 1;

  int* levelPtr = new int[3];
  levelPtr[0] = 0;
  levelPtr[1] = numOfTiles;
  levelPtr[2] = numOfTiles + numUfTiles;
  int* parPtr = new int[numOfTiles + numUfTiles + 1];
  int* partition = new int[2 * M];
  int* parType = new int[2 * M];
  parPtr[0] = 0;

  int cnt = 0;
  int pCounter = 0;
  curr = head;
  while (curr->next != nullptr) {
    curr = curr->next;
    for (int j = curr->start; j < curr->end; j++) {
      partition[cnt] = j;
      parType[cnt] = 0;
      cnt++;
    }
    for (int fi : curr->fusedIters) {
      partition[cnt] = fi;
      parType[cnt] = 1;
      cnt++;
    }
    parPtr[pCounter + 1] = cnt;
    pCounter += 1;
  }

  curr = head->next;
  while (curr != nullptr) {
    VariableTile* tmp = curr;
    curr = curr->next;
    delete tmp;
  }
  delete head;

  for (int i = numOfTiles; i < numOfTiles + numUfTiles; i++) {
    int p = i - numOfTiles;
    int partEnd = ufPartPtr[p + 1];
    for (int j = ufPartPtr[p]; j < partEnd; j++) {
      partition[cnt] = unfusedIters[j];
      parType[cnt] = 1;
      cnt++;
    }
    parPtr[i + 1] = cnt;
  }

  int** out = new int*[4];
  out[0] = levelPtr;
  out[1] = parPtr;
  out[2] = partition;
  out[3] = parType;
  return out;
}

int** swiftware::fusion::createSpMMSpMMFixedSchedule(int M, int MTileSize) {
  int safeTileSize = std::max(1, MTileSize);
  int numTiles = CEIL(M, safeTileSize);
  int* levelPtr = new int[3];
  int* parPtr = new int[2 * numTiles + 1];
  int* partition = new int[2 * M];
  int* parType = new int[2 * M];

  levelPtr[0] = 0;
  levelPtr[1] = numTiles;
  levelPtr[2] = 2 * numTiles;
  parPtr[0] = 0;
  int cnt = 0;
  for (int tile = 0; tile < numTiles; tile++) {
    int start = tile * safeTileSize;
    int end = std::min(M, start + safeTileSize);
    for (int row = start; row < end; row++) {
      partition[cnt] = row;
      parType[cnt] = 0;
      cnt++;
    }
    parPtr[tile + 1] = cnt;
  }
  for (int tile = 0; tile < numTiles; tile++) {
    int start = tile * safeTileSize;
    int end = std::min(M, start + safeTileSize);
    for (int row = start; row < end; row++) {
      partition[cnt] = row;
      parType[cnt] = 1;
      cnt++;
    }
    parPtr[numTiles + tile + 1] = cnt;
  }

  int** out = new int*[4];
  out[0] = levelPtr;
  out[1] = parPtr;
  out[2] = partition;
  out[3] = parType;
  return out;
}

std::vector<int*> swiftware::fusion::createSchedule(int32_t* Ap, int32_t* Ai, int64_t ARows, int64_t MTileSize){
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

int swiftware::fusion::calculateWorkingSetSizeForGCNLayer(int Nnz, int UniqueColNnz, int BCol, int CCol, int TileSize, int DataSize) {
  return (CCol * BCol * 2 + TileSize * CCol + TileSize * BCol + UniqueColNnz * BCol + UniqueColNnz * CCol) * DataSize + Nnz * 4;
}


void swiftware::fusion::fusedMKLGeMMSpMMTransposedWeight(
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

void swiftware::fusion::fusedMKLGeMMSpMM(
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

void swiftware::fusion::fusedMKLGeMMSpMMTransposedWeightVT(
        const int M, const int *__restrict__ Ap, const int *__restrict__ Ai,
        const float *__restrict__ Ax, const int InputChannelDim,
        const int OutDim1, const float * Features,
        const float * Weight1, float * Output,
        const int NumThreads, const int LevelNo, const int *LevelPtr,
        const int *MixPtr, const int *L2Ptr) {
    int numKernels = 2;
    int k1Counter = 0;
    int residueStart1 = OutDim1 - OutDim1 % 32;
    float *imRes1 = new float[M * OutDim1]{};
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
                    CblasRowMajor, CblasNoTrans, CblasTrans, tileSize, OutDim1,
                    InputChannelDim, 1., Features + kBeginL1 * InputChannelDim,
                    InputChannelDim, Weight1, InputChannelDim, 0.,
                    imRes1 + kBeginL1 * OutDim1, OutDim1);
            int kBeginL2 = MixPtr[j1 * numKernels + 1];
            int kEndL2 = MixPtr[(j1+1) * numKernels + 1];
            rowWiseSpMMKernel(Ap, Ai, Ax, OutDim1, Output, L2Ptr, residueStart1,
                                        imRes1, kBeginL2, kEndL2);
            }
        }
    }
    delete[] imRes1;
}

void swiftware::fusion::fusedMKLGeMMSpMMVT(
        const int M, const int *__restrict__ Ap, const int *__restrict__ Ai,
        const float *__restrict__ Ax, const int InputChannelDim,
        const int OutDim1, const float * Features,
        const float * Weight1, float * Output,
        const int NumThreads, const int LevelNo, const int *LevelPtr,
        const int *MixPtr, const int *L2Ptr) {
    int numKernels = 2;
    int residueStart1 = OutDim1 - OutDim1 % 32;
    float *imRes1 = new float[M * OutDim1]{};
    for (int i1 = 0; i1 < LevelNo; i1++) {
#pragma omp parallel num_threads(NumThreads)
        {
#pragma omp for
        for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1++) {
            int kBeginL1 = MixPtr[j1 * numKernels];
            int kEndL1 = MixPtr[(j1+1) * numKernels];
            int tileSize = kEndL1 - kBeginL1;
            cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, tileSize, OutDim1,
                    InputChannelDim, 1., Features + kBeginL1 * InputChannelDim,
                    InputChannelDim, Weight1, OutDim1, 0.,
                    imRes1 + kBeginL1 * OutDim1, OutDim1);
            int kBeginL2 = MixPtr[j1 * numKernels + 1];
            int kEndL2 = MixPtr[(j1+1) * numKernels + 1];
            rowWiseSpMMKernel(Ap, Ai, Ax, OutDim1, Output, L2Ptr, residueStart1,
                                        imRes1, kBeginL2, kEndL2);
            }
        }
    }
    delete[] imRes1;
}

void swiftware::fusion::fusedSpMMSpMMVectorizedAvx256SP(
    int M, int FeatureDim, const int *Ap, const int *Ai, const float *Ax,
    const int *Bp, const int *Bi, const float *Bx, const float *Features,
    float *Output, float *Intermediate, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType,
    int NumThreads) {
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NumThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            for (int kk = 0; kk < FeatureDim; kk += 32) {
              auto acxV1 = _mm256_loadu_ps(Intermediate + i * FeatureDim + kk);
              auto acxV2 = _mm256_loadu_ps(Intermediate + i * FeatureDim + kk + 8);
              auto acxV3 = _mm256_loadu_ps(Intermediate + i * FeatureDim + kk + 16);
              auto acxV4 = _mm256_loadu_ps(Intermediate + i * FeatureDim + kk + 24);
              int j = Ap[i];
              for (; j < Ap[i + 1] - 1; j += 2) {
                int aij1 = Ai[j] * FeatureDim;
                int aij2 = Ai[j + 1] * FeatureDim;
                auto axV1 = _mm256_set1_ps(Ax[j]);
                auto axV2 = _mm256_set1_ps(Ax[j + 1]);
                auto cxV11 = _mm256_loadu_ps(Features + aij1 + kk);
                auto cxV12 = _mm256_loadu_ps(Features + aij1 + kk + 8);
                auto cxV13 = _mm256_loadu_ps(Features + aij1 + kk + 16);
                auto cxV14 = _mm256_loadu_ps(Features + aij1 + kk + 24);
                auto cxV21 = _mm256_loadu_ps(Features + aij2 + kk);
                auto cxV22 = _mm256_loadu_ps(Features + aij2 + kk + 8);
                auto cxV23 = _mm256_loadu_ps(Features + aij2 + kk + 16);
                auto cxV24 = _mm256_loadu_ps(Features + aij2 + kk + 24);
                acxV1 = _mm256_fmadd_ps(axV1, cxV11, acxV1);
                acxV1 = _mm256_fmadd_ps(axV2, cxV21, acxV1);
                acxV2 = _mm256_fmadd_ps(axV1, cxV12, acxV2);
                acxV2 = _mm256_fmadd_ps(axV2, cxV22, acxV2);
                acxV3 = _mm256_fmadd_ps(axV1, cxV13, acxV3);
                acxV3 = _mm256_fmadd_ps(axV2, cxV23, acxV3);
                acxV4 = _mm256_fmadd_ps(axV1, cxV14, acxV4);
                acxV4 = _mm256_fmadd_ps(axV2, cxV24, acxV4);
              }
              for (; j < Ap[i + 1]; ++j) {
                int aij = Ai[j] * FeatureDim;
                auto axv0 = _mm256_set1_ps(Ax[j]);
                auto cxV11 = _mm256_loadu_ps(Features + aij + kk);
                auto cxV12 = _mm256_loadu_ps(Features + aij + kk + 8);
                auto cxV13 = _mm256_loadu_ps(Features + aij + kk + 16);
                auto cxV14 = _mm256_loadu_ps(Features + aij + kk + 24);
                acxV1 = _mm256_fmadd_ps(axv0, cxV11, acxV1);
                acxV2 = _mm256_fmadd_ps(axv0, cxV12, acxV2);
                acxV3 = _mm256_fmadd_ps(axv0, cxV13, acxV3);
                acxV4 = _mm256_fmadd_ps(axv0, cxV14, acxV4);
              }
              _mm256_storeu_ps(Intermediate + i * FeatureDim + kk, acxV1);
              _mm256_storeu_ps(Intermediate + i * FeatureDim + kk + 8, acxV2);
              _mm256_storeu_ps(Intermediate + i * FeatureDim + kk + 16, acxV3);
              _mm256_storeu_ps(Intermediate + i * FeatureDim + kk + 24, acxV4);
            }
          } else {
            for (int kk = 0; kk < FeatureDim; kk += 32) {
              auto dxV1 = _mm256_loadu_ps(Output + i * FeatureDim + kk);
              auto dxV2 = _mm256_loadu_ps(Output + i * FeatureDim + kk + 8);
              auto dxV3 = _mm256_loadu_ps(Output + i * FeatureDim + kk + 16);
              auto dxV4 = _mm256_loadu_ps(Output + i * FeatureDim + kk + 24);
              int k = Bp[i];
              for (; k < Bp[i + 1] - 1; k += 2) {
                int bij1 = Bi[k] * FeatureDim;
                int bij2 = Bi[k + 1] * FeatureDim;
                auto bxV1 = _mm256_set1_ps(Bx[k]);
                auto bxV2 = _mm256_set1_ps(Bx[k + 1]);
                auto acxV11 = _mm256_loadu_ps(Intermediate + bij1 + kk);
                auto acxV12 = _mm256_loadu_ps(Intermediate + bij1 + kk + 8);
                auto acxV13 = _mm256_loadu_ps(Intermediate + bij1 + kk + 16);
                auto acxV14 = _mm256_loadu_ps(Intermediate + bij1 + kk + 24);
                auto acxV21 = _mm256_loadu_ps(Intermediate + bij2 + kk);
                auto acxV22 = _mm256_loadu_ps(Intermediate + bij2 + kk + 8);
                auto acxV23 = _mm256_loadu_ps(Intermediate + bij2 + kk + 16);
                auto acxV24 = _mm256_loadu_ps(Intermediate + bij2 + kk + 24);
                dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
                dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
                dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
                dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
                dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
                dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
                dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
                dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
              }
              for (; k < Bp[i + 1]; ++k) {
                int bij = Bi[k] * FeatureDim;
                auto bxv0 = _mm256_set1_ps(Bx[k]);
                auto cxV11 = _mm256_loadu_ps(Intermediate + bij + kk);
                auto cxV12 = _mm256_loadu_ps(Intermediate + bij + kk + 8);
                auto cxV13 = _mm256_loadu_ps(Intermediate + bij + kk + 16);
                auto cxV14 = _mm256_loadu_ps(Intermediate + bij + kk + 24);
                dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
                dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
                dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
                dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
              }
              _mm256_storeu_ps(Output + i * FeatureDim + kk, dxV1);
              _mm256_storeu_ps(Output + i * FeatureDim + kk + 8, dxV2);
              _mm256_storeu_ps(Output + i * FeatureDim + kk + 16, dxV3);
              _mm256_storeu_ps(Output + i * FeatureDim + kk + 24, dxV4);
            }
          }
        }
      }
    }
  }
}

inline void
swiftware::fusion::rowWiseSpMMKernel(const int *Ap, const int *Ai, const float *Ax, const int OutputChannelDim, float *Output,
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

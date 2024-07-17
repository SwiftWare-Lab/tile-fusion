//
// Created by salehm32 on 03/11/23.
//

#ifndef SPARSE_FUSION_FUSIONINSPECTOR_H
#define SPARSE_FUSION_FUSIONINSPECTOR_H

#include "Stats.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include "sparse-fusion/SparseFusion.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <set>

using namespace swiftware::benchmark;

// inter-layer fusion scheduler from sparse fusion on SpMM-SpMM
// each layer is fused(SpMM of GeMVs)
class InspectorForAllFused {
protected:
  sym_lib::ScheduleParameters Sp;
  Stats *St;

public:
  InspectorForAllFused(sym_lib::ScheduleParameters Sp1, Stats *St1)
      : Sp(Sp1), St(St1) {}

  sym_lib::MultiDimensionalSet *
  generateFusedScheduleForAllFused(sym_lib::CSR *AdjMtx) {
    Sp._num_w_partition =
        std::max<int>(AdjMtx->m / Sp.IterPerPartition, 2 * Sp._num_threads);
    auto *sf01 = new sym_lib::SparseFusion(&Sp, 2);
    auto *mvDAG = sym_lib::diagonal(AdjMtx->m, 1.0);
    auto *tmpCSCCSR = new sym_lib::CSC(AdjMtx->m, AdjMtx->n, AdjMtx->nnz,
                                       AdjMtx->p, AdjMtx->i, AdjMtx->x);
    // sym_lib::print_csc(1, "Di", 6, Di->p, Di->i, Di->x);
    sf01->fuse(0, mvDAG, tmpCSCCSR);

    // sf01->print_final_list();
    sf01->fuse(1, mvDAG, tmpCSCCSR);
    // sf01->print_final_list();
    auto pt = St->OtherStats["PackingType"];
    sym_lib::MultiDimensionalSet *fusedCompSet =
        sf01->getFusedCompressed((int)pt[0]);
    //    FusedCompSet->print_3d();
    delete sf01;
    delete mvDAG;
    delete tmpCSCCSR;
    return fusedCompSet;
  }
};

// inter-layer fusion scheduler for tiledFusedCSC where each layer use
// parallelized GeMMs
class InspectorForAllTiledFusedCSC {
protected:
public:
  sym_lib::MultiDimensionalSet *
  generateFusedScheduleForAllTiledFusedCSC(sym_lib::CSC *AdjMtx, int TileSize) {
    sym_lib::MultiDimensionalSet *fusedCompSet =
        new sym_lib::MultiDimensionalSet();
    int numOfTiles = AdjMtx->m / TileSize;
    if (AdjMtx->m % TileSize > 0) {
      numOfTiles = AdjMtx->m / TileSize + 1;
    }
    int numOfComputeNodes = numOfTiles * 2;
    fusedCompSet->ptr1_ = new int[2];
    fusedCompSet->ptr1_[0] = 0;
    fusedCompSet->ptr1_[1] = numOfComputeNodes;
    fusedCompSet->id_ = new int[numOfComputeNodes];
    fusedCompSet->type_ = new int[numOfComputeNodes];
    int rowsLastTile[AdjMtx->m];
    // Finding last tile that has nonzero in each row
    findRowsLastNonzeroTile(AdjMtx->m, TileSize, AdjMtx, rowsLastTile);
    int tileLastNeededTile[numOfTiles];
    // Finding last tile that is needed to be computed in the first loop for
    // each tile in the second loop
    findLastTileNeededInFirstLoopForEachTileInSecondLoop(
        AdjMtx->m, numOfTiles, TileSize, AdjMtx, rowsLastTile,
        tileLastNeededTile);
    // Generating fused schedule based on output of previous two steps
    generateFusedSchedule(AdjMtx->m, numOfTiles, TileSize, tileLastNeededTile,
                          fusedCompSet);
    return fusedCompSet;
  }

private:
  void findRowsLastNonzeroTile(int NumOfNodes, int TileSize,
                               const sym_lib::CSC *AdjMtx, int *RowsLastTile) {
    for (int i = 0; i < NumOfNodes; i += TileSize) {
      for (int ii = 0; ii < TileSize; ++ii) {
        if (i + ii >= NumOfNodes) {
          break;
        }
        for (int j = AdjMtx->p[i + ii]; j < AdjMtx->p[i + ii + 1]; ++j) {
          RowsLastTile[AdjMtx->i[j]] = i / TileSize;
        }
      }
    }
  }

  void findLastTileNeededInFirstLoopForEachTileInSecondLoop(
      int NumOfNodes, int NumOfTiles, int TileSize, const sym_lib::CSC *AdjMtx,
      const int *RowsLastTile, int *TileLastNeededTile) {
    memset(TileLastNeededTile, 0, NumOfTiles * sizeof(int));
    for (int i = 0; i < NumOfNodes; i += TileSize) {
      for (int ii = 0; ii < TileSize; ++ii) {
        if (i + ii >= NumOfNodes) {
          break;
        }
        for (int j = AdjMtx->p[i + ii]; j < AdjMtx->p[i + ii + 1]; j++) {
          if (RowsLastTile[AdjMtx->i[j]] > TileLastNeededTile[i / TileSize]) {
            TileLastNeededTile[i / TileSize] = RowsLastTile[AdjMtx->i[j]];
          }
        }
      }
    }
  }

  void generateFusedSchedule(int NumOfNodes, int NumOfTiles, int TileSize,
                             const int *TileLastNeededTile,
                             sym_lib::MultiDimensionalSet *FusedCompSet) {
    int idCounter = 0;
    for (int i = 0; i < NumOfNodes; i += TileSize) {
      FusedCompSet->id_[idCounter] = i;
      FusedCompSet->type_[idCounter] = 0;
      if (i != NumOfNodes - TileSize && i / TileSize == NumOfTiles - 1) {
        FusedCompSet->type_[idCounter] = 2;
      }
      idCounter++;
      for (int k = 0; k < NumOfNodes; k += TileSize) {
        if (TileLastNeededTile[k / TileSize] == i / TileSize) {
          FusedCompSet->id_[idCounter] = k;
          FusedCompSet->type_[idCounter] = 1;
          if (k != NumOfNodes - TileSize && k / TileSize == NumOfTiles - 1) {
            FusedCompSet->type_[idCounter] = 3;
          }
          idCounter++;
        }
      }
    }
  }
};

struct TiledFusedLayerSchedulingParameters {
  int *GeMMUpperBounds;
  int *GeMMLowerBounds;
  int MaxGeMMTileSize;

  TiledFusedLayerSchedulingParameters() {
    MaxGeMMTileSize = 0;
    GeMMUpperBounds = nullptr;
    GeMMLowerBounds = nullptr;
  }
  ~TiledFusedLayerSchedulingParameters() {
    delete[] GeMMUpperBounds;
    delete[] GeMMLowerBounds;
  }
};

// intra-layer fusion scheduler for tiledFusedParallel where for each
// row-wised tile in CSR matrix, range of GeMM that should be computed is
// recognized and is given in the schedule dataset
class InspectorForTiledFused {
public:
  TiledFusedLayerSchedulingParameters *
  generateGeMMTileForEachSpMMTile(sym_lib::CSR *AdjMtx, int TileSize) {
    TiledFusedLayerSchedulingParameters *sp =
        new TiledFusedLayerSchedulingParameters();
    int numOfTiles = ceil((double)AdjMtx->m / TileSize);
    sp->GeMMLowerBounds = new int[numOfTiles];
    sp->GeMMUpperBounds = new int[numOfTiles];
    for (int i = 0; i < AdjMtx->m; i += TileSize) {
      int smallestIndex = AdjMtx->m;
      int biggestIndex = 0;
      for (int ii = 0; ii < TileSize; ii++) {
        if (i + ii >= AdjMtx->m) {
          break;
        }
        if (AdjMtx->i[AdjMtx->p[i + ii]] < smallestIndex) {
          smallestIndex = AdjMtx->i[AdjMtx->p[i + ii]];
        }
        if (AdjMtx->i[AdjMtx->p[i + ii + 1] - 1] > biggestIndex) {
          biggestIndex = AdjMtx->i[AdjMtx->p[i + ii + 1] - 1];
        }
      }
      sp->GeMMLowerBounds[i / TileSize] = smallestIndex;
      sp->GeMMUpperBounds[i / TileSize] = biggestIndex + 1;
      if (biggestIndex - smallestIndex + 1 > sp->MaxGeMMTileSize) {
        sp->MaxGeMMTileSize = biggestIndex - smallestIndex + 1;
      }
    }
    return sp;
  }
};

// intera-layer fusion scheduler for tiledFusedCSCParallel that uses coloring of
// conflict graph to generate create wave fronts.
class InspectorForSingleLayerTiledFusedCSCParallel {
public:
  sym_lib::MultiDimensionalSet *
  generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
      std::map<int, std::vector<int>> ColorToTiles, int NumOfNodes,
      int TileSize) {
    int numOfTiles = (int)ceil((double)NumOfNodes / TileSize);
    sym_lib::MultiDimensionalSet *fusedCompSet =
        new sym_lib::MultiDimensionalSet();
    fusedCompSet->n1_ = ColorToTiles.rbegin()->first + 1;
    fusedCompSet->ptr1_ = new int[fusedCompSet->n1_ + 1];
    fusedCompSet->ptr1_[0] = 0;
    fusedCompSet->id_ = new int[numOfTiles];
    for (std::map<int, std::vector<int>>::iterator it = ColorToTiles.begin();
         it != ColorToTiles.end(); ++it) {
      fusedCompSet->ptr1_[it->first + 1] =
          fusedCompSet->ptr1_[it->first] + it->second.size();
      for (int i = 0; i < it->second.size(); i++) {
        fusedCompSet->id_[fusedCompSet->ptr1_[it->first] + i] = it->second[i];
      }
    }
    fusedCompSet->type_ = new int[numOfTiles];
    for (int i = 0; i < numOfTiles - 1; i++) {
      fusedCompSet->type_[i] = TileSize;
    }
    if (NumOfNodes % TileSize == 0) {
      fusedCompSet->type_[numOfTiles - 1] = TileSize;
    } else {
      fusedCompSet->type_[numOfTiles - 1] = NumOfNodes % TileSize;
    }
    return fusedCompSet;
  }

protected:
  int printColoring(std::map<std::string, int> &coloring,
                    std::map<int, std::vector<int>> &colorToTiles) const {
    int maxNum = 0;
    for (std::map<std::string, int>::iterator it = coloring.begin();
         it != coloring.end(); ++it) {
      if (it->second > maxNum)
        maxNum = it->second;
    }
    for (std::map<int, std::vector<int>>::iterator it = colorToTiles.begin();
         it != colorToTiles.end(); ++it) {
      std::cout << it->first << " : " << it->second.size() << std::endl;
    }
    std::cout << maxNum << std::endl;
    return maxNum;
  }
};

// Intera-layer fusion scheduler for tiledFusedCSCParallel that uses coloring of
// conflict graph to generate create wave fronts. This class implements kTiling
// on the ordinary conflict graph and generates new copies from each computation
// in the same wavefront to have kTiles.
class InspectorForSingleLayerTiledFusedCSCParallelWithReplicatedKTiles
    : public InspectorForSingleLayerTiledFusedCSCParallel {
public:
  sym_lib::MultiDimensionalSet *generateScheduleBasedOnConflictGraphColoring(
      std::map<int, std::vector<int>> ColorToTiles, int NumOfNodes,
      int TileSize, int OutputSize, int KTileSize) {
    int numOfTiles = (int)ceil((double)NumOfNodes / TileSize);
    int numOfKTiles = OutputSize / KTileSize;
    sym_lib::MultiDimensionalSet *fusedCompSet =
        new sym_lib::MultiDimensionalSet();
    fusedCompSet->n1_ = ColorToTiles.rbegin()->first + 1;
    fusedCompSet->ptr1_ = new int[fusedCompSet->n1_ + 1];
    fusedCompSet->ptr1_[0] = 0;
    fusedCompSet->id_ = new int[numOfTiles * numOfKTiles];
    for (std::map<int, std::vector<int>>::iterator it = ColorToTiles.begin();
         it != ColorToTiles.end(); ++it) {
      fusedCompSet->ptr1_[it->first + 1] =
          fusedCompSet->ptr1_[it->first] + it->second.size() * numOfKTiles;
      for (int i = 0; i < it->second.size(); i++) {
        for (int k = 0; k < numOfKTiles; k++){
          fusedCompSet->id_[fusedCompSet->ptr1_[it->first] + i*numOfKTiles + k] = it->second[i];
        }
      }
    }
    fusedCompSet->type_ = new int[numOfTiles];
    for (int i = 0; i < numOfTiles - 1; i++) {
      fusedCompSet->type_[i] = TileSize;
    }
    if (NumOfNodes % TileSize == 0) {
      fusedCompSet->type_[numOfTiles - 1] = TileSize;
    } else {
      fusedCompSet->type_[numOfTiles - 1] = NumOfNodes % TileSize;
    }
    return fusedCompSet;
  }
};

class InspectorForSingleLayerTiledFusedCSCParallelWithSchedulingKTiles
    : public InspectorForSingleLayerTiledFusedCSCParallel {
public:
  sym_lib::MultiDimensionalSet *generateScheduleBasedOnConflictGraphColoring(
      std::map<int, std::vector<int>> ColorToTiles, int NumOfNodes,
      int TileSize, int OutputSize, int KTileSize) {
    int numOfTiles = (int)ceil((double)NumOfNodes / TileSize);
    int numOfKTiles = OutputSize / KTileSize;
    sym_lib::MultiDimensionalSet *fusedCompSet =
        new sym_lib::MultiDimensionalSet();
    fusedCompSet->n1_ = ColorToTiles.rbegin()->first + 1;
    fusedCompSet->ptr1_ = new int[fusedCompSet->n1_ + 1];
    fusedCompSet->ptr1_[0] = 0;
    fusedCompSet->ptr2_ = new int[numOfTiles * numOfKTiles];
    fusedCompSet->id_ = new int[numOfTiles * numOfKTiles];
    for (std::map<int, std::vector<int>>::iterator it = ColorToTiles.begin();
         it != ColorToTiles.end(); ++it) {
      fusedCompSet->ptr1_[it->first + 1] =
          fusedCompSet->ptr1_[it->first] + it->second.size();
      for (int i = 0; i < it->second.size(); i++) {
        fusedCompSet->id_[fusedCompSet->ptr1_[it->first] + i] = it->second[i];
      }
    }
    fusedCompSet->type_ = new int[numOfTiles];
    for (int i = 0; i < numOfTiles - 1; i++) {
      fusedCompSet->type_[i] = TileSize;
    }
    if (NumOfNodes % TileSize == 0) {
      fusedCompSet->type_[numOfTiles - 1] = TileSize;
    } else {
      fusedCompSet->type_[numOfTiles - 1] = NumOfNodes % TileSize;
    }
    return fusedCompSet;
  }
};

// Intera-layer fusion scheduler for tiledFusedCSC where create wavefronts based
// on conflict graph of CSC matrix. for wavefronts with less than a specific
// amount of nodes, it merges the computation nodes(tiles) and run the new nodes
// using parallelized GeMMs.
// n1_: number of parallel workloads
// n2_: number of sequential region tiles (with parallel GeMMs)
// n3_: max tile size in the sequential region tiles
// ptr1_: 0->n1_ + 1: start and end of each workload
// id_: 0->ptr[n1_ + 1]: parallel region tiles
// id_: ptr[n1_ + 1]->ptr[n1_ + 1]+n2_: sequential region tiles
// type_: pointer array to start and end of each tile
class InspectorForSingleLayerTiledFusedCSCCombined
    : public InspectorForSingleLayerTiledFusedCSCParallel {

public:
  virtual sym_lib::MultiDimensionalSet *
  generateScheduleBasedOnConflictGraphColoring(
      std::map<int, std::vector<int>> ColorToTiles, int NumOfNodes,
      int TileSize, int WorkloadMinSize) {
    std::vector<std::vector<int>> workloads;
    std::set<int> sequentialRegionTiles;
    for (auto tileGroup : ColorToTiles) {
      if (tileGroup.second.size() >= WorkloadMinSize) {
        workloads.push_back(tileGroup.second);
      } else {
        sequentialRegionTiles.insert(tileGroup.second.begin(),
                               tileGroup.second.end());
      }
    }
    std::map<int, int> sequentialRegionTileToNewSize =
        getNewSequentialRegionTilesMap(sequentialRegionTiles);
    std::map<int, int> tileToNewSize =
        updateNewTilesMapWithWorkloadTiles(workloads, sequentialRegionTileToNewSize);
    int numOfNewTiles = tileToNewSize.size();
    int *tilePtr = new int[numOfNewTiles + 1];
    int maxTileSize = 0;
    tilePtr[0] = 0;
    int i = 0;
    for (auto tile : tileToNewSize) {
      int newTileSize = tile.second * TileSize;
      tilePtr[i + 1] = tilePtr[i] + newTileSize;
      if (newTileSize > maxTileSize) {
        maxTileSize = newTileSize;
      }
      i++;
    }
    tilePtr[numOfNewTiles] = NumOfNodes;
    std::map<int, int> oldToNewIndex;
    for (auto it = tileToNewSize.begin(); it != tileToNewSize.end(); it++) {
      oldToNewIndex[it->first] = std::distance(tileToNewSize.begin(), it);
    }
    sym_lib::MultiDimensionalSet *fusedCompSet =
        new sym_lib::MultiDimensionalSet();
    fusedCompSet->n1_ = workloads.size();
    fusedCompSet->ptr1_ = new int[fusedCompSet->n1_ + 1];
    fusedCompSet->n2_ = sequentialRegionTileToNewSize.size();
    fusedCompSet->n3_ = maxTileSize;
    fusedCompSet->id_ = new int[tileToNewSize.size()];
    fusedCompSet->type_ = tilePtr;
    fusedCompSet->ptr1_[0] = 0;
    for (i = 0; i < workloads.size(); i++) {
      fusedCompSet->ptr1_[i + 1] = fusedCompSet->ptr1_[i] + workloads[i].size();
      for (int j = 0; j < workloads[i].size(); j++) {
        fusedCompSet->id_[fusedCompSet->ptr1_[i] + j] =
            oldToNewIndex[workloads[i][j]];
      }
    }
    int standAloneTilesStart = fusedCompSet->ptr1_[fusedCompSet->n1_];
    i = standAloneTilesStart;
    for (auto saTile : sequentialRegionTileToNewSize) {
      fusedCompSet->id_[i] = oldToNewIndex[saTile.first];
      i++;
    }
    return fusedCompSet;
  }

protected:
  std::map<int, int> updateNewTilesMapWithWorkloadTiles(
      std::vector<std::vector<int>> Workloads,
      std::map<int, int> StandAloneTileToNewSize) {
    std::map<int, int> tileToNewSize(StandAloneTileToNewSize);
    for (auto workload : Workloads) {
      for (auto tile : workload) {
        tileToNewSize[tile] = 1;
      }
    }
    return tileToNewSize;
  }

  std::map<int, int>
  getNewSequentialRegionTilesMap(std::set<int> StandAloneTiles) {
    std::map<int, int> tileToNewSize;
    tileToNewSize[*StandAloneTiles.begin()] = 1;
    StandAloneTiles.erase(StandAloneTiles.begin());
    for (auto tile : StandAloneTiles) {
      auto lastNewTile = tileToNewSize.rbegin();
      if (tile - lastNewTile->first == lastNewTile->second) {
        lastNewTile->second++;
      } else {
        tileToNewSize[tile] = 1;
      }
    }
    return tileToNewSize;
  }
};

class InspectorForSingleLayerTiledFusedCSCCombinedWithKTiling
    : public InspectorForSingleLayerTiledFusedCSCCombined {
public:
  sym_lib::MultiDimensionalSet *generateScheduleBasedOnConflictGraphColoring(
      std::map<int, std::vector<int>> ColorToTiles, int NumOfNodes,
      int TileSize, int WorkloadMinSize, int OutputSize, int KTileSize) {
    std::vector<std::vector<int>> workloads;
    std::set<int> standAloneTiles;
    int numOfKTiles = OutputSize / KTileSize;
    for (auto tileGroup : ColorToTiles) {
      if (tileGroup.second.size() >= WorkloadMinSize) {
        workloads.push_back(tileGroup.second);
      } else {
        standAloneTiles.insert(tileGroup.second.begin(),
                               tileGroup.second.end());
      }
    }
    std::map<int, int> standAloneTileToNewSize =
        getNewSequentialRegionTilesMap(standAloneTiles);
    std::map<int, int> tileToNewSize =
        updateNewTilesMapWithWorkloadTiles(workloads, standAloneTileToNewSize);
    int numOfNewTiles = tileToNewSize.size();
    int *tilePtr = new int[numOfNewTiles + 1];
    int maxTileSize = 0;
    tilePtr[0] = 0;
    int i = 0;
    for (auto tile : tileToNewSize) {
      int newTileSize = tile.second * TileSize;
      tilePtr[i + 1] = tilePtr[i] + newTileSize;
      if (newTileSize > maxTileSize) {
        maxTileSize = newTileSize;
      }
      i++;
    }
    tilePtr[numOfNewTiles] = NumOfNodes;
    std::map<int, int> oldToNewIndex;
    for (auto it = tileToNewSize.begin(); it != tileToNewSize.end(); it++) {
      oldToNewIndex[it->first] = std::distance(tileToNewSize.begin(), it);
    }
    sym_lib::MultiDimensionalSet *fusedCompSet =
        new sym_lib::MultiDimensionalSet();
    fusedCompSet->n1_ = workloads.size();
    fusedCompSet->ptr1_ = new int[fusedCompSet->n1_ + 1];
    fusedCompSet->n2_ = standAloneTileToNewSize.size();
    fusedCompSet->n3_ = maxTileSize;
    fusedCompSet->id_ = new int[tileToNewSize.size() * numOfKTiles];
    fusedCompSet->type_ = tilePtr;
    fusedCompSet->ptr1_[0] = 0;
    for (i = 0; i < workloads.size(); i++) {
      fusedCompSet->ptr1_[i + 1] =
          fusedCompSet->ptr1_[i] + workloads[i].size() * numOfKTiles;
      for (int j = 0; j < workloads[i].size(); j++) {
        for (int k = 0; k < numOfKTiles; k++) {
          fusedCompSet->id_[fusedCompSet->ptr1_[i] + j * numOfKTiles + k] =
              oldToNewIndex[workloads[i][j]];
        }
      }
    }
    int standAloneTilesStart = fusedCompSet->ptr1_[fusedCompSet->n1_];
    i = standAloneTilesStart;
    for (auto saTile : standAloneTileToNewSize) {
      fusedCompSet->id_[i] = oldToNewIndex[saTile.first];
      i++;
    }
    return fusedCompSet;
  }
};

class InspectorForTileFusedCSRVariableTileSize{
protected:
  sym_lib::ScheduleParameters Sp;
  Stats *St;
  struct VariableTile{
    int Start;
    int End;
    std::vector<int> FusedIters;
    VariableTile* Next;
    VariableTile(int Start, int End){
      this->Start = Start;
      this->End = End;
      this->Next = NULLPNTR;
    }
  };

  //not used for now but later can replace the above struct.
  struct UFVariableTilePtr{
    int Index;
    UFVariableTilePtr* Next;
    UFVariableTilePtr(int Index){
      this->Index = Index;
      this->Next = NULLPNTR;
    }
  };
public:
  InspectorForTileFusedCSRVariableTileSize(sym_lib::ScheduleParameters Sp1, Stats *St1)
      : Sp(Sp1), St(St1) {}
  sym_lib::MultiDimensionalSet* generateVariableTileSizeSchedule(sym_lib::CSR *ACsr, int BCol, int DataSize=8){
    std::vector<VariableTile> pTiles;
    std::vector<int> unfusedIters;
    int CACHE_SIZE = Sp.TileM;
    int *ai = ACsr->i;
    int *ap = ACsr->p;
    int initialTileSize = std::min(4096,int(ACsr->m));
    int extraIters = ACsr->m % initialTileSize;
    int extraRemoved = 0;
    int numOfTiles = ACsr->m / initialTileSize;

    int extraIterPerTile = std::ceil(extraIters / double(numOfTiles));
    VariableTile* head = new VariableTile(0,0);
    VariableTile* curr = head;
    //create initial tiles
    for (int i = 0; i < numOfTiles; i++) {
      int start = initialTileSize * i + extraRemoved;
      int end = start + initialTileSize;
      if (extraIters > extraRemoved) {
        int ext = std::min(extraIters-extraRemoved, extraIterPerTile);
        end += ext;
        extraRemoved += ext;
      }
      auto *vt = new VariableTile(start, end);
      curr->Next = vt;
      curr = curr->Next;
    }
    //    std::cout << extraIters << " " << extraRemoved << std::endl;
    // create initial tiles fused iterations
    for (int i = 0; i < ACsr->m; i++) {
      bool isFused = false;
      curr = head;
      while (curr->Next != NULLPNTR){
        curr = curr->Next;
        if (ai[ap[i]] >= curr->Start && ai[ap[i + 1] - 1] < curr->End) {
          curr->FusedIters.push_back(i);
          isFused = true;
          break;
        }
      }
      if (!isFused)
        unfusedIters.push_back(i);
    }
    // shrinking tiles
    curr = head;
    while(curr->Next != NULLPNTR){
      auto *prev = curr;
      curr = curr->Next;
      int tileSize = curr->End - curr->Start;
      int* firstColPtr = ai + ap[curr->Start];
      int* lastColPtr = ai + ap[curr->End];
      int nnzNum = ap[curr->End]-ap[curr->Start];
      std::set<int> uniqueColumns(firstColPtr,lastColPtr);
      int workingSet = calculateWorkingSetSize(nnzNum, uniqueColumns.size(), BCol, tileSize, curr->FusedIters.size(), DataSize);
      if (workingSet > CACHE_SIZE && tileSize > 1){
        int separator = tileSize/2 + curr->Start;
        auto *vt1 = new VariableTile(curr->Start, separator);
        auto *vt2 = new VariableTile(separator, curr->End);
        for (auto fi: curr->FusedIters){
          if (ai[ap[fi+1]-1] < separator){
            vt1->FusedIters.push_back(fi);
          }
          else if(ai[ap[fi]] >= separator){
            vt2->FusedIters.push_back(fi);
          }
          else{
            unfusedIters.push_back(fi);
          }
        }
        vt1->Next = vt2;
        vt2->Next = curr->Next;
        prev->Next = vt1;
        numOfTiles += 1;
        delete curr;
        curr = prev;
      }
    }
    // creating schedule multi dimensional set
    sym_lib::MultiDimensionalSet* fusedCompSet = new sym_lib::MultiDimensionalSet();
    fusedCompSet->n1_ = 2;
    fusedCompSet->ptr1_ = new int[3];
    fusedCompSet->ptr1_[0] = 0;
    fusedCompSet->ptr1_[1] = numOfTiles;
    fusedCompSet->ptr1_[2] = numOfTiles + Sp._num_threads;
    fusedCompSet->ptr2_ = new int[numOfTiles + Sp._num_threads + 1];
    fusedCompSet->id_ = new int[2 * ACsr->m];
    fusedCompSet->type_ = new int[2 * ACsr->m];
    fusedCompSet->ptr2_[0] = 0;
    int cnt = 0;
    int pCounter = 0;
    curr = head;
    while(curr->Next != NULLPNTR){
      curr = curr->Next;
      for (int j = curr->Start; j < curr->End; j++) {
        fusedCompSet->id_[cnt] = j;
        fusedCompSet->type_[cnt] = 0;
        cnt++;
      }
      for (int fi : curr->FusedIters) {
        fusedCompSet->id_[cnt] = fi;
        fusedCompSet->type_[cnt] = 1;
        cnt++;
      }
      fusedCompSet->ptr2_[pCounter + 1] = cnt;
      pCounter+=1;
    }
    extractTilesSizeData(head);
    // delete the tile tree
    curr = head->Next;
    while(curr != NULLPNTR){
      auto *tmp = curr;
      curr = curr->Next;
      delete tmp;
    }
    delete head;
    int unfusedPerPart = ceil(unfusedIters.size() / float(Sp._num_threads));
    for (int i = numOfTiles; i < numOfTiles + Sp._num_threads; i++) {
      int p = i - numOfTiles;
      int partEnd =
          std::min((p + 1) * unfusedPerPart, int(unfusedIters.size()));
      for (int j = p * unfusedPerPart; j < partEnd; j++) {
        fusedCompSet->id_[cnt] = unfusedIters[j];
        fusedCompSet->type_[cnt] = 1;
        cnt++;
      }
      fusedCompSet->ptr2_[i + 1] = cnt;
    }
    int fusedNodesNum = fusedCompSet->getNumberOfFusedNodes();
    int fusedNnzNum = fusedCompSet->getFusedNnzNum(ACsr);
    this->St->OtherStats["Number of Fused Nodes"] = {(double)fusedNodesNum};
    this->St->OtherStats["Number of Fused nnz"] = {(double)fusedNnzNum};
    return fusedCompSet;
  }

  sym_lib::MultiDimensionalSet* generateVariableTileSizeScheduleForBothWavefronts(sym_lib::CSR *ACsr, int BCol, int DataSize=8){
    std::vector<VariableTile> pTiles;
    int CACHE_SIZE = Sp.TileM;
    int *ai = ACsr->i;
    int *ap = ACsr->p;
    int INITIAL_TILE_SIZE = 4096;
    int initialTileSize = std::min(INITIAL_TILE_SIZE,int(ACsr->m));
    int extraIters = ACsr->m % initialTileSize;
    int extraRemoved = 0;
    int numOfTiles = ACsr->m / initialTileSize;
    std::vector<int> unfusedIters;

    int extraIterPerTile = std::ceil(extraIters / double(numOfTiles));
    VariableTile* head = new VariableTile(0,0);
    VariableTile* curr = head;
    //create initial tiles
    for (int i = 0; i < numOfTiles; i++) {
      int start = initialTileSize * i + extraRemoved;
      int end = start + initialTileSize;
      if (extraIters > extraRemoved) {
        int ext = std::min(extraIters-extraRemoved, extraIterPerTile);
        end += ext;
        extraRemoved += ext;
      }
      auto *vt = new VariableTile(start, end);
      curr->Next = vt;
      curr = curr->Next;
    }
    //    std::cout << extraIters << " " << extraRemoved << std::endl;
    // create initial tiles fused iterations

    bool isFused = false;
    curr = head;
    while (curr->Next != NULLPNTR){
      curr = curr->Next;
      for (int i = curr->Start; i < curr->End; i++) {
        if (ai[ap[i]] >= curr->Start && ai[ap[i + 1] - 1] < curr->End) {
          curr->FusedIters.push_back(i);
        }
        else
          unfusedIters.push_back(i);
      }
    }
    // shrinking tiles
    curr = head;
    while(curr->Next != NULLPNTR){
      auto *prev = curr;
      curr = curr->Next;
      int tileSize = curr->End - curr->Start;
      int* firstColPtr = ai + ap[curr->Start];
      int* lastColPtr = ai + ap[curr->End];
      int nnzNum = ap[curr->End]-ap[curr->Start];
      std::set<int> uniqueColumns(firstColPtr,lastColPtr);
      int workingSet = calculateWorkingSetSize(nnzNum, uniqueColumns.size(), BCol, tileSize, curr->FusedIters.size(), DataSize);
      if (workingSet > CACHE_SIZE && tileSize > 1){
        int separator = tileSize/2 + curr->Start;
        auto *vt1 = new VariableTile(curr->Start, separator);
        auto *vt2 = new VariableTile(separator, curr->End);
        for (auto fi: curr->FusedIters){
          if (ai[ap[fi+1]-1] < separator){
            vt1->FusedIters.push_back(fi);
          }
          else if(ai[ap[fi]] >= separator){
            vt2->FusedIters.push_back(fi);
          }
          else{
            unfusedIters.push_back(fi);
          }
        }
        vt1->Next = vt2;
        vt2->Next = curr->Next;
        prev->Next = vt1;
        numOfTiles += 1;
        delete curr;
        curr = prev;
      }
    }
    std::sort(unfusedIters.begin(), unfusedIters.end());
    std::vector<int> ufPartPtr;
    int MIN_STRIDE = 16;
    std::set<int> uniqueColumns;
    int nnzNum = 0;
    int uft = 0;
    int ufTileSize = 0;
    ufPartPtr.push_back(0);
    while (uft < unfusedIters.size()){
      for (int ii = uft; ii < std::min(int(unfusedIters.size()), uft + MIN_STRIDE); ii++){
        int row = unfusedIters[ii];
        uniqueColumns.insert(ai + ap[row], ai + ap[row+1]);
        nnzNum += ap[row+1] - ap[row];
        ufTileSize += 1;
      }
      int workingSet = calculateWorkingSetSize(nnzNum, uniqueColumns.size(), BCol, ufTileSize, 0, DataSize);
      if((workingSet < CACHE_SIZE) || (ufTileSize == 1)){
        uft += MIN_STRIDE;
      }
      else{
        ufPartPtr.push_back(uft);
        nnzNum = 0;
        uniqueColumns.erase(uniqueColumns.begin(), uniqueColumns.end());
        if (ufTileSize <= MIN_STRIDE){
          MIN_STRIDE = MIN_STRIDE / 2;
        }
        if (ufTileSize >= 3*MIN_STRIDE){
          MIN_STRIDE = MIN_STRIDE * 2;
        }
        ufTileSize = 0;
      }
    }
    ufPartPtr.push_back(unfusedIters.size());
    int numUfTiles = ufPartPtr.size() - 1;
        // creating schedule multi dimensional set
    sym_lib::MultiDimensionalSet* fusedCompSet = new sym_lib::MultiDimensionalSet();
    fusedCompSet->n1_ = 2;
    fusedCompSet->ptr1_ = new int[3];
    fusedCompSet->ptr1_[0] = 0;
    fusedCompSet->ptr1_[1] = numOfTiles;
    fusedCompSet->ptr1_[2] = numOfTiles + numUfTiles;
    fusedCompSet->ptr2_ = new int[numOfTiles + numUfTiles + 1];
    fusedCompSet->ker_begin_ = new int[(numOfTiles + numUfTiles)*2];
    fusedCompSet->id_ = new int[2 * ACsr->m];
    fusedCompSet->type_ = new int[2 * ACsr->m];
    fusedCompSet->ptr2_[0] = 0;
    int cnt = 0;
    int pCounter = 0;
    curr = head;
    while(curr->Next != NULLPNTR){
      curr = curr->Next;
      for (int j = curr->Start; j < curr->End; j++) {
        fusedCompSet->id_[cnt] = j;
        fusedCompSet->type_[cnt] = 0;
        cnt++;
      }
      fusedCompSet->ker_begin_[(pCounter)*2] = cnt;
      for (int fi : curr->FusedIters) {
        fusedCompSet->id_[cnt] = fi;
        fusedCompSet->type_[cnt] = 1;
        cnt++;
      }
      fusedCompSet->ker_begin_[(pCounter)*2+1] = cnt;
      fusedCompSet->ptr2_[pCounter + 1] = cnt;
      pCounter+=1;
    }
    extractTilesSizeData(head);
    // delete the tile tree
    curr = head->Next;
    while(curr != NULLPNTR) {
      auto *tmp = curr;
      curr = curr->Next;
      delete tmp;
    }
    delete head;
    for (int i = numOfTiles; i < numOfTiles + numUfTiles; i++) {
      int p = i - numOfTiles;
      int partEnd = ufPartPtr[p+1];
      fusedCompSet->ker_begin_[i*2] = cnt;
      for (int j = ufPartPtr[p]; j < partEnd; j++) {
        fusedCompSet->id_[cnt] = unfusedIters[j];
        fusedCompSet->type_[cnt] = 1;
        cnt++;
      }
      fusedCompSet->ker_begin_[i*2 + 1] = cnt;
      fusedCompSet->ptr2_[i + 1] = cnt;
    }
    int fusedNodesNum = fusedCompSet->getNumberOfFusedNodes();
    int fusedNnzNum = fusedCompSet->getFusedNnzNum(ACsr);
    this->St->OtherStats["Number of Fused Nodes"] = {(double)fusedNodesNum};
    this->St->OtherStats["Number of Fused nnz"] = {(double)fusedNnzNum};
    return fusedCompSet;
  }


  virtual int calculateWorkingSetSize(int Nnz, int UniqueColsNum, int BCol, int TileSize, int FusedIters, int DataSize = 8){
    return (Nnz + UniqueColsNum * BCol + TileSize * BCol + FusedIters * BCol) * DataSize + Nnz * 4;
  }

  void extractTilesSizeData(VariableTile* Head){
    std::vector<int> tileSizes;
    VariableTile* curr = Head;
    while(curr->Next != NULLPNTR){
      curr = curr->Next;
      tileSizes.push_back(curr->End - curr->Start);
    }
    float average = std::accumulate(tileSizes.begin(), tileSizes.end(),0.0) / tileSizes.size();
    float var = 0;
    for( int i = 0; i < tileSizes.size(); i++ )
    {
      var += (tileSizes[i] - average) * (tileSizes[i] - average);
    }
    var /= tileSizes.size();
    float sd = sqrt(var);
    this->St->OtherStats["Tile Size Mean"] = {average};
    this->St->OtherStats["Tile Size STD"] = {sd};
  }
};

class InspectorForTileFusedCSRVariableTileSizeWithKTiles8: public InspectorForTileFusedCSRVariableTileSize{
protected:
  int calculateWorkingSetSize(int Nnz, int UniqueColsNum, int BCol, int TileSize, int FusedIters, int DataSize=8) override{
    return (Nnz + UniqueColsNum * 16 + TileSize* 16 + FusedIters* 16) * DataSize + Nnz*4;
  }
public:
  InspectorForTileFusedCSRVariableTileSizeWithKTiles8(sym_lib::ScheduleParameters Sp1, Stats *St1)
      : InspectorForTileFusedCSRVariableTileSize(Sp1, St1) {}
};

#endif // SPARSE_FUSION_FUSIONINSPECTOR_H

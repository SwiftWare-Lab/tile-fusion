//
// Created by salehm32 on 03/11/23.
//
#include "Stats.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include "sparse-fusion/SparseFusion.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <set>
#ifndef SPARSE_FUSION_FUSIONINSPECTOR_H
#define SPARSE_FUSION_FUSIONINSPECTOR_H

using namespace swiftware::benchmark;
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

class InspectorForSingleLayerTiledFusedCSCParallelWithKTiling
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

class InspectorForSingleLayerTiledFusedCSCCombined
    : public InspectorForSingleLayerTiledFusedCSCParallel {

public:
  virtual sym_lib::MultiDimensionalSet *generateScheduleBasedOnConflictGraphColoring(
      std::map<int, std::vector<int>> ColorToTiles, int NumOfNodes,
      int TileSize, int WorkloadMinSize) {
    std::vector<std::vector<int>> workloads;
    std::set<int> standAloneTiles;
    for (auto tileGroup : ColorToTiles) {
      if (tileGroup.second.size() >= WorkloadMinSize) {
        workloads.push_back(tileGroup.second);
      } else {
        standAloneTiles.insert(tileGroup.second.begin(),
                               tileGroup.second.end());
      }
    }
    std::map<int, int> standAloneTileToNewSize =
        getNewStandAloneTilesMap(standAloneTiles);
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
    for (auto saTile : standAloneTileToNewSize) {
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

  std::map<int, int> getNewStandAloneTilesMap(std::set<int> StandAloneTiles) {
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
    int numOfKTiles = OutputSize/KTileSize;
    for (auto tileGroup : ColorToTiles) {
      if (tileGroup.second.size() >= WorkloadMinSize) {
        workloads.push_back(tileGroup.second);
      } else {
        standAloneTiles.insert(tileGroup.second.begin(),
                               tileGroup.second.end());
      }
    }
    std::map<int, int> standAloneTileToNewSize =
        getNewStandAloneTilesMap(standAloneTiles);
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
    fusedCompSet->id_ = new int[tileToNewSize.size()*numOfKTiles];
    fusedCompSet->type_ = tilePtr;
    fusedCompSet->ptr1_[0] = 0;
    for (i = 0; i < workloads.size(); i++) {
      fusedCompSet->ptr1_[i + 1] = fusedCompSet->ptr1_[i] + workloads[i].size()*numOfKTiles;
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

#endif // SPARSE_FUSION_FUSIONINSPECTOR_H

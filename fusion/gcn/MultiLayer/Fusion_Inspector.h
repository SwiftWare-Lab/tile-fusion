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
  generateFusedScheduleForSingleLayerTiledFusedCSCParallel(sym_lib::CSC *AdjMtx,
                                                           int TileSize) {
    int numOfTiles = (int)ceil((double)AdjMtx->m / TileSize);
    std::map<std::string, std::vector<std::string>> conflictGraph =
        createTilesConflictGraph(AdjMtx, TileSize);
    std::map<std::string, int> coloring = dsaturColoring(conflictGraph);
    std::map<int, std::vector<int>> colorToTiles = getColorToTilesMap(coloring);
    sym_lib::MultiDimensionalSet *fusedCompSet =
        new sym_lib::MultiDimensionalSet();
    fusedCompSet->n1_ = colorToTiles.rbegin()->first + 1;
    fusedCompSet->ptr1_ = new int[fusedCompSet->n1_ + 1];
    fusedCompSet->ptr1_[0] = 0;
    fusedCompSet->id_ = new int[numOfTiles];
    for (std::map<int, std::vector<int>>::iterator it = colorToTiles.begin();
         it != colorToTiles.end(); ++it) {
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
    if (AdjMtx->m % TileSize == 0) {
      fusedCompSet->type_[numOfTiles - 1] = TileSize;
    } else {
      fusedCompSet->type_[numOfTiles - 1] = AdjMtx->m % TileSize;
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

  std::map<int, std::vector<int>>
  getColorToTilesMap(std::map<std::string, int> &coloring) const {
    std::map<int, std::vector<int>> colorToTiles;
    for (std::map<std::string, int>::iterator it = coloring.begin();
         it != coloring.end(); ++it) {
      if (colorToTiles.find(it->second) == colorToTiles.end()) {
        colorToTiles[it->second] = std::vector<int>();
      }
      colorToTiles[it->second].push_back(std::stoi(it->first));
    }
    return colorToTiles;
  }

  void greedyColoring(std::map<std::string, std::vector<std::string>> Graph) {
    int numOfNodes = Graph.size();
    int result[numOfNodes];
    bool available[numOfNodes];
    for (int i = 0; i < numOfNodes; i++) {
      result[i] = -1;
      available[i] = true;
    }
    result[0] = 0;
    for (int i = 1; i < numOfNodes; i++) {
      std::vector<std::string> adjList = Graph[std::to_string(i)];
      for (int j = 0; j < adjList.size(); j++) {
        if (result[std::stoi(adjList[j])] != -1) {
          available[result[std::stoi(adjList[j])]] = false;
        }
      }
      for (int j = 0; j < numOfNodes; j++) {
        if (available[j]) {
          result[i] = j;
          break;
        }
      }
      for (int j = 0; j < numOfNodes; j++) {
        available[j] = true;
      }
    }
    int maxNum = 0;
    for (int i = 0; i < numOfNodes; i++) {
      if (result[i] > maxNum)
        maxNum = result[i];
    }
  }

  std::map<std::string, int>
  dsaturColoring(std::map<std::string, std::vector<std::string>> Graph) {
    std::map<std::string, int> graphColors;
    if (Graph.size() == 0) {
      return std::map<std::string, int>();
    }

    std::vector<std::string> todo;
    std::string maxDegree = "";
    int degree = -1;

    // find maximal degree vertex to color first and color with 0
    for (std::map<std::string, std::vector<std::string>>::iterator i =
             Graph.begin();
         i != Graph.end(); i++) {
      if ((int)i->second.size() > degree) {
        degree = i->second.size();
        maxDegree = i->first;
      }
    }
    if (maxDegree == "") {
      graphColors = std::map<std::string, int>();
      return std::map<std::string, int>();
    }
    graphColors[maxDegree] = 0;

    // Create saturation_level so that we can see which graph nodes have the
    // highest saturation without having to scan through the entire graph
    // each time
    std::map<std::string, int> saturationLevel;

    // Add all nodes and set their saturation level to 0
    for (std::map<std::string, std::vector<std::string>>::iterator i =
             Graph.begin();
         i != Graph.end(); i++) {
      saturationLevel[i->first] = 0;
    }

    // For the single node that has been colored, increment its neighbors so
    // that their current saturation level is correct
    for (int i = 0; i < Graph[maxDegree].size(); i++) {
      saturationLevel[Graph[maxDegree][i]] += 1;
    }

    // Set the saturation level of the already completed node to -infinity so
    // that it is not chosen and recolored
    saturationLevel[maxDegree] = INT_MIN;

    // Populate the todo list with the rest of the vertices that need to be
    // colored
    for (std::map<std::string, std::vector<std::string>>::iterator i =
             Graph.begin();
         i != Graph.end(); i++) {
      if (i->first != maxDegree) {
        graphColors[i->first] = -1;
        todo.push_back(i->first);
      }
    }

    // Color all the remaining nodes in the todo list
    while (!todo.empty()) {
      int saturation = -1;
      std::string saturationName = "";
      std::vector<int> saturationColors;
      // Find the vertex with the highest saturation level, since we keep the
      // saturation levels along the way we can do this in a single pass
      for (std::map<std::string, int>::iterator i = saturationLevel.begin();
           i != saturationLevel.end(); i++) {
        // Find the highest saturated node and keep its name and neighbors
        // colors
        if (i->second > saturation) {
          saturation = i->second;
          saturationName = i->first;

          // Since we're in this loop it means we've found a new most saturated
          // node, which means we need to clear the old list of neighbors colors
          // and replace it with the new highest saturated nodes neighbors
          // colors Since uncolored nodes are given a -1, we can add all
          // neighbors and start the check for lowest available color at greater
          // than 0
          saturationColors.clear();
          for (int j = 0; j < Graph[i->first].size(); j++) {
            saturationColors.push_back(graphColors[Graph[i->first][j]]);
          }
        }
      }
      if (saturationName == "") {
        graphColors = std::map<std::string, int>();
        return graphColors;
      }

      // We now know the most saturated node, so we remove it from the todo list
      for (std::vector<std::string>::iterator itr = todo.begin();
           itr != todo.end(); itr++) {
        if ((*itr) == saturationName) {
          todo.erase(itr);
          break;
        }
      }

      // Find the lowest color that is not being used by any of the most
      // saturated nodes neighbors, then color the most saturated node
      int lowest_color = 0;
      int done = 0;
      while (!done) {
        done = 1;
        for (unsigned i = 0; i < saturationColors.size(); i++) {
          if (saturationColors[i] == lowest_color) {
            lowest_color += 1;
            done = 0;
          }
        }
      }
      graphColors[saturationName] = lowest_color;

      // Since we have colored another node, that nodes neighbors have now
      // become more saturated, so we increase each ones saturation level
      // However we first check that that node has not already been colored
      //(This check is only necessary for enormeous test cases, but is
      // included here for robustness)
      for (int i = 0; i < Graph[saturationName].size(); i++) {
        if (saturationLevel[Graph[saturationName][i]] != INT_MIN) {
          saturationLevel[Graph[saturationName][i]] += 1;
        }
      }
      saturationLevel[saturationName] = INT_MIN;
    }
    return graphColors;
  }

  std::map<std::string, std::vector<std::string>>
  createTilesConflictGraph(sym_lib::CSC *AdjMtx, int TileSize) {
    std::map<std::string, std::vector<std::string>> conflictGraph;
    int numOfTiles = (int)ceil((double)AdjMtx->m / TileSize);
    for (int i = 0; i < numOfTiles; i++) {
      int iStart = i * TileSize;
      int iEnd = std::min(iStart + TileSize, (int)AdjMtx->m);
      int aSize = AdjMtx->p[iEnd] - AdjMtx->p[iStart];
      int *a = new int[aSize];
      std::string iStr = std::to_string(i);
      if (conflictGraph.find(iStr) == conflictGraph.end()) {
        conflictGraph[iStr] = std::vector<std::string>();
      }
      for (int j = i + 1; j < numOfTiles; j++) {
        int jStart = j * TileSize;
        int jEnd = std::min(jStart + TileSize, (int)AdjMtx->m);
        int bSize = AdjMtx->p[jEnd] - AdjMtx->p[jStart];
        int *b = new int[bSize];
        std::memcpy(a, AdjMtx->i + AdjMtx->p[iStart], aSize * sizeof(int));
        std::memcpy(b, AdjMtx->i + AdjMtx->p[jStart], bSize * sizeof(int));
        if (checkIfTwoArraysHasSameValue(a, b, aSize, bSize)) {
          std::string jStr = std::to_string(j);
          if (conflictGraph.find(jStr) == conflictGraph.end()) {
            conflictGraph[jStr] = std::vector<std::string>();
          }
          conflictGraph[iStr].push_back(jStr);
          conflictGraph[jStr].push_back(iStr);
        }
        delete[] b;
      }
      delete[] a;
    }
    return conflictGraph;
  }

  bool checkIfTwoArraysHasSameValue(int *A, int *B, int ASize, int BSize) {
    std::sort(A, A + ASize);
    std::sort(B, B + BSize);
    int i = 0;
    int j = 0;
    while (i < ASize && j < BSize) {
      if (A[i] == B[j]) {
        return true;
      }
      if (A[i] < B[j]) {
        i++;
      } else {
        j++;
      }
    }
    return false;
  }
};

class InspectorForSingleLayerTiledFusedCSCCombined
    : public InspectorForSingleLayerTiledFusedCSCParallel {

public:
  sym_lib::MultiDimensionalSet *
  generateScheduleForSingleLayerTiledFusedCSCCombined(sym_lib::CSC *AdjMtx,
                                                      int TileSize) {
    int numOfTiles = (int)ceil((double)AdjMtx->m / TileSize);
    std::map<std::string, std::vector<std::string>> conflictGraph =
        createTilesConflictGraph(AdjMtx, TileSize);
    std::map<std::string, int> coloring = dsaturColoring(conflictGraph);
    std::map<int, std::vector<int>> colorToTiles = getColorToTilesMap(coloring);
    return generateScheduleBasedOnConflictGraphColoring(colorToTiles, 10,
                                                        TileSize, AdjMtx->m);
  }

private:

  sym_lib::MultiDimensionalSet *  generateScheduleBasedOnConflictGraphColoring(
      std::map<int, std::vector<int>> ColorToTiles, int WorkloadMinSize,
      int TileSize, int NumOfNodes) {
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
      if(newTileSize > maxTileSize){
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

  std::map<int, int>
  updateNewTilesMapWithWorkloadTiles(std::vector<std::vector<int>> Workloads,
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
  getNewStandAloneTilesMap(std::set<int> StandAloneTiles) {
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

#endif // SPARSE_FUSION_FUSIONINSPECTOR_H

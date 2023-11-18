//
// Created by mehdi on 11/15/23.
//

#include "aggregation/def.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>
#ifndef SPARSE_FUSION_GRAPHCOLORING_H
#define SPARSE_FUSION_GRAPHCOLORING_H

#endif // SPARSE_FUSION_GRAPHCOLORING_H

class DsaturColoringForConflictGraph {

public:
  std::map<int, std::vector<int>>
  generateGraphColoringForConflictGraphOf(sym_lib::CSC *AdjMtx, int TileSize) {
    std::map<std::string, std::vector<std::string>> conflictGraph =
        createTilesConflictGraph(AdjMtx, TileSize);
    std::map<std::string, int> coloring = dsaturColoring(conflictGraph);
    std::map<int, std::vector<int>> colorToTiles = getColorToTilesMap(coloring);
    return colorToTiles;
  }

protected:
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

  std::map<int, std::vector<int>>
  getColorToTilesMap(std::map<std::string, int> &coloring) {
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

class DsaturColoringForConflictGraphWithKTiling : public DsaturColoringForConflictGraph{
public:
  std::map<int, std::vector<int>>
  generateGraphColoringForConflictGraphOf(sym_lib::CSC *AdjMtx, int TileSize, int OutputSize, int KTileSize) {
    std::map<std::string, std::vector<std::string>> conflictGraph =
        createTilesConflictGraphWithKTiling(AdjMtx, TileSize, OutputSize, KTileSize);
    std::map<std::string, int> coloring = dsaturColoring(conflictGraph);
    std::map<int, std::vector<int>> colorToTiles = getColorToTilesMap(coloring);
    return colorToTiles;
  }

protected:
  std::map<std::string, std::vector<std::string>>
  createTilesConflictGraphWithKTiling(sym_lib::CSC *AdjMtx, int TileSize, int OutputSize, int KTileSize) {
    std::map<std::string, std::vector<std::string>> conflictGraph;
    int numOfTiles = (int)ceil((double)AdjMtx->m / TileSize);
    int numOfKTiles = OutputSize/ KTileSize;
    for (int i = 0; i < numOfTiles; i++) {
      int iStart = i * TileSize;
      int iEnd = std::min(iStart + TileSize, (int)AdjMtx->m);
      int aSize = AdjMtx->p[iEnd] - AdjMtx->p[iStart];
      int *a = new int[aSize];
      std::string iStr = std::to_string(i*numOfKTiles);
      if (conflictGraph.find(iStr) == conflictGraph.end()) {
        conflictGraph[iStr] = std::vector<std::string>();
      }
      for (int k = 1; k < numOfKTiles; k++){
        std::string ikStr = std::to_string(i*numOfKTiles+k);
        if (conflictGraph.find(ikStr) == conflictGraph.end()) {
          conflictGraph[ikStr] = std::vector<std::string>();
        }
      }
      for (int j = i + 1; j < numOfTiles; j++) {
        int jStart = j * TileSize;
        int jEnd = std::min(jStart + TileSize, (int)AdjMtx->m);
        int bSize = AdjMtx->p[jEnd] - AdjMtx->p[jStart];
        int *b = new int[bSize];
        std::memcpy(a, AdjMtx->i + AdjMtx->p[iStart], aSize * sizeof(int));
        std::memcpy(b, AdjMtx->i + AdjMtx->p[jStart], bSize * sizeof(int));
        if (checkIfTwoArraysHasSameValue(a, b, aSize, bSize)) {
          std::string jStr = std::to_string(j*numOfKTiles);
          if (conflictGraph.find(jStr) == conflictGraph.end()) {
            conflictGraph[jStr] = std::vector<std::string>();
          }
          conflictGraph[iStr].push_back(jStr);
          conflictGraph[jStr].push_back(iStr);
          for (int k = 1; k < numOfKTiles; k++){
            std::string ikStr = std::to_string(i*numOfKTiles+k);
            std::string jkStr = std::to_string(j*numOfKTiles+k);
            if (conflictGraph.find(jkStr) == conflictGraph.end()) {
              conflictGraph[jkStr] = std::vector<std::string>();
            }
            conflictGraph[ikStr].push_back(jkStr);
            conflictGraph[jkStr].push_back(ikStr);
          }
        }
        delete[] b;
      }
      delete[] a;
    }
    return conflictGraph;
  }

};
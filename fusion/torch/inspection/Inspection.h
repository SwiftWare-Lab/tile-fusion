//
// Created by salehm32 on 05/01/24.
//
#include <algorithm>
#include <climits>
#include <cmath>
#include <set>
#include <map>
#ifndef FUSED_GCN_INSPECTION_H
#define FUSED_GCN_INSPECTION_H
#define NULLPNTR nullptr

struct MultiDimensionalSet{
    int n1_{}, n2_{}, n3_{};
    int d_{};
    int *ptr1_{}, *ptr2_{};
    int *id_{}, *type_{};
    int *w_par_type_{}; // the type of each w-partition
    int *ker_begin_{}; // ker_begin[i][j], starting index of kernel j in w-partition i

    // redundancy mode
    bool *is_redundancy_{};
    int *map_redundancy_{};

    MultiDimensionalSet() : ptr1_(NULLPNTR), ptr2_(NULLPNTR), id_(NULLPNTR),
                            type_(NULLPNTR), w_par_type_(NULLPNTR), ker_begin_(NULLPNTR),
                            is_redundancy_(NULLPNTR), map_redundancy_(NULLPNTR){};

    MultiDimensionalSet(const int n_hl, int *hl_ptr, int *par_ptr,
                        int *partition) {
        n2_ = 0;
        type_ = NULLPNTR;
        w_par_type_ = NULLPNTR;
        ker_begin_ = NULLPNTR;
        is_redundancy_ = NULLPNTR;
        map_redundancy_ = NULLPNTR;
    }

    explicit MultiDimensionalSet(int N) {
        n1_ = n2_ = N;
        ptr2_ = NULLPNTR;
        ptr1_ = new int[n1_ + 1]();
        id_ = new int[n1_];
        type_ = new int[n1_];
        w_par_type_ = NULLPNTR;
        ker_begin_ = NULLPNTR;
        is_redundancy_ = NULLPNTR;
        map_redundancy_ = NULLPNTR;
    }

    /*
     * For 2D
     */
    MultiDimensionalSet(int N1, int N3) {
        n1_ =  N1;
        n2_ = 0;
        n3_ = N3;
        ptr2_ = NULLPNTR;
        ptr1_ = new int[n1_ + 1]();
        id_ = new int[n3_];
        type_= new int[n3_];
        w_par_type_ = NULLPNTR;
        ker_begin_ = NULLPNTR;
        is_redundancy_=NULLPNTR;
        map_redundancy_=NULLPNTR;
    }

    /*
     * for 3D
     */
    MultiDimensionalSet(int N1, int N2, int N3) :MultiDimensionalSet() {
        n1_ = N1;
        n2_ = N2;
        n3_ = N3;
        ptr1_ = new int[n1_ + 1]();
        ptr2_ = new int[n2_ + 1]();
        id_ = new int[n3_];
        type_ = new int[n3_];
        w_par_type_ = NULLPNTR;
        ker_begin_ = NULLPNTR;
        is_redundancy_ = NULLPNTR;
        map_redundancy_ = NULLPNTR;
    }

/*
 * for 3D with depth
 */
    MultiDimensionalSet(int N1, int N2, int N3, int D, int Red) {
        n1_ = N1;
        n2_ = N2;
        n3_ = N3;
        d_ = D;
        ptr1_ = new int[n1_ + 1]();
        ptr2_ = new int[n2_ + 1]();
        id_ = new int[n3_];
        type_= new int[n3_];
        w_par_type_ = new int[n1_*n2_]();
        ker_begin_ = new int[n2_*d_]();
        is_redundancy_ = new bool[d_]();
        map_redundancy_ = new int[Red]();
    }

//    MultiDimensionalSet(
//            const std::vector<std::vector<FusedNode*>> &FusedSchedule);
//    MultiDimensionalSet(
//            const std::vector<std::vector<FusedNode*>> &FusedSchedule,
//            int PerPartition);

    ~MultiDimensionalSet() {
        delete []ptr1_;
        delete []ptr2_;
        delete []id_;
        delete []type_;
        delete []w_par_type_;
        delete []ker_begin_;
        delete []is_redundancy_;
        delete []map_redundancy_;
    }

    int getNumberOfFusedNodes() {
        int fusedCounter = 0;
        for (int i = 0; i < n1_; ++i) {
            for (int j = ptr1_[i]; j < ptr1_[i + 1]; ++j) {
                for (int k = ptr2_[j]; k < ptr2_[j + 1]; ++k) {
                    if (i == 0 && type_[k] == 1) {
                        fusedCounter += 1;
                    }
                }
            }
        }
        return fusedCounter;
    }
};

// intera-layer fusion scheduler for tiledFusedCSCParallel that uses coloring of
// conflict graph to generate create wave fronts.
class InspectorForSingleLayerTiledFusedCSCParallel {
public:
    MultiDimensionalSet *
    generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            std::map<int, std::vector<int>> ColorToTiles, int NumOfNodes,
            int TileSize) {
        int numOfTiles = (int)ceil((double)NumOfNodes / TileSize);
        MultiDimensionalSet *fusedCompSet =
                new MultiDimensionalSet();
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

class InspectorForSingleLayerTiledFusedCSCCombined
        : public InspectorForSingleLayerTiledFusedCSCParallel {

public:
    virtual MultiDimensionalSet *
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
        MultiDimensionalSet *fusedCompSet =
                new MultiDimensionalSet();
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
#endif //FUSED_GCN_INSPECTION_H

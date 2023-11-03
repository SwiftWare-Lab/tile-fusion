//
// Created by salehm32 on 03/11/23.
//
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include "sparse-fusion/SparseFusion.h"
#include "Stats.h"
#ifndef SPARSE_FUSION_FUSIONINSPECTOR_H
#define SPARSE_FUSION_FUSIONINSPECTOR_H

using namespace swiftware::benchmark;
class InspectorForAllFused {
protected:
  sym_lib::ScheduleParameters Sp;
  Stats *St;

public:
  InspectorForAllFused(sym_lib::ScheduleParameters Sp1, Stats *St1) : Sp(Sp1), St(St1) {}

  sym_lib::MultiDimensionalSet* generateFusedScheduleForAllFused(sym_lib::CSR *AdjMtx) {
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
    sym_lib::MultiDimensionalSet *fusedCompSet = sf01->getFusedCompressed((int)pt[0]);
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
  InspectorForAllTiledFusedCSC() {}
  ~InspectorForAllTiledFusedCSC() {}

  sym_lib::MultiDimensionalSet *
  generateFusedScheduleForAllTiledFusedCSC(sym_lib::CSR *AdjMtx, int TileSize) {
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
        AdjMtx->m, numOfTiles, TileSize, AdjMtx, rowsLastTile, tileLastNeededTile);
    // Generating fused schedule based on output of previous two steps
    generateFusedSchedule(AdjMtx->m, numOfTiles, TileSize, tileLastNeededTile, fusedCompSet);
    return fusedCompSet;
  }

private:
  void findRowsLastNonzeroTile(int NumOfNodes, int TileSize, const sym_lib::CSR *AdjMtx,
                               int *RowsLastTile) {
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
      int NumOfNodes, int NumOfTiles , int TileSize, const sym_lib::CSR *AdjMtx, const int *RowsLastTile,
      int *TileLastNeededTile) {
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

#endif // SPARSE_FUSION_FUSIONINSPECTOR_H

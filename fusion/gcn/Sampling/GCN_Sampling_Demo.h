//
// Created by salehm32 on 03/11/23.
//
#include "../MultiLayer/GCN_Multi_Layer_Demo_Utils.h"
#ifndef SPARSE_FUSION_GCN_SAMPLING_DEMO_H
#define SPARSE_FUSION_GCN_SAMPLING_DEMO_H

class GCNFusedParallelWithOmittingEmptyRows : public GCNIntraFusedSequential {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;

  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet = generateSimpleFusedSchedule(InTensor->NumThreads);
    t.stop();
    return t;
  }

  sym_lib::MultiDimensionalSet *generateSimpleFusedSchedule(int NumOfThreads) {
    sym_lib::MultiDimensionalSet *fusedSchedule =
        new sym_lib::MultiDimensionalSet();
    fusedSchedule->n1_ = 2;
    fusedSchedule->ptr1_ = new int[3];
    fusedSchedule->ptr2_ = new int[2 * NumOfThreads + 1];
    fusedSchedule->id_ = new int[InTensor->LayerMasks[0].size() +
                                 InTensor->LayerMasks[1].size()];
    fusedSchedule->type_ = new int[InTensor->LayerMasks[0].size() +
                                   InTensor->LayerMasks[1].size()];
    fusedSchedule->ptr1_[0] = 0;
    fusedSchedule->ptr1_[1] = NumOfThreads;
    fusedSchedule->ptr1_[2] = 2 * NumOfThreads;
    fusedSchedule->ptr2_[0] = 0;
    sym_lib::CSR *l1 = InTensor->LayerMaskedMatrices[0];
    sym_lib::CSR *l2 = InTensor->LayerMaskedMatrices[1];
    int iterPerPartitionL1 =
        std::ceil(float(InTensor->LayerMasks[0].size()) / NumOfThreads);
    std::vector<std::set<int>> l1Partitions(NumOfThreads);
    int partitionCntr = 0;
    int p = 0;
    int idCounter = 0;
    std::set<int> fusedNodes;
    for (int i = 0; i < l1->m; i++) {
      if (l1->p[i + 1] == l1->p[i]) {
        continue;
      }
      l1Partitions[p].insert(i);
      fusedSchedule->id_[idCounter] = i;
      fusedSchedule->type_[idCounter] = 0;
      idCounter++;
      partitionCntr++;
      if (partitionCntr == iterPerPartitionL1 || i == l1->m - 1) {
        partitionCntr = 0;
        for (int i1 = 0; i1 <= l2->m; i1++) {
          if (l2->p[i1 + 1] == l2->p[i1]) {
            continue;
          }
          bool flag = true;
          for (int j1 = l2->p[i1]; j1 < l2->p[i1 + 1]; j1++) {
            if (l1Partitions[p].find(l2->i[j1]) == l1Partitions[p].end()) {
              flag = false;
              break;
            }
          }
          if (flag && fusedNodes.find(i1) == fusedNodes.end()) {
            fusedSchedule->id_[idCounter] = i1;
            fusedSchedule->type_[idCounter] = 1;
            idCounter++;
            fusedNodes.insert(i1);
          }
        }
        fusedSchedule->ptr2_[p + 1] = idCounter;
        p++;
      }
    }
    //    fusedSchedule->ptr2_[NumOfThreads] = idCounter;
    int unfusedNum = InTensor->LayerMasks[1].size() - fusedNodes.size();
    partitionCntr = 0;
    p = 0;
    int iterPerPartitionL2 = ceil(float(unfusedNum) / NumOfThreads);
    for (int i = 0; i < l2->m; i++) {
      if (l2->p[i + 1] == l2->p[i]) {
        continue;
      }
      if (fusedNodes.find(i) != fusedNodes.end()) {
        continue;
      }
      fusedSchedule->id_[idCounter] = i;
      fusedSchedule->type_[idCounter] = 1;
      idCounter++;
      partitionCntr++;
      if (partitionCntr == iterPerPartitionL2) {
        partitionCntr = 0;
        fusedSchedule->ptr2_[p + 1 + NumOfThreads] = idCounter;
        p++;
      }
    }
    fusedSchedule->ptr2_[2 * NumOfThreads] = idCounter;
    return fusedSchedule;
  }

  Timer execute() override {
    Timer t;
    OutTensor->reset();
    t.start();
    forwardForFusedLayersParallelWithBatching(
        InTensor->LayerMaskedMatrices[0]->m,
        InTensor->LayerMaskedMatrices[0]->p,
        InTensor->LayerMaskedMatrices[0]->i,
        InTensor->LayerMaskedMatrices[0]->x,
        InTensor->LayerMaskedMatrices[1]->p,
        InTensor->LayerMaskedMatrices[1]->i,
        InTensor->LayerMaskedMatrices[1]->x, InTensor->FeatureMatrix->col,
        InTensor->EmbedDim, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1, InTensor->Weight2,
        OutTensor->SecondLayerOutput, OutTensor->FirstLayerOutput,
        InTensor->NumThreads, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_);
    t.stop();
    return t;
  }

public:
  GCNFusedParallelWithOmittingEmptyRows(GnnTensorInputs *In1, Stats *Stat1,
                                        sym_lib::ScheduleParameters SpIn)
      : GCNIntraFusedSequential(In1, Stat1), Sp(SpIn) {}
  ~GCNFusedParallelWithOmittingEmptyRows() { delete FusedCompSet; }
};

class GCNFusedWithOmittingEmptyRows : public GCNIntraFusedSequential {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  int TileSize;

  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet = generateSimpleFusedSchedule();
    t.stop();
    return t;
  }

  sym_lib::MultiDimensionalSet *generateSimpleFusedSchedule() {
    sym_lib::MultiDimensionalSet *fusedSchedule =
        new sym_lib::MultiDimensionalSet();
    fusedSchedule->n1_ = 2;
    fusedSchedule->ptr1_ = new int[3];
    fusedSchedule->ptr2_ = new int[3];
    int allNodesNum =
        InTensor->LayerMasks[0].size() + InTensor->LayerMasks[1].size();
    fusedSchedule->id_ = new int[allNodesNum];
    fusedSchedule->type_ = new int[allNodesNum];
    fusedSchedule->ptr1_[0] = 0;
    fusedSchedule->ptr1_[1] = 1;
    fusedSchedule->ptr1_[2] = 2;
    fusedSchedule->ptr2_[0] = 0;
    sym_lib::CSR *l1 = InTensor->LayerMaskedMatrices[0];
    sym_lib::CSR *l2 = InTensor->LayerMaskedMatrices[1];
    std::set<int> fusedNodes;
    int idCounter = 0;
    for (int i = 0; i < l1->m; i += TileSize) {
      for (int j = i; j < i + TileSize; j++) {
        if (j >= l1->m)
          break;
        if (l1->p[j + 1] == l1->p[j])
          continue;
        fusedSchedule->id_[idCounter] = j;
        fusedSchedule->type_[idCounter] = 0;
        idCounter++;
      }
      for (int i1 = 0; i1 < l2->m; i1++) {
        if (l2->p[i1] == l2->p[i1 + 1])
          continue;
        bool flag = true;
        for (int j1 = l2->p[i1]; j1 < l2->p[i1 + 1]; j1++) {
          int neigh = l2->i[j1];
          if (neigh >= i + TileSize || neigh < i) {
            flag = false;
            break;
          }
        }
        if (flag && fusedNodes.find(i1) == fusedNodes.end()) {
          fusedSchedule->id_[idCounter] = i1;
          fusedSchedule->type_[idCounter] = 1;
          fusedNodes.insert(i1);
          idCounter++;
        }
      }
    }
    fusedSchedule->ptr2_[1] = idCounter;
    for (int i = 0; i < l2->m; i++) {
      if (l2->p[i] == l2->p[i + 1] || fusedNodes.find(i) != fusedNodes.end()) {
        continue;
      }
      fusedSchedule->id_[idCounter] = i;
      fusedSchedule->type_[idCounter] = 1;
      idCounter++;
    }
    fusedSchedule->ptr2_[2] = idCounter;
    assert(idCounter == allNodesNum);
    this->St->OtherStats["Number of Fused Nodes"] = {double(fusedNodes.size())};
    return fusedSchedule;
  }

  Timer execute() override {
    Timer t;
    OutTensor->reset();
    t.start();
    forwardForFusedLayersWithBatching(
        InTensor->LayerMaskedMatrices[0]->m,
        InTensor->LayerMaskedMatrices[0]->p,
        InTensor->LayerMaskedMatrices[0]->i,
        InTensor->LayerMaskedMatrices[0]->x,
        InTensor->LayerMaskedMatrices[1]->p,
        InTensor->LayerMaskedMatrices[1]->i,
        InTensor->LayerMaskedMatrices[1]->x, InTensor->FeatureMatrix->col,
        InTensor->EmbedDim, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1, InTensor->Weight2,
        OutTensor->SecondLayerOutput, OutTensor->FirstLayerOutput,
        InTensor->NumThreads, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_);
    t.stop();
    return t;
  }

public:
  GCNFusedWithOmittingEmptyRows(GnnTensorInputs *In1, Stats *Stat1,
                                sym_lib::ScheduleParameters SpIn, int TileSize1)
      : GCNIntraFusedSequential(In1, Stat1), Sp(SpIn), TileSize(TileSize1) {}
  ~GCNFusedWithOmittingEmptyRows() { delete FusedCompSet; }
};

class GCNFusedWithRegisterReuse : public GCNIntraFusedSequential {
protected:
  int TileSize;
  Timer execute() override {
    Timer t;
    OutTensor->reset();
    t.start();
    forwardForFusedLayersWithBatchingRegisterReuse(
        InTensor->LayerMaskedMatrices[0]->m,
        InTensor->LayerMaskedMatrices[0]->p,
        InTensor->LayerMaskedMatrices[0]->i,
        InTensor->LayerMaskedMatrices[0]->x,
        InTensor->LayerMaskedMatrices[1]->p,
        InTensor->LayerMaskedMatrices[1]->i,
        InTensor->LayerMaskedMatrices[1]->x, InTensor->FeatureMatrix->col,
        InTensor->EmbedDim, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1, InTensor->Weight2,
        OutTensor->SecondLayerOutput, OutTensor->FirstLayerOutput, TileSize);
    t.stop();
    return t;
  }

public:
  GCNFusedWithRegisterReuse(GnnTensorInputs *In1, Stats *Stat1, int TileSize1)
      : GCNIntraFusedSequential(In1, Stat1), TileSize(TileSize1) {}
};

#endif // SPARSE_FUSION_GCN_SAMPLING_DEMO_H

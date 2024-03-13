//
// Created by salehm32 on 20/10/23.
//

#ifdef MKL
#include "../GCN_Layer_MKL_Forward_Utils.h"
#else
#include "../GCN_Layer_Forward_Utils.h"
#endif
#include "sparse-fusion/Fusion_Inspector.h"
#include "../MultiLayer/GCN_Multi_Layer_Demo_Utils.h"
#include "SWTensorBench.h"
#ifndef SPARSE_FUSION_GCN_SINGLE_LAYER_DEMO_UTILS_H
#define SPARSE_FUSION_GCN_SINGLE_LAYER_DEMO_UTILS_H

#endif // SPARSE_FUSION_GCN_SINGLE_LAYER_DEMO_UTILS_H

using namespace swiftware::benchmark;

class GCNSingleLayerFused : public GCNIntraFusedSequential {
protected:
  bool verify(double &Error) override {
    bool retValue = true;
    if (In->CorrectSol == nullptr)
      return true;
    double infNorm = 0;
    for (int i = 0; i < InTensor->NumOfNodes * InTensor->Weight1->row; ++i) {
      if (std::abs(OutTensor->FirstLayerOutput[i] - In->CorrectSol[i]) >
          infNorm) {
        infNorm = std::abs(OutTensor->FirstLayerOutput[i] - In->CorrectSol[i]);
      }
    }
    Error = (double)infNorm;
    if (infNorm > In->Threshold) {
      retValue = false;
    }
    return retValue;
  }

  Timer execute() override {
    OutTensor->reset();
    set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayer(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->row, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerFused(GnnTensorInputs *In1, Stats *Stat1)
      : GCNIntraFusedSequential(In1, Stat1) {}
};

class GCNSingleLayerFusedParallel : public GCNSingleLayerFused {
protected:
  Timer execute() override {
    OutTensor->reset();
    Timer t;
    set_num_threads(1);
    t.start();
    forwardForOneLayerParallel(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->row, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerFusedParallel(GnnTensorInputs *In1, Stats *Stat1)
      : GCNSingleLayerFused(In1, Stat1) {}
};

#ifdef MKL
class GCNSingleLayerMKL : public GCNSingleLayerFused {

protected:
  sparse_matrix_t MKLAdj;
  Timer execute() override {
    OutTensor->reset();
    mkl_set_num_threads(InTensor->NumThreads);
    Timer t;
    double *intermediateResult = new double[InTensor->NumOfNodes * InTensor->Weight1->row]{};
    t.start();
    forwardForOneLayerWithMKLGeMMAndMKLSpMM(
        InTensor->NumOfNodes, MKLAdj, InTensor->FeatureMatrix->a,
        InTensor->FeatureMatrix->col, InTensor->Weight1->a,
        InTensor->Weight1->row, OutTensor->FirstLayerOutput, intermediateResult);
    t.stop();
    delete[] intermediateResult;
    return t;
  }

public:
  GCNSingleLayerMKL(GnnTensorInputs *In1, Stats *Stat1)
      : GCNSingleLayerFused(In1, Stat1) {
    mkl_sparse_d_create_csr(
        &MKLAdj, SPARSE_INDEX_BASE_ZERO, this->InTensor->NumOfNodes,
        this->InTensor->NumOfNodes, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->p + 1, this->InTensor->AdjacencyMatrix->i,
        this->InTensor->AdjacencyMatrix->x);
  }
  ~GCNSingleLayerMKL() { mkl_free(MKLAdj); }
};

#endif

class GCNSingleLayerUnFusedCSRMKLGeMM : public GCNSingleLayerFused {

protected:
  Timer execute() override {
    OutTensor->reset();
    double *intermediateResult = new double[InTensor->NumOfNodes * InTensor->Weight1->row]{};
    set_num_threads(InTensor->NumThreads);
    Timer t;
    t.start();
    forwardForOneLayerWithMKLGeMMAndSpMM(
        InTensor->NumOfNodes, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->a, InTensor->FeatureMatrix->col,
        InTensor->Weight1->a, InTensor->Weight1->row,
        OutTensor->FirstLayerOutput, intermediateResult, InTensor->NumThreads);
    t.stop();
    delete[] intermediateResult;
    return t;
  }

public:
  GCNSingleLayerUnFusedCSRMKLGeMM(GnnTensorInputs *In1, Stats *Stat1)
      : GCNSingleLayerFused(In1, Stat1) {}
};

class GCNSingleLayerUnfusedCSC : public GCNSingleLayerFused {

protected:
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(InTensor->NumThreads);
    Timer t;
    t.start();
    forwardForOneLayerUnfusedCSC(
        InTensor->NumOfNodes, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->a, InTensor->FeatureMatrix->col,
        InTensor->Weight1->a, InTensor->Weight1->row,
        OutTensor->FirstLayerOutput);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerUnfusedCSC(GnnTensorInputs *In1, Stats *Stat1)
      : GCNSingleLayerFused(In1, Stat1) {}
};

class GCNSingleLayerTiledFused : public GCNSingleLayerFused {
  int TileSize;
  InspectorForTiledFused *Inspector;
  TiledFusedLayerSchedulingParameters *Sp;

  Timer analysis() override {
    Timer t;
    t.start();
    Sp = Inspector->generateGeMMTileForEachSpMMTile(InTensor->AdjacencyMatrix,
                                                    TileSize);
    t.stop();
    return t;
  }

protected:
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(InTensor->NumThreads);
    Timer t;
    t.start();
    forwardForOneLayerTiled(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->col, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, TileSize, Sp->GeMMLowerBounds,
        Sp->GeMMUpperBounds, Sp->MaxGeMMTileSize);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFused(GnnTensorInputs *In1, Stats *Stat1, int TileSize1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1) {
    Inspector = new InspectorForTiledFused();
  }
  ~GCNSingleLayerTiledFused() {
    delete Sp;
    delete Inspector;
  }
};

class GCNSingleLayerTiledFusedParallel : public GCNSingleLayerFused {
protected:
  int TileSize;
  InspectorForTiledFused *Inspector;
  TiledFusedLayerSchedulingParameters *Sp;

  Timer analysis() override {
    Timer t;
    t.start();
    Sp = Inspector->generateGeMMTileForEachSpMMTile(InTensor->AdjacencyMatrix,
                                                    TileSize);
    t.stop();
    return t;
  }

  Timer execute() override {
    OutTensor->reset();
    set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerTiledParallel(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->col, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, TileSize, InTensor->NumThreads,
        Sp->GeMMLowerBounds, Sp->GeMMUpperBounds, Sp->MaxGeMMTileSize);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedParallel(GnnTensorInputs *In1, Stats *Stat1,
                                   int TileSize1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1) {
    Inspector = new InspectorForTiledFused();
  }

  ~GCNSingleLayerTiledFusedParallel() {
    delete Sp;
    delete Inspector;
  }
};

class GCNSingleLayerTiledFusedCSC : public GCNSingleLayerFused {
protected:
  int TileSize;
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(InTensor->NumThreads);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiled(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->col, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, TileSize);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedCSC(GnnTensorInputs *In1, Stats *Stat1, int TileSize1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1) {}
};

class GCNSingleLayerFusedCSC : public GCNSingleLayerFused {
protected:
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerFromCSC(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->col, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerFusedCSC(GnnTensorInputs *In1, Stats *Stat1)
      : GCNSingleLayerFused(In1, Stat1) {}
};

class GCNSingleLayerTiledFusedCSCParallelAtomic : public GCNSingleLayerFused {
protected:
  int TileSize;
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiledParallel(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->row, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, TileSize, InTensor->NumThreads);
    t.stop();
    return t;
  }
public:
  GCNSingleLayerTiledFusedCSCParallelAtomic(
      GnnTensorInputs *In1, Stats *Stat1, int TileSize1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1)
         {}
};

class GCNSingleLayerTiledFusedCSCParallel : public GCNSingleLayerFused {
protected:
  int TileSize;
  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForSingleLayerTiledFusedCSCParallel *Inspector;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet =
        Inspector->generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            ConflictGraphColoring, InTensor->NumOfNodes, TileSize);
    t.stop();
    return t;
  }
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiledParallelV2(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->row, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, TileSize, InTensor->NumThreads,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        FusedCompSet->type_);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedCSCParallel(
      GnnTensorInputs *In1, Stats *Stat1, int TileSize1,
      std::map<int, std::vector<int>> ConflictGraphColoring1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1),
        ConflictGraphColoring(ConflictGraphColoring1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }
  ~GCNSingleLayerTiledFusedCSCParallel() {
    delete FusedCompSet;
    delete Inspector;
  }
};

class GCNSingleLayerTiledFusedCSCParallelWithKTiling
    : public GCNSingleLayerFused {
protected:
  int TileSize;
  int KTileSize;
  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForSingleLayerTiledFusedCSCParallel *Inspector;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet =
        Inspector->generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            ConflictGraphColoring, InTensor->NumOfNodes, TileSize);
    t.stop();
    return t;
  }
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiledParallelWithKTiling(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->row, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, TileSize, InTensor->NumThreads,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        FusedCompSet->type_, KTileSize);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedCSCParallelWithKTiling(
      GnnTensorInputs *In1, Stats *Stat1, int TileSize1,
      std::map<int, std::vector<int>> ConflictGraphColoring1, int KTileSize1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1),
        ConflictGraphColoring(ConflictGraphColoring1), KTileSize(KTileSize1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }
  ~GCNSingleLayerTiledFusedCSCParallelWithKTiling() {
    delete FusedCompSet;
    delete Inspector;
  }
};

class GCNSingleLayerTiledFusedCSCParallelWithSchedulingKTiling
    : public GCNSingleLayerFused {
protected:
  int TileSize;
  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForSingleLayerTiledFusedCSCParallelWithSchedulingKTiles *Inspector;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  int KTileSize;

  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet = Inspector->generateScheduleBasedOnConflictGraphColoring(
        ConflictGraphColoring, InTensor->NumOfNodes, TileSize,
        InTensor->Weight1->row, KTileSize);
    t.stop();
    return t;
  }
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiledParallelWithSchedulingForKTiling(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->row, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, TileSize, InTensor->NumThreads,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        FusedCompSet->type_, KTileSize);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedCSCParallelWithSchedulingKTiling(
      GnnTensorInputs *In1, Stats *Stat1, int TileSize1,
      std::map<int, std::vector<int>> ConflictGraphColoring1, int KTileSize1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1),
        ConflictGraphColoring(ConflictGraphColoring1), KTileSize(KTileSize1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallelWithSchedulingKTiles();
  }
  ~GCNSingleLayerTiledFusedCSCParallelWithSchedulingKTiling() {
    delete FusedCompSet;
    delete Inspector;
  }
};

class GCNSingleLayerTiledFusedCSCCombined : public GCNSingleLayerFused {
protected:
  int TileSize;
  int WorkloadMinSize;
  std::map<int, std::vector<int>> ConflictGraphColoring;

  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForSingleLayerTiledFusedCSCCombined *Inspector;
  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet = Inspector->generateScheduleBasedOnConflictGraphColoring(
        ConflictGraphColoring, InTensor->NumOfNodes, TileSize, WorkloadMinSize);
    t.stop();
    return t;
  }
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiledParallelCombined(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->row, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, TileSize, FusedCompSet->n3_,
        InTensor->NumThreads, FusedCompSet->n1_, FusedCompSet->n2_,
        FusedCompSet->ptr1_, FusedCompSet->id_, FusedCompSet->type_);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedCSCCombined(
      GnnTensorInputs *In1, Stats *Stat1, int TileSize1, int WorkloadMinSize1,
      std::map<int, std::vector<int>> ConflictGraphColoring1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1),
        ConflictGraphColoring(ConflictGraphColoring1),
        WorkloadMinSize(WorkloadMinSize1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCCombined();
  }
  ~GCNSingleLayerTiledFusedCSCCombined() {
    delete FusedCompSet;
    delete Inspector;
  }
};

class GCNSingleLayerTiledFusedCSCCombinedWithKTiling
    : public GCNSingleLayerFused {
protected:
  int TileSize;
  int WorkloadMinSize;
  int KTileSize;
  std::map<int, std::vector<int>> ConflictGraphColoring;

  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForSingleLayerTiledFusedCSCCombinedWithKTiling *Inspector;
  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet = Inspector->generateScheduleBasedOnConflictGraphColoring(
        ConflictGraphColoring, InTensor->NumOfNodes, TileSize, WorkloadMinSize,
        InTensor->Weight1->row, KTileSize);
    t.stop();
    return t;
  }
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiledParallelCombinedWithKTiling(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->row, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, TileSize, FusedCompSet->n3_,
        InTensor->NumThreads, FusedCompSet->n1_, FusedCompSet->n2_,
        FusedCompSet->ptr1_, FusedCompSet->id_, FusedCompSet->type_, KTileSize);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedCSCCombinedWithKTiling(
      GnnTensorInputs *In1, Stats *Stat1, int TileSize1, int WorkloadMinSize1,
      std::map<int, std::vector<int>> ConflictGraphColoring1, int KTileSize1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1),
        ConflictGraphColoring(ConflictGraphColoring1),
        WorkloadMinSize(WorkloadMinSize1), KTileSize(KTileSize1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCCombinedWithKTiling();
  }
  ~GCNSingleLayerTiledFusedCSCCombinedWithKTiling() {
    delete FusedCompSet;
    delete Inspector;
  }
};

class GCNSingleLayerSparseFusedParallel : public GCNSingleLayerFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForAllFused *Inspector;

  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet =
        Inspector->generateFusedScheduleForAllFused(InTensor->AdjacencyMatrix);
    t.stop();
    return t;
  }

  Timer execute() override {
    Timer t;
    set_num_threads(1);
    OutTensor->reset();
    t.start();
    forwardForOneLayerFusedParallel(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->row, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, InTensor->NumThreads, FusedCompSet->n1_,
        FusedCompSet->ptr1_, FusedCompSet->ptr2_, FusedCompSet->id_,
        FusedCompSet->type_);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerSparseFusedParallel(GnnTensorInputs *In1, Stats *Stat1,
                                    sym_lib::ScheduleParameters SpIn)
      : GCNSingleLayerFused(In1, Stat1) {
    Inspector = new InspectorForAllFused(SpIn, Stat1);
  }
  ~GCNSingleLayerSparseFusedParallel() {
    delete FusedCompSet;
    delete Inspector;
  }
};

class GCNSingleLayerSparseFusedParallelWithGeMM : public GCNSingleLayerFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForAllFused *Inspector;

  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet =
        Inspector->generateFusedScheduleForAllFused(InTensor->AdjacencyMatrix);
    t.stop();
    return t;
  }

  Timer execute() override {
    double *intermediateResult = new double[InTensor->NumOfNodes*InTensor->Weight1->row];
    Timer t;
    St->OtherStats["FusedIterations"] = {(double)FusedCompSet->getNumberOfFusedNodes()};
    set_num_threads(1);
    OutTensor->reset();
    t.start();
    forwardForOneLayerFusedParallelSeparated(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->row, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, intermediateResult, InTensor->NumThreads, FusedCompSet->n1_,
        FusedCompSet->ptr1_, FusedCompSet->ptr2_, FusedCompSet->ker_begin_,
        FusedCompSet->id_, FusedCompSet->type_);
    t.stop();
    delete[] intermediateResult;
    return t;
  }

public:
  GCNSingleLayerSparseFusedParallelWithGeMM(GnnTensorInputs *In1, Stats *Stat1,
                                            sym_lib::ScheduleParameters SpIn)
      : GCNSingleLayerFused(In1, Stat1) {
    Inspector = new InspectorForAllFused(SpIn, Stat1);
  }
  ~GCNSingleLayerSparseFusedParallelWithGeMM() {
    delete FusedCompSet;
    delete Inspector;
  }
};

#ifdef __AVX2__
class GCNSingleLayerFusedCSCParallelVectorized : public GCNSingleLayerFused {
protected:
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(1);
    Timer t;
    t.start();

    forwardForOneLayerFromCSCVectorized(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->col, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  GCNSingleLayerFusedCSCParallelVectorized(GnnTensorInputs *In1, Stats *Stat1)
      : GCNSingleLayerFused(In1, Stat1) {}
};

class GCNSingleLayerTiledFusedCSCVectorized : public GCNSingleLayerFused {
protected:
  int TileSize;

  Timer execute() override {
    OutTensor->reset();
    set_num_threads(InTensor->NumThreads);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiledVectorized(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->col, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, TileSize);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedCSCVectorized(GnnTensorInputs *In1, Stats *Stat1,
                                        int TileSize1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1) {}
};

class GCNSingleLayerTiledFusedCSCCombinedVectorized
    : public GCNSingleLayerFused {
protected:
  int TileSize;
  int WorkloadMinSize;
  std::map<int, std::vector<int>> ConflictGraphColoring;

  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForSingleLayerTiledFusedCSCCombined *Inspector;
  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet = Inspector->generateScheduleBasedOnConflictGraphColoring(
        ConflictGraphColoring, InTensor->NumOfNodes, TileSize, WorkloadMinSize);
    t.stop();
    return t;
  }
  Timer execute() override {
    OutTensor->reset();
    set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiledParallelCombinedVectorized(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->Weight1->row, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1->a,
        OutTensor->FirstLayerOutput, TileSize, FusedCompSet->n3_,
        InTensor->NumThreads, FusedCompSet->n1_, FusedCompSet->n2_,
        FusedCompSet->ptr1_, FusedCompSet->id_, FusedCompSet->type_);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedCSCCombinedVectorized(
      GnnTensorInputs *In1, Stats *Stat1, int TileSize1, int WorkloadMinSize1,
      std::map<int, std::vector<int>> ConflictGraphColoring1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1),
        ConflictGraphColoring(ConflictGraphColoring1),
        WorkloadMinSize(WorkloadMinSize1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCCombined();
  }
  ~GCNSingleLayerTiledFusedCSCCombinedVectorized() {
    delete FusedCompSet;
    delete Inspector;
  }
};

#endif



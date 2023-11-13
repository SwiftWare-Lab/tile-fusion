//
// Created by salehm32 on 20/10/23.
//

#ifdef MKL
#include "../GCN_Layer_MKL_Forward_Utils.h"
#include "../MultiLayer/Fusion_Inspector.h"
#else
#include "GCN_Layer_Forward_Utils.h"
#endif
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
    for (int i = 0; i < InTensor->NumOfNodes * InTensor->EmbedDim; ++i) {
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
    mkl_set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayer(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
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
    mkl_set_num_threads(1);
    t.start();
    forwardForOneLayerParallel(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
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
  Timer execute() override {
    OutTensor->reset();
    mkl_set_num_threads(InTensor->NumThreads);
    sparse_matrix_t MKLAdj;
    mkl_sparse_d_create_csr(
        &MKLAdj, SPARSE_INDEX_BASE_ZERO, this->InTensor->NumOfNodes,
        this->InTensor->NumOfNodes, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->p + 1, this->InTensor->AdjacencyMatrix->i,
        this->InTensor->AdjacencyMatrix->x);
    Timer t;
    t.start();
    forwardForOneLayerWithGeMMAndSpMM(
        InTensor->NumOfNodes, MKLAdj, InTensor->FeatureMatrix->a,
        InTensor->FeatureMatrix->col, InTensor->Weight1, InTensor->EmbedDim,
        OutTensor->FirstLayerOutput);
    t.stop();
    mkl_free(MKLAdj);
    return t;
  }

public:
  GCNSingleLayerMKL(GnnTensorInputs *In1, Stats *Stat1)
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
    mkl_set_num_threads(InTensor->NumThreads);
    Timer t;
    t.start();
    forwardForOneLayerTiled(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
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
    mkl_set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerTiledParallel(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
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
    mkl_set_num_threads(InTensor->NumThreads);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiled(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
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
    mkl_set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerFromCSC(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
        OutTensor->FirstLayerOutput);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerFusedCSC(GnnTensorInputs *In1, Stats *Stat1)
      : GCNSingleLayerFused(In1, Stat1) {}
};

#ifdef __AVX2__
class GCNSingleLayerFusedCSCParallelVectorized : public GCNSingleLayerFused {
protected:
  Timer execute() override {
    OutTensor->reset();
    mkl_set_num_threads(1);
    Timer t;
    t.start();

    forwardForOneLayerFromCSCVectorized(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
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
    mkl_set_num_threads(InTensor->NumThreads);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiledVectorized(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
        OutTensor->FirstLayerOutput, TileSize);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedCSCVectorized(GnnTensorInputs *In1, Stats *Stat1,
                                        int TileSize1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1) {}
};

#endif

class GCNSingleLayerTiledFusedCSCParallel : public GCNSingleLayerFused {
protected:
  int TileSize;
  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForSingleLayerTiledFusedCSCParallel *Inspector;
  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet =
        Inspector->generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            InTensor->AdjacencyMatrixCSC, TileSize);
    t.stop();
    return t;
  }
  Timer execute() override {
    OutTensor->reset();
    mkl_set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiledParallelV2(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
        OutTensor->FirstLayerOutput, TileSize, InTensor->NumThreads,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        FusedCompSet->type_);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedCSCParallel(GnnTensorInputs *In1, Stats *Stat1,
                                      int TileSize1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }
  ~GCNSingleLayerTiledFusedCSCParallel() {
    delete FusedCompSet;
    delete Inspector;
  }
};

class GCNSingleLayerTiledFusedCSCCombined : public GCNSingleLayerFused {
protected:
  int TileSize;
  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForSingleLayerTiledFusedCSCCombined *Inspector;
  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet =
        Inspector->generateScheduleForSingleLayerTiledFusedCSCCombined(
            InTensor->AdjacencyMatrixCSC, TileSize);
    t.stop();
    return t;
  }
  Timer execute() override {
    OutTensor->reset();
    mkl_set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerFromCSCTiledParallelV2(
        InTensor->AdjacencyMatrixCSC->m, InTensor->AdjacencyMatrixCSC->p,
        InTensor->AdjacencyMatrixCSC->i, InTensor->AdjacencyMatrixCSC->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
        OutTensor->FirstLayerOutput, TileSize, InTensor->NumThreads,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        FusedCompSet->type_);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTiledFusedCSCCombined(GnnTensorInputs *In1, Stats *Stat1,
                                      int TileSize1)
      : GCNSingleLayerFused(In1, Stat1), TileSize(TileSize1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCCombined();
  }
  ~GCNSingleLayerTiledFusedCSCCombined() {
    delete FusedCompSet;
    delete Inspector;
  }
};
#endif

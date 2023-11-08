//
// Created by mehdi on 6/28/23.
//
#ifdef MKL
#include "../GCN_Layer_MKL_Forward_Utils.h"
#else
#include "../GCN_Layer_Forward_Utils.h"
#endif
#include "Fusion_Inspector.h"
#include "SWTensorBench.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include "sparse-fusion/SparseFusion.h"
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#ifndef SPARSE_FUSION_GCN_LAYER_DEMO_H
#define SPARSE_FUSION_GCN_LAYER_DEMO_H

#endif // SPARSE_FUSION_GCN_LAYER_DEMO_H
using namespace swiftware::benchmark;

double *generateRandomDenseMatrix(int M, int N) {
  std::random_device randDev;
  std::mt19937 generator(randDev());
  std::uniform_real_distribution<double> distr(-1., 1.);
  double *weight = new double[M * N];
  for (int i = 0; i < M * N; i++) {
    weight[i] = distr(generator);
  }
  return weight;
}

struct GnnTensorInputs : public Inputs<double> {
  double *Weight1, *Weight2;
  int *Degrees;
  sym_lib::Dense *FeatureMatrix;
  sym_lib::CSR *AdjacencyMatrix;
  size_t EmbedDim;
  size_t NumOfNodes;
  size_t BatchSize;
  std::vector<std::set<int>> LayerMasks;
  std::vector<sym_lib::CSR *> LayerMaskedMatrices;

  void normalizeAdjacencyMatrix() {
    this->Degrees = new int[this->NumOfNodes];
    for (int i = 0; i < this->NumOfNodes; i++) {
      this->Degrees[i] =
          this->AdjacencyMatrix->p[i + 1] - this->AdjacencyMatrix->p[i];
    }
    for (int i = 0; i < NumOfNodes; i++) {
      for (int j = AdjacencyMatrix->p[i]; j < AdjacencyMatrix->p[i + 1]; j++) {
        AdjacencyMatrix->x[j] =
            AdjacencyMatrix->x[j] /
            sqrt(Degrees[i] * Degrees[AdjacencyMatrix->i[j]]);
      }
    }
  }

  sym_lib::CSR *generateMaskedMatrix(std::set<int> NodeMask,
                                     sym_lib::CSR *AdjMatrix) {
    int *ap = AdjMatrix->p;
    int *ai = AdjMatrix->i;
    double *ax = AdjMatrix->x;
    int bnnz = 0;
    for (auto n : NodeMask) {
      bnnz += (ap[n + 1] - ap[n]);
    }
    sym_lib::CSR *bCsr = new sym_lib::CSR(AdjMatrix->m, AdjMatrix->n, bnnz);
    int *bp = bCsr->p;
    int *bi = bCsr->i;
    double *bx = bCsr->x;
    bp[0] = 0;
    int counter = 0;
    for (int i = 0; i < AdjMatrix->m; i++) {
      if (NodeMask.find(i) != NodeMask.end()) {
        for (int j = ap[i]; j < ap[i + 1]; j++) {
          bi[counter] = ai[j];
          bx[counter] = ax[j];
          counter++;
        }
        bp[i + 1] = counter;
      } else {
        bp[i + 1] = bp[i];
      }
    }
    return bCsr;
  }

  std::vector<std::set<int>> generateLayerMasks() {
    int numberOfNodes = this->NumOfNodes;
    int batchSize = this->BatchSize;
    std::vector<int> lastLayerMaskVector(numberOfNodes);
    std::iota(lastLayerMaskVector.begin(), lastLayerMaskVector.end(), 0);
    auto rng = std::default_random_engine{};
    std::shuffle(lastLayerMaskVector.begin(), lastLayerMaskVector.end(), rng);
    std::set<int> lastLayerMask(lastLayerMaskVector.begin(),
                                lastLayerMaskVector.begin() + batchSize);
    getPreviousLayerFeatureMask(lastLayerMask);
    std::vector<std::set<int>> layerMasks;
    layerMasks.emplace_back(getPreviousLayerFeatureMask(lastLayerMask));
    layerMasks.emplace_back(lastLayerMask);
    return layerMasks;
  }

  std::set<int> getPreviousLayerFeatureMask(std::set<int> LayerMask) {
    std::set<int> previousLayerMask;
    int *adjMtxIndex = this->AdjacencyMatrix->i;
    int *adjMtxP = this->AdjacencyMatrix->p;
    for (auto node : LayerMask) {
      previousLayerMask.emplace(node);
      for (int j = adjMtxP[node]; j < adjMtxP[node + 1]; j++) {
        previousLayerMask.emplace(adjMtxIndex[j]);
      }
    }
    return previousLayerMask;
  }

  GnnTensorInputs(double *Weight1, double *Weight2,
                  sym_lib::Dense *FeatureMatrix, sym_lib::CSC *AdjMtxCSC,
                  size_t NumOfNodes, size_t EmbedDim, size_t BatchSize,
                  int NumThreads1, int NumTrial1, std::string ExpN)
      : Inputs<double>(NumTrial1, NumThreads1, ExpN), Weight1(Weight1),
        Weight2(Weight2), FeatureMatrix(FeatureMatrix), NumOfNodes(NumOfNodes),
        EmbedDim(EmbedDim), BatchSize(BatchSize) {
    this->CorrectSol = nullptr;
    this->AdjacencyMatrix = sym_lib::csc_to_csr(AdjMtxCSC);
    this->LayerMasks = generateLayerMasks();
    this->normalizeAdjacencyMatrix();
    for (auto mask : LayerMasks) {
      LayerMaskedMatrices.emplace_back(
          this->generateMaskedMatrix(mask, this->AdjacencyMatrix));
    }
  }

  ~GnnTensorInputs() {
    delete[] Weight1;
    delete[] Weight2;
    delete FeatureMatrix;
    delete AdjacencyMatrix;
    delete LayerMaskedMatrices[0];
    delete LayerMaskedMatrices[1];
    delete[] Degrees;
  }
};

struct GnnTensorOutputs : public Outputs<double> {
  double *FirstLayerOutput, *SecondLayerOutput;
  size_t EmbedDim, NumOfNodes;

  GnnTensorOutputs(size_t EmbedDim, size_t NumOfNodes)
      : NumOfNodes(NumOfNodes), EmbedDim(EmbedDim) {
    this->FirstLayerOutput = new double[NumOfNodes * EmbedDim]{};
    this->SecondLayerOutput = new double[NumOfNodes * EmbedDim]{};
  }
  ~GnnTensorOutputs() {
    delete[] FirstLayerOutput;
    delete[] SecondLayerOutput;
  }

  void reset() {
    std::fill_n(FirstLayerOutput, EmbedDim * NumOfNodes, 0.0);
    std::fill_n(SecondLayerOutput, NumOfNodes * EmbedDim, 0.0);
  }
};

class GCNIntraFusedSequential : public SWTensorBench<double> {
protected:
  GnnTensorInputs *InTensor;
  void setup() override {
    this->St->OtherStats["Number of Sampled Nodes"] = {
        double(InTensor->LayerMasks[1].size())};
    this->St->OtherStats["Number of First Layer Nodes"] = {
        double(InTensor->LayerMasks[0].size())};
    this->St->OtherStats["Number of Fused Nodes"] = {0.};
  }

  bool verify(double &Error) override {
    bool retValue = true;
    if (In->CorrectSol == nullptr)
      return true;
    double infNorm = 0;
    for (int i = 0; i < InTensor->NumOfNodes * InTensor->EmbedDim; ++i) {
      if (std::abs(OutTensor->SecondLayerOutput[i] - In->CorrectSol[i]) >
          infNorm) {
        infNorm = std::abs(OutTensor->SecondLayerOutput[i] - In->CorrectSol[i]);
      }
    }
    Error = (double)infNorm;
    if (infNorm > In->Threshold) {
      retValue = false;
    }
    return retValue;
  }

  void preExecute() override {}

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
    forwardForOneLayer(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->EmbedDim, InTensor->EmbedDim, InTensor->Degrees,
        OutTensor->FirstLayerOutput, InTensor->Weight2,
        OutTensor->SecondLayerOutput);
    t.stop();
    return t;
  }

public:
  GnnTensorOutputs *OutTensor;
  GCNIntraFusedSequential(GnnTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<double>(In1, Stat1) {
    OutTensor = new GnnTensorOutputs(In1->EmbedDim, In1->NumOfNodes);
    InTensor = In1;
  }

  ~GCNIntraFusedSequential() { delete OutTensor; }
};

class GCNIntraFusedParallel : public GCNIntraFusedSequential {
protected:
  Timer execute() override {
    Timer t;
    mkl_set_num_threads(1);
    OutTensor->reset();
    t.start();
    forwardForOneLayerParallel(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
        OutTensor->FirstLayerOutput, InTensor->NumThreads);
    forwardForOneLayerParallel(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->EmbedDim, InTensor->EmbedDim, InTensor->Degrees,
        OutTensor->FirstLayerOutput, InTensor->Weight2,
        OutTensor->SecondLayerOutput, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  GCNIntraFusedParallel(GnnTensorInputs *In1, Stats *Stat1)
      : GCNIntraFusedSequential(In1, Stat1) {}
};

class GCNAllFusedParallel : public GCNIntraFusedSequential {
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
    mkl_set_num_threads(1);
    OutTensor->reset();
    t.start();
    forwardForFusedLayersParallel(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->EmbedDim,
        InTensor->Degrees, InTensor->FeatureMatrix->a, InTensor->Weight1,
        InTensor->Weight2, OutTensor->SecondLayerOutput,
        OutTensor->FirstLayerOutput, InTensor->NumThreads, FusedCompSet->n1_,
        FusedCompSet->ptr1_, FusedCompSet->ptr2_, FusedCompSet->id_,
        FusedCompSet->type_);
    t.stop();
    return t;
  }

public:
  GCNAllFusedParallel(GnnTensorInputs *In1, Stats *Stat1,
                      sym_lib::ScheduleParameters SpIn)
      : GCNIntraFusedSequential(In1, Stat1) {
    Inspector = new InspectorForAllFused(SpIn, Stat1);
  }
  ~GCNAllFusedParallel() {
    delete FusedCompSet;
    delete Inspector;
  }
};

#ifdef MKL
class GCNUnfused : public GCNIntraFusedSequential {
  sparse_matrix_t MKLFirstLayerAdj, MKLSecondLayerAdj;

protected:
  Timer execute() override {
    OutTensor->reset();
    mkl_set_num_threads(InTensor->NumThreads);
    Timer t;
    t.start();
    forwardForOneLayerWithGeMMAndSpMM(
        InTensor->NumOfNodes, MKLFirstLayerAdj, InTensor->FeatureMatrix->a,
        InTensor->FeatureMatrix->col, InTensor->Weight1, InTensor->EmbedDim,
        OutTensor->FirstLayerOutput);
    forwardForOneLayerWithGeMMAndSpMM(
        InTensor->NumOfNodes, MKLSecondLayerAdj, OutTensor->FirstLayerOutput,
        InTensor->EmbedDim, InTensor->Weight2, InTensor->EmbedDim,
        OutTensor->SecondLayerOutput);
    t.stop();
    return t;
  }

public:
  GCNUnfused(GnnTensorInputs *In1, Stats *Stat1)
      : GCNIntraFusedSequential(In1, Stat1) {
    mkl_sparse_d_create_csr(
        &MKLFirstLayerAdj, SPARSE_INDEX_BASE_ZERO, this->InTensor->NumOfNodes,
        this->InTensor->NumOfNodes, InTensor->LayerMaskedMatrices[0]->p,
        InTensor->AdjacencyMatrix->p + 1, InTensor->AdjacencyMatrix->i,
        InTensor->AdjacencyMatrix->x);
    mkl_sparse_d_create_csr(
        &MKLSecondLayerAdj, SPARSE_INDEX_BASE_ZERO, this->InTensor->NumOfNodes,
        this->InTensor->NumOfNodes, InTensor->LayerMaskedMatrices[1]->p,
        InTensor->AdjacencyMatrix->p + 1, InTensor->AdjacencyMatrix->i,
        InTensor->AdjacencyMatrix->x);
  }
  ~GCNUnfused() {
    mkl_free(MKLFirstLayerAdj);
    mkl_free(MKLSecondLayerAdj);
  }
};

class GCNIntraFusedUsingCSCSequential : public GCNIntraFusedSequential {
protected:
  Timer execute() override {
    Timer t;
    mkl_set_num_threads(1);
    OutTensor->reset();
    t.start();
    forwardForOneLayerFromCSC(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
        OutTensor->FirstLayerOutput);
    forwardForOneLayerFromCSC(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->EmbedDim, InTensor->EmbedDim, InTensor->Degrees,
        OutTensor->FirstLayerOutput, InTensor->Weight2,
        OutTensor->SecondLayerOutput);
    t.stop();
    return t;
  }

public:
  GCNIntraFusedUsingCSCSequential(GnnTensorInputs *In1, Stats *Stat1)
      : GCNIntraFusedSequential(In1, Stat1) {}
};

class GCNIntraTiledFusedUsingCSC : public GCNIntraFusedUsingCSCSequential {
protected:
  int TileSize;

  Timer execute() override {
    Timer t;
    mkl_set_num_threads(InTensor->NumThreads);
    OutTensor->reset();
    t.start();
    forwardForOneLayerFromCSCTiled(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
        OutTensor->FirstLayerOutput, TileSize);
    forwardForOneLayerFromCSCTiled(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->EmbedDim, InTensor->EmbedDim, InTensor->Degrees,
        OutTensor->FirstLayerOutput, InTensor->Weight2,
        OutTensor->SecondLayerOutput, TileSize);
    t.stop();
    return t;
  }

public:
  GCNIntraTiledFusedUsingCSC(GnnTensorInputs *In1, Stats *Stat1, int TileSize1)
      : GCNIntraFusedUsingCSCSequential(In1, Stat1), TileSize(TileSize1) {}
};

class GCNIntraLayerTiledFused : public GCNIntraFusedUsingCSCSequential {
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
    Timer t;
    mkl_set_num_threads(InTensor->NumThreads);
    OutTensor->reset();
    t.start();
    forwardForOneLayerTiled(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1,
        OutTensor->FirstLayerOutput, TileSize, Sp->GeMMLowerBounds,
        Sp->GeMMUpperBounds, Sp->MaxGeMMTileSize);
    forwardForOneLayerTiled(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->EmbedDim, InTensor->EmbedDim, InTensor->Degrees,
        OutTensor->FirstLayerOutput, InTensor->Weight2,
        OutTensor->SecondLayerOutput, TileSize, Sp->GeMMLowerBounds,
        Sp->GeMMUpperBounds, Sp->MaxGeMMTileSize);
    t.stop();
    return t;
  }

public:
  GCNIntraLayerTiledFused(GnnTensorInputs *In1, Stats *Stat1, int TileSize1)
      : GCNIntraFusedUsingCSCSequential(In1, Stat1), TileSize(TileSize1) {
    Inspector = new InspectorForTiledFused();
  }
  ~GCNIntraLayerTiledFused() {
    delete Inspector;
    delete Sp;
  }
};

class GCNAllTiledFusedCSC : public GCNIntraFusedSequential {
protected:
  int TileSize;
  InspectorForAllTiledFusedCSC *Inspector;
  sym_lib::MultiDimensionalSet *FusedCompSet;
  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet = Inspector->generateFusedScheduleForAllTiledFusedCSC(
        InTensor->AdjacencyMatrix, TileSize);
    t.stop();
    return t;
  }

  Timer execute() override {
    Timer t;
    mkl_set_num_threads(InTensor->NumThreads);
    OutTensor->reset();
    t.start();
    forwardForFusedLayersFromCSCTiled(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x,
        InTensor->FeatureMatrix->col, InTensor->EmbedDim, InTensor->EmbedDim,
        InTensor->Degrees, InTensor->FeatureMatrix->a, InTensor->Weight1,
        InTensor->Weight2, OutTensor->FirstLayerOutput,
        OutTensor->SecondLayerOutput, TileSize, FusedCompSet->ptr1_,
        FusedCompSet->id_, FusedCompSet->type_);
    t.stop();
    return t;
  }

public:
  GCNAllTiledFusedCSC(GnnTensorInputs *In1, Stats *Stat1, int TileSize1)
      : GCNIntraFusedSequential(In1, Stat1), TileSize(TileSize1) {
    Inspector = new InspectorForAllTiledFusedCSC();
  }
  ~GCNAllTiledFusedCSC() {
    delete Inspector;
    delete FusedCompSet;
  }
};

#endif

//////////////// Sampling Based
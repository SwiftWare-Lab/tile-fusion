//
// Created by mehdi on 6/28/23.
//
#include "SWTensorBench.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/GCNConv.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include "sparse-fusion/SparseFusion.h"
#include <numeric>
#include <random>
#include <set>

#ifndef SPARSE_FUSION_GCN_LAYER_DEMO_H
#define SPARSE_FUSION_GCN_LAYER_DEMO_H

#endif // SPARSE_FUSION_GCN_LAYER_DEMO_H
using namespace swiftware::benchmark;

double *generateWeightMatrix(int FeatDim, int EmbedDim) {
  std::random_device randDev;
  std::mt19937 generator(randDev());
  std::uniform_real_distribution<double> distr(-1., 1.);
  double *weight = new double[FeatDim * EmbedDim];
  for (int i = 0; i < FeatDim * EmbedDim; i++) {
    weight[i] = distr(generator);
  }
  return weight;
}

struct GnnTensorInputs : public Inputs<double> {
  double *Weight1, *Weight2;
  sym_lib::Dense *FeatureMatrix;
  sym_lib::CSR *AdjacencyMatrix;
  size_t EmbedDim, NumOfClasses;
  size_t NumOfNodes;
  size_t BatchSize;

  GnnTensorInputs(double *Weight1, double *Weight2,
                  sym_lib::Dense *FeatureMatrix, sym_lib::CSR *AdjacencyMatrix,
                  size_t NumOfNodes, size_t EmbedDim, size_t NumOfClasses,
                  size_t BatchSize, int NumThreads1, int NumTrial1,
                  std::string ExpN)
      : Inputs<double>(NumTrial1, NumThreads1, ExpN), Weight1(Weight1),
        Weight2(Weight2), FeatureMatrix(FeatureMatrix),
        AdjacencyMatrix(AdjacencyMatrix), NumOfNodes(NumOfNodes),
        NumOfClasses(NumOfClasses), EmbedDim(EmbedDim), BatchSize(BatchSize) {}

  GnnTensorInputs(sym_lib::Dense *FeatureMtx, sym_lib::CSC *AdjMtxCsc,
                  size_t NumOfNodes, size_t EmbedDim, size_t NumOfClasses,
                  size_t BatchSize, int NumThreads1, int NumTrial1,
                  std::string ExpN)
      : Inputs<double>(NumTrial1, NumThreads1, ExpN), FeatureMatrix(FeatureMtx),
        NumOfNodes(NumOfNodes), EmbedDim(EmbedDim), NumOfClasses(NumOfClasses),
        BatchSize(BatchSize) {
    this->CorrectSol = nullptr;
    this->Weight1 = generateWeightMatrix(FeatureMtx->col, EmbedDim);
    this->Weight2 = generateWeightMatrix(EmbedDim, NumOfClasses);
    this->AdjacencyMatrix = sym_lib::csc_to_csr(AdjMtxCsc);
  }
  ~GnnTensorInputs() {
    delete[] Weight1;
    delete[] Weight2;
    delete FeatureMatrix;
    delete AdjacencyMatrix;
  }
};

struct GnnTensorOutputs : public Outputs<double> {
  double *FirstLayerOutput, *SecondLayerOutput;
  size_t EmbedDim, NumOfClasses, NumOfNodes;

  GnnTensorOutputs(size_t EmbedDim, size_t NumOfClasses, size_t NumOfNodes) {
    this->FirstLayerOutput = new double[NumOfNodes * EmbedDim];
    this->SecondLayerOutput = new double[NumOfNodes * NumOfClasses];
  }
  ~GnnTensorOutputs() {
    delete[] FirstLayerOutput;
    delete[] SecondLayerOutput;
  }
};

class GCNSequential : public SWTensorBench<double> {
protected:
  GnnTensorInputs *InTensor;
  GnnTensorOutputs *OutTensor;
  sym_lib::gnn::GCNConvSequential *FirstConvLayer;
  sym_lib::gnn::GCNConvSequential *SecondConvLayer;

  void setup() override {}

  void preExecute() override {}
  Timer execute() override {
    Timer t;
    auto layerMasks = generateLayerMasks();
    t.start();
    FirstConvLayer->forward(InTensor->FeatureMatrix->a, layerMasks[0]);
    SecondConvLayer->forward(OutTensor->FirstLayerOutput, layerMasks[1]);
    t.stop();
    return t;
  }
  std::vector<std::vector<int>> generateLayerMasks() {
    int numberOfNodes = this->InTensor->NumOfNodes;
    int batchSize = this->InTensor->BatchSize;
    std::vector<int> lastLayerMaskVector(numberOfNodes);
    std::iota(lastLayerMaskVector.begin(), lastLayerMaskVector.end(), 1);
    auto rng = std::default_random_engine{};
    std::shuffle(lastLayerMaskVector.begin(), lastLayerMaskVector.end(), rng);
    std::set<int> lastLayerMask(lastLayerMaskVector.begin(),
                                lastLayerMaskVector.begin() + batchSize);
    std::vector<std::vector<int>> layerMasks;
    layerMasks.emplace_back(convertSetToVector(lastLayerMask));
    layerMasks.emplace_back(
        convertSetToVector(getPreviousLayerFeatureMask(lastLayerMask)));
    return layerMasks;
  }

  std::vector<int> convertSetToVector(std::set<int> S) {
    std::vector<int> v;
    v.reserve(S.size());
    std::copy(S.begin(), S.end(), std::back_inserter(v));
    return v;
  }

  std::set<int> getPreviousLayerFeatureMask(std::set<int> LayerMask) {
    std::set<int> previousLayerMask;
    int *adjMtxIndex = this->InTensor->AdjacencyMatrix->i;
    int *adjMtxP = this->InTensor->AdjacencyMatrix->p;
    for (auto node : LayerMask) {
      for (int j = adjMtxP[node]; j < adjMtxP[node]; j++) {
        previousLayerMask.emplace(adjMtxIndex[j]);
      }
    }
    return previousLayerMask;
  }

public:
  GCNSequential(GnnTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<double>(In1, Stat1) {
    OutTensor =
        new GnnTensorOutputs(In1->EmbedDim, In1->NumOfClasses, In1->NumOfNodes);
    InTensor = In1;
    FirstConvLayer = new sym_lib::gnn::GCNConvSequential(
        In1->AdjacencyMatrix, OutTensor->FirstLayerOutput, In1->Weight1,
        In1->FeatureMatrix->col, In1->EmbedDim);
    SecondConvLayer = new sym_lib::gnn::GCNConvSequential(
        In1->AdjacencyMatrix, OutTensor->SecondLayerOutput, In1->Weight2,
        In1->EmbedDim, In1->NumOfClasses);
  }
};

class GCNParallel : public SWTensorBench<double> {
protected:
  GnnTensorInputs *InTensor;
  GnnTensorOutputs *OutTensor;
  sym_lib::gnn::GCNConvParallel *FirstConvLayer;
  sym_lib::gnn::GCNConvParallel *SecondConvLayer;

  void setup() override {}

  void preExecute() override {}
  Timer execute() override {
    Timer t;
    auto layerMasks = generateLayerMasks();
    t.start();
    FirstConvLayer->forward(InTensor->FeatureMatrix->a, layerMasks[0]);
    SecondConvLayer->forward(OutTensor->FirstLayerOutput, layerMasks[1]);
    t.stop();
    return t;
  }
  std::vector<std::vector<int>> generateLayerMasks() {
    int numberOfNodes = this->InTensor->NumOfNodes;
    int batchSize = this->InTensor->BatchSize;
    std::vector<int> lastLayerMaskVector(numberOfNodes);
    std::iota(lastLayerMaskVector.begin(), lastLayerMaskVector.end(), 1);
    auto rng = std::default_random_engine{};
    std::shuffle(lastLayerMaskVector.begin(), lastLayerMaskVector.end(), rng);
    std::set<int> lastLayerMask(lastLayerMaskVector.begin(),
                                lastLayerMaskVector.begin() + batchSize);
    std::vector<std::vector<int>> layerMasks;
    layerMasks.emplace_back(convertSetToVector(lastLayerMask));
    layerMasks.emplace_back(
        convertSetToVector(getPreviousLayerFeatureMask(lastLayerMask)));
    return layerMasks;
  }

  std::vector<int> convertSetToVector(std::set<int> S) {
    std::vector<int> v;
    v.reserve(S.size());
    std::copy(S.begin(), S.end(), std::back_inserter(v));
    return v;
  }

  std::set<int> getPreviousLayerFeatureMask(std::set<int> LayerMask) {
    std::set<int> previousLayerMask;
    int *adjMtxIndex = this->InTensor->AdjacencyMatrix->i;
    int *adjMtxP = this->InTensor->AdjacencyMatrix->p;
    for (auto node : LayerMask) {
      for (int j = adjMtxP[node]; j < adjMtxP[node]; j++) {
        previousLayerMask.emplace(adjMtxIndex[j]);
      }
    }
    return previousLayerMask;
  }

public:
  GCNParallel(GnnTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<double>(In1, Stat1) {
    OutTensor =
        new GnnTensorOutputs(In1->EmbedDim, In1->NumOfClasses, In1->NumOfNodes);
    InTensor = In1;
    FirstConvLayer = new sym_lib::gnn::GCNConvParallel(
        In1->AdjacencyMatrix, OutTensor->FirstLayerOutput, In1->Weight1,
        In1->FeatureMatrix->col, In1->EmbedDim, In1->NumThreads);
    SecondConvLayer = new sym_lib::gnn::GCNConvParallel(
        In1->AdjacencyMatrix, OutTensor->SecondLayerOutput, In1->Weight2,
        In1->EmbedDim, In1->NumOfClasses, In1->NumThreads);
  }
};

class GCNFused : public SWTensorBench<double> {
protected:
  GnnTensorInputs *InTensor;
  GnnTensorOutputs *OutTensor;
  sym_lib::gnn::GCNConvFused *GCNNetwork;
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;

  void setup() override {}

  void preExecute() override {}

  Timer analysis() override {
    Timer t;
    t.start();
    // sym_lib::ScheduleParameters sp;
    // sp._num_threads = InTensor->NumThreads;
    //  create the fused set
    Sp._num_w_partition =
        std::max<int>(InTensor->AdjacencyMatrix->m / Sp._num_w_partition,
                      2 * Sp._num_threads);
    auto *sf01 = new sym_lib::SparseFusion(&Sp, 2);
    auto *mvDAG = sym_lib::diagonal(InTensor->AdjacencyMatrix->m, 1.0);
    sf01->fuse(0, mvDAG, NULLPNTR);
    auto *tmpCSCCSR = new sym_lib::CSC(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->n,
        InTensor->AdjacencyMatrix->nnz, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x);
    // sf01->print_final_list();
    sf01->fuse(1, mvDAG, tmpCSCCSR);
    // sf01->print_final_list();
    SpInfo = sf01->measureReuse(tmpCSCCSR);
    // std::cout<<" -> "<<spi.TotalReuseC<<std::endl;
    auto pt = St->OtherStats["PackingType"];
    // FusedCompSet = sf01->getFusedCompressed((int) pt[0]);
    // FusedCompSet->print_3d();
    delete sf01;
    delete mvDAG;

    t.stop();
    return t;
  }

  Timer execute() override {
    Timer t;
    auto layerMasks = generateLayerMasks();
    t.start();
    GCNNetwork->forward(InTensor->FeatureMatrix->a, FusedCompSet->n1_, FusedCompSet->ptr1_,
                        FusedCompSet->ptr2_, FusedCompSet->id_,
                        FusedCompSet->type_, layerMasks[0]);
    t.stop();
    return t;
  }
  std::vector<std::vector<int>> generateLayerMasks() {
    int numberOfNodes = this->InTensor->NumOfNodes;
    int batchSize = this->InTensor->BatchSize;
    std::vector<int> lastLayerMaskVector(numberOfNodes);
    std::iota(lastLayerMaskVector.begin(), lastLayerMaskVector.end(), 1);
    auto rng = std::default_random_engine{};
    std::shuffle(lastLayerMaskVector.begin(), lastLayerMaskVector.end(), rng);
    std::set<int> lastLayerMask(lastLayerMaskVector.begin(),
                                lastLayerMaskVector.begin() + batchSize);
    std::vector<std::vector<int>> layerMasks;
    layerMasks.emplace_back(convertSetToVector(lastLayerMask));
    layerMasks.emplace_back(
        convertSetToVector(getPreviousLayerFeatureMask(lastLayerMask)));
    return layerMasks;
  }

  std::vector<int> convertSetToVector(std::set<int> S) {
    std::vector<int> v;
    v.reserve(S.size());
    std::copy(S.begin(), S.end(), std::back_inserter(v));
    return v;
  }

  std::set<int> getPreviousLayerFeatureMask(std::set<int> LayerMask) {
    std::set<int> previousLayerMask;
    int *adjMtxIndex = this->InTensor->AdjacencyMatrix->i;
    int *adjMtxP = this->InTensor->AdjacencyMatrix->p;
    for (auto node : LayerMask) {
      for (int j = adjMtxP[node]; j < adjMtxP[node]; j++) {
        previousLayerMask.emplace(adjMtxIndex[j]);
      }
    }
    return previousLayerMask;
  }

public:
  GCNFused(GnnTensorInputs *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn)
      : SWTensorBench<double>(In1, Stat1), Sp(SpIn) {
    OutTensor =
        new GnnTensorOutputs(In1->EmbedDim, In1->NumOfClasses, In1->NumOfNodes);
    InTensor = In1;

    GCNNetwork = new sym_lib::gnn::GCNConvFused(
        In1->AdjacencyMatrix, OutTensor->SecondLayerOutput,
        OutTensor->FirstLayerOutput, In1->Weight1, In1->Weight2,
        In1->FeatureMatrix->col, In1->NumOfClasses, In1->EmbedDim,
        this->InTensor->NumThreads);
  }
};

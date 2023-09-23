//
// Created by mehdi on 6/28/23.
//
#include "SWTensorBench.h"
#include "aggregation/def.h"
#include "sparse-fusion/GCNConv.h"
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

  GnnTensorInputs(sym_lib::Dense *FeatureMtx, sym_lib::CSR *AdjacencyMatrix,
                  size_t NumOfNodes, size_t EmbedDim, size_t NumOfClasses,
                  size_t BatchSize, int NumThreads1, int NumTrial1,
                  std::string ExpN)
      : Inputs<double>(NumTrial1, NumThreads1, ExpN), FeatureMatrix(FeatureMtx),
        AdjacencyMatrix(AdjacencyMatrix), NumOfNodes(NumOfNodes),
        EmbedDim(EmbedDim), NumOfClasses(NumOfClasses), BatchSize(BatchSize) {
    this->CorrectSol = nullptr;
    this->Weight1 = generateWeightMatrix(FeatureMtx->col, EmbedDim);
    this->Weight2 = generateWeightMatrix(EmbedDim, NumOfClasses);
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

class GCNGnn : public SWTensorBench<double> {
protected:
  GnnTensorInputs *InTensor;
  GnnTensorOutputs *OutTensor;
  sym_lib::gnn::GCNConv *FirstConvLayer;
  sym_lib::gnn::GCNConv *SecondConvLayer;

  void setup() override {}

  void preExecute() override {}
  Timer execute() override {
    Timer t;
    t.start();
    auto layerMasks = generateLayerMasks();
    FirstConvLayer->forward(InTensor->FeatureMatrix->a, layerMasks[0]);
    SecondConvLayer->forward(OutTensor->FirstLayerOutput, layerMasks[1]);
    t.stop();
    return t;
  }
  std::vector<std::set<int>> generateLayerMasks() {
    int numberOfNodes = this->InTensor->NumOfNodes;
    int batchSize = this->InTensor->BatchSize;
    std::vector<int> lastLayerMaskVector(numberOfNodes);
    std::iota(lastLayerMaskVector.begin(), lastLayerMaskVector.end(), 1);
    auto rng = std::default_random_engine{};
    std::shuffle(lastLayerMaskVector.begin(), lastLayerMaskVector.end(), rng);
    std::set<int> lastLayerMask(lastLayerMaskVector.begin(),
                                lastLayerMaskVector.begin() + batchSize);
    std::vector<std::set<int>> layerMasks;
    layerMasks.emplace_back(lastLayerMask);
    layerMasks.emplace_back(getPreviousLayerFeatureMask(lastLayerMask));
    return layerMasks;
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
  GCNGnn(GnnTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<double>(In1, Stat1) {
    OutTensor =
        new GnnTensorOutputs(In1->EmbedDim, In1->NumOfClasses, In1->NumOfNodes);
    InTensor = In1;
    FirstConvLayer = new sym_lib::gnn::GCNConv(
        In1->AdjacencyMatrix, OutTensor->FirstLayerOutput, In1->Weight1,
        In1->FeatureMatrix->col, In1->EmbedDim);
    SecondConvLayer = new sym_lib::gnn::GCNConv(
        In1->AdjacencyMatrix, OutTensor->SecondLayerOutput, In1->Weight2,
        In1->EmbedDim, In1->NumOfClasses);
  }
};

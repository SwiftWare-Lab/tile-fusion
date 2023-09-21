//
// Created by mehdi on 6/28/23.
//
#include "SWTensorBench.h"
#include "aggregation/def.h"
#include "sparse-fusion/GCNConv.h"
#include <random>

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
  double *Weight1, *Weight2, *Feature;
  sym_lib::CSR *AdjacencyMatrix;
  size_t FeatDim, EmbedDim, NumOfClasses;
  size_t NumOfNodes;

  GnnTensorInputs(double *Weight1, double *Weight2, double *Feature,
                  sym_lib::CSR *AdjacencyMatrix, size_t NumOfNodes,
                  size_t FeatDim, size_t EmbedDim, size_t NumOfClasses,
                  int NumThreads1, int NumTrial1, std::string ExpN)
      : Inputs<double>(NumTrial1, NumThreads1, ExpN), Weight1(Weight1),
        Weight2(Weight2), Feature(Feature), AdjacencyMatrix(AdjacencyMatrix),
        NumOfNodes(NumOfNodes), FeatDim(FeatDim), NumOfClasses(NumOfClasses),
        EmbedDim(EmbedDim) {}

  GnnTensorInputs(double *Feature, sym_lib::CSR *AdjacencyMatrix,
                  size_t NumOfNodes, size_t FeatDim, size_t EmbedDim,
                  size_t NumOfClasses, int NumThreads1, int NumTrial1,
                  std::string ExpN)
      : Inputs<double>(NumTrial1, NumThreads1, ExpN), Feature(Feature),
        AdjacencyMatrix(AdjacencyMatrix), NumOfNodes(NumOfNodes),
        FeatDim(FeatDim), EmbedDim(EmbedDim), NumOfClasses(NumOfClasses) {
    this->CorrectSol = nullptr;
    this->Weight1 = generateWeightMatrix(FeatDim, EmbedDim);
    this->Weight2 = generateWeightMatrix(EmbedDim, NumOfClasses);
  }
  ~GnnTensorInputs() {
    delete[] Weight1;
    delete[] Weight2;
    delete[] Feature;
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
    FirstConvLayer->forward(InTensor->Feature);
    SecondConvLayer->forward(OutTensor->FirstLayerOutput);
    t.stop();
    return t;
  }

public:
  GCNGnn(GnnTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<double>(In1, Stat1) {
    OutTensor =
        new GnnTensorOutputs(In1->EmbedDim, In1->NumOfClasses, In1->NumOfNodes);
    InTensor = In1;
    FirstConvLayer = new sym_lib::gnn::GCNConv(
        In1->AdjacencyMatrix, OutTensor->FirstLayerOutput, In1->Weight1,
        In1->FeatDim, In1->EmbedDim);
    SecondConvLayer = new sym_lib::gnn::GCNConv(
        In1->AdjacencyMatrix, OutTensor->SecondLayerOutput, In1->Weight2,
        In1->EmbedDim, In1->NumOfClasses);
  }
};

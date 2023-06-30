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

float *generate_weight_matrix(int featDim, int embedDim) {
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_real_distribution<float> distr(-1., 1.);
  float *weight = new float[featDim * embedDim];
  for (int i = 0; i < featDim * embedDim; i++) {
    weight[i] = distr(generator);
  }
  return weight;
}

struct GnnTensorInputs : public Inputs<float> {
  float *weight1, *weight2, *feature;
  sym_lib::CSR *adjacencyMatrix;
  size_t featDim, embedDim, numOfClasses;
  size_t numOfNodes;

  GnnTensorInputs(float *weight1, float *weight2, float *feature,
                  sym_lib::CSR *adjacencyMatrix, size_t numOfNodes,
                  size_t featDim, size_t embedDim, size_t numOfClasses,
                  int NumThreads1, int NumTrial1, std::string ExpN)
      : Inputs<float>(NumTrial1, NumThreads1, ExpN), weight1(weight1),
        weight2(weight2), feature(feature), adjacencyMatrix(adjacencyMatrix),
        numOfNodes(numOfNodes), featDim(featDim), numOfClasses(numOfClasses),
        embedDim(embedDim) {}

  GnnTensorInputs(float *feature, sym_lib::CSR *adjacencyMatrix,
                  size_t NumOfNodes, size_t featDim, size_t embedDim,
                  size_t numOfClasses, int NumThreads1, int NumTrial1,
                  std::string ExpN)
      : Inputs<float>(NumTrial1, NumThreads1, ExpN), feature(feature),
        adjacencyMatrix(adjacencyMatrix), numOfNodes(NumOfNodes),
        featDim(featDim), embedDim(embedDim), numOfClasses(numOfClasses) {
    this->weight1 = generate_weight_matrix(featDim, embedDim);
    this->weight2 = generate_weight_matrix(embedDim, embedDim);
  }
  ~GnnTensorInputs() {
    delete[] weight1;
    delete[] weight2;
    delete[] feature;
    delete adjacencyMatrix;
  }
};

struct GnnTensorOutputs : public Outputs<float> {
  float *firstLayerOutput, *secondLayerOutput;
  size_t embedDim, numOfClasses, numOfNodes;

  GnnTensorOutputs(size_t embedDim, size_t numOfClasses, size_t numOfNodes) {
    this->firstLayerOutput = new float[numOfNodes * embedDim];
    this->secondLayerOutput = new float[numOfNodes * numOfClasses];
  }
  ~GnnTensorOutputs() {
    delete[] firstLayerOutput;
    delete[] secondLayerOutput;
  }
};

class GCNGnn : public SWTensorBench<float> {
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
    FirstConvLayer->forward(InTensor->feature);
    FirstConvLayer->forward(OutTensor->firstLayerOutput);
    t.stop();
    return t;
  }

public:
  GCNGnn(GnnTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<float>(In1, Stat1) {
    OutTensor =
        new GnnTensorOutputs(In1->embedDim, In1->numOfClasses, In1->numOfNodes);
    InTensor = In1;
    FirstConvLayer = new sym_lib::gnn::GCNConv(
        In1->adjacencyMatrix, OutTensor->firstLayerOutput, In1->weight1,
        In1->featDim, In1->embedDim);
    SecondConvLayer = new sym_lib::gnn::GCNConv(
        In1->adjacencyMatrix, OutTensor->secondLayerOutput, In1->weight2,
        In1->embedDim, In1->numOfClasses);
  }
};

//
// Created by salehm32 on 07/03/24.
//
#ifdef MKL
#include "../GCN_Layer_MKL_Forward_Utils.h"
#else
#include "../GCN_Layer_Forward_Utils.h"
#endif
#include "SWTensorBench.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Inspector.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include "sparse-fusion/SparseFusion.h"
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <set>

#ifndef SPARSE_FUSION_GCN_SINGLE_LAYER_SP_DEMO_UTILS_H
#define SPARSE_FUSION_GCN_SINGLE_LAYER_SP_DEMO_UTILS_H
struct GnnTensorSpInputs : public Inputs<float> {
  float *Weight1;
  int EmbedDim;
  int FeatureDim;
  int *Degrees;
  float *FeatureMatrix;
  sym_lib::CSR *AdjacencyMatrix;
  float* AMValues;
  sym_lib::CSC *AdjacencyMatrixCSC;
  size_t NumOfNodes;
  size_t BatchSize;

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

  GnnTensorSpInputs(float *Weight1, float *FeatureMatrix,
                    sym_lib::CSC *AdjMtxCSC, size_t NumOfNodes, size_t EmbedDim,
                    size_t FeatDim,int NumThreads1,
                    int NumTrial1, std::string ExpN)
      : Inputs<float>(NumTrial1, NumThreads1, ExpN), Weight1(Weight1),
        FeatureMatrix(FeatureMatrix), NumOfNodes(NumOfNodes), EmbedDim(EmbedDim),
        FeatureDim(FeatDim){
    this->CorrectSol = nullptr;
    this->AdjacencyMatrix = sym_lib::csc_to_csr(AdjMtxCSC);
    this->normalizeAdjacencyMatrix();
    this->AMValues = new float[this->AdjacencyMatrix->nnz];
    for (int i = 0; i < this->AdjacencyMatrix->nnz; i++) {
      this->AMValues[i] = (float)this->AdjacencyMatrix->x[i];
    }
    this->AdjacencyMatrixCSC = sym_lib::csr_to_csc(this->AdjacencyMatrix);
  }

  ~GnnTensorSpInputs() {
    delete[] Weight1;
    delete[] FeatureMatrix;
    delete AdjacencyMatrix;
    delete AdjacencyMatrixCSC;
    delete[] AMValues;
    delete[] Degrees;
  }
};

struct GnnTensorSpOutputs : public Outputs<float> {
  float *FirstLayerOutput;
  size_t EmbedDim, NumOfNodes;

  GnnTensorSpOutputs(size_t EmbedDim, size_t NumOfNodes)
      : NumOfNodes(NumOfNodes), EmbedDim(EmbedDim) {
    this->FirstLayerOutput = new float [NumOfNodes * EmbedDim]{};
  }
  ~GnnTensorSpOutputs() {
    delete[] FirstLayerOutput;
  }

  void reset() {
    std::fill_n(FirstLayerOutput, EmbedDim * NumOfNodes, 0.0);
  }
};

class GCNSingleLayerUnFusedCSRMKLGeMMSP : public SWTensorBench<float> {
protected:
  GnnTensorSpInputs *InTensor;
  void setup() override {
    //    this->St->OtherStats["Number of Sampled Nodes"] = {
    //        double(InTensor->LayerMasks[1].size())};
    //    this->St->OtherStats["Number of First Layer Nodes"] = {
    //        double(InTensor->LayerMasks[0].size())};
    this->St->OtherStats["FusedIterations"] = {0.};
    this->St->OtherStats["Min Workload Size"] = {10.};
  }

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

  void preExecute() override {}

  Timer execute() override {
    OutTensor->reset();
    float *intermediateResult = new float [InTensor->NumOfNodes * InTensor->EmbedDim]{};
    mkl_set_num_threads(InTensor->NumThreads);
    Timer t;
    t.start();
    forwardForOneLayerWithMKLGeMMAndSpMMSPVectorized(
        InTensor->NumOfNodes, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AMValues,
        InTensor->FeatureMatrix, InTensor->FeatureDim,
        InTensor->Weight1, InTensor->EmbedDim,
        OutTensor->FirstLayerOutput, intermediateResult, InTensor->NumThreads);
    t.stop();
    delete[] intermediateResult;
    return t;
  }

public:
  GnnTensorSpOutputs *OutTensor;
  GCNSingleLayerUnFusedCSRMKLGeMMSP(GnnTensorSpInputs *In1, Stats *Stat1)
      : SWTensorBench<float>(In1, Stat1) {
    OutTensor = new GnnTensorSpOutputs(In1->EmbedDim, In1->NumOfNodes);
    InTensor = In1;
  }

  ~GCNSingleLayerUnFusedCSRMKLGeMMSP() { delete OutTensor; }
};

class GCNSingleLayerMKL_SP : public GCNSingleLayerUnFusedCSRMKLGeMMSP {

protected:
  sparse_matrix_t MKLAdj;
  Timer execute() override {
    OutTensor->reset();
    mkl_set_num_threads(InTensor->NumThreads);
    Timer t;
    float *intermediateResult = new float [InTensor->NumOfNodes * InTensor->EmbedDim]{};
    t.start();
    forwardForOneLayerWithMKLGeMMAndMKLSpMMSP(
        InTensor->NumOfNodes, MKLAdj, InTensor->FeatureMatrix,
        InTensor->FeatureDim, InTensor->Weight1,
        InTensor->EmbedDim, OutTensor->FirstLayerOutput, intermediateResult);
    t.stop();
    delete[] intermediateResult;
    return t;
  }

public:
  GCNSingleLayerMKL_SP(GnnTensorSpInputs *In1, Stats *Stat1)
      : GCNSingleLayerUnFusedCSRMKLGeMMSP(In1, Stat1) {
    mkl_sparse_s_create_csr(
        &MKLAdj, SPARSE_INDEX_BASE_ZERO, this->InTensor->NumOfNodes,
        this->InTensor->NumOfNodes, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->p + 1, this->InTensor->AdjacencyMatrix->i,
        this->InTensor->AMValues);
  }
  ~GCNSingleLayerMKL_SP() { mkl_free(MKLAdj); }
};

class GCNSingleLayerSparseFusedParallelWithGeMM_SP : public GCNSingleLayerUnFusedCSRMKLGeMMSP {
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
    float *intermediateResult = new float [InTensor->NumOfNodes*InTensor->EmbedDim];
    Timer t;
    St->OtherStats["FusedIterations"] = {(double)FusedCompSet->getNumberOfFusedNodes()};
    mkl_set_num_threads(1);
    OutTensor->reset();
    t.start();
    forwardForOneLayerFusedParallelSeparatedVectorizedSP(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AMValues,
        InTensor->FeatureDim, InTensor->EmbedDim,
        InTensor->FeatureMatrix, InTensor->Weight1,
        OutTensor->FirstLayerOutput, intermediateResult, InTensor->NumThreads, FusedCompSet->n1_,
        FusedCompSet->ptr1_, FusedCompSet->ptr2_, FusedCompSet->ker_begin_,
        FusedCompSet->id_);
    t.stop();
    delete[] intermediateResult;
    return t;
  }

public:
  GCNSingleLayerSparseFusedParallelWithGeMM_SP(GnnTensorSpInputs *In1, Stats *Stat1,
                                            sym_lib::ScheduleParameters SpIn)
      : GCNSingleLayerUnFusedCSRMKLGeMMSP(In1, Stat1) {
    Inspector = new InspectorForAllFused(SpIn, Stat1);
  }
  ~GCNSingleLayerSparseFusedParallelWithGeMM_SP() {
    delete FusedCompSet;
    delete Inspector;
  }
};

#endif // SPARSE_FUSION_GCN_SINGLE_LAYER_SP_DEMO_UTILS_H

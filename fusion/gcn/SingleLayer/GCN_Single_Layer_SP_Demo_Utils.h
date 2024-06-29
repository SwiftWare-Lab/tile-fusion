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
#include "sparse-fusion/SparseFusionWithRedundancy.h"
#include "../gemm_spmm_codegen.h"
#include <cassert>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <set>

#define CEIL(x, y) (((x) + (y)-1) / (y))

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

class GCNSingleLayerMKLSpMMGeMM_SP : public GCNSingleLayerUnFusedCSRMKLGeMMSP {

protected:
  sparse_matrix_t MKLAdj;
  Timer execute() override {
    OutTensor->reset();
    mkl_set_num_threads(InTensor->NumThreads);
    Timer t;
    float *intermediateResult = new float [InTensor->NumOfNodes * InTensor->FeatureDim]{};
    std::fill_n(intermediateResult, InTensor->NumOfNodes * InTensor->FeatureDim, 0.0);
    t.start();
    forwardForOneLayerWithMKLSpMMAndMKLGeMMSP(
        InTensor->NumOfNodes, MKLAdj, InTensor->FeatureMatrix,
        InTensor->FeatureDim, InTensor->Weight1,
        InTensor->EmbedDim, OutTensor->FirstLayerOutput, intermediateResult);
    t.stop();
    delete[] intermediateResult;
    return t;
  }

public:
  GCNSingleLayerMKLSpMMGeMM_SP(GnnTensorSpInputs *In1, Stats *Stat1)
      : GCNSingleLayerUnFusedCSRMKLGeMMSP(In1, Stat1) {
    mkl_sparse_s_create_csr(
        &MKLAdj, SPARSE_INDEX_BASE_ZERO, this->InTensor->NumOfNodes,
        this->InTensor->NumOfNodes, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->p + 1, this->InTensor->AdjacencyMatrix->i,
        this->InTensor->AMValues);
  }
  ~GCNSingleLayerMKLSpMMGeMM_SP() { mkl_free(MKLAdj); }
};

class GCNSingleLayerSpMMGeMVFused : public GCNSingleLayerUnFusedCSRMKLGeMMSP {

protected:
  Timer execute() override {
    OutTensor->reset();
    mkl_set_num_threads(1);
    Timer t;
    t.start();
    forwardForOneLayerSpMMGemVFusedSp(
        InTensor->NumOfNodes, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AMValues, InTensor->FeatureDim,
        InTensor->EmbedDim, InTensor->FeatureMatrix, InTensor->Weight1,
        OutTensor->FirstLayerOutput, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerSpMMGeMVFused(GnnTensorSpInputs *In1, Stats *Stat1)
      : GCNSingleLayerUnFusedCSRMKLGeMMSP(In1, Stat1) {
  }
  ~GCNSingleLayerSpMMGeMVFused() {}
};

class GCNSingleLayerCSRAtomicFused : public GCNSingleLayerUnFusedCSRMKLGeMMSP {
  sym_lib::ScheduleParameters Sp;
protected:
  Timer execute() override {
    OutTensor->reset();
    mkl_set_num_threads(1);
    float *intermediateResult = new float [InTensor->NumOfNodes*InTensor->EmbedDim];
    Timer t;
    t.start();
    forwardForOneLayerFusedParallelCSCAtomicSP(
        InTensor->NumOfNodes, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AMValues, InTensor->FeatureDim,
        InTensor->EmbedDim, InTensor->FeatureMatrix, InTensor->Weight1,
        OutTensor->FirstLayerOutput, intermediateResult,
        InTensor->NumThreads, Sp.IterPerPartition);
    t.stop();
    delete[]  intermediateResult;
    return t;
  }

public:
  GCNSingleLayerCSRAtomicFused(GnnTensorSpInputs *In1, Stats *Stat1, sym_lib::ScheduleParameters Sp1)
      : GCNSingleLayerUnFusedCSRMKLGeMMSP(In1, Stat1), Sp(Sp1) {
  }
  ~GCNSingleLayerCSRAtomicFused() {}
};


class GCNSingleLayerSpMMVectorizedGeMMFusedSp : public GCNSingleLayerUnFusedCSRMKLGeMMSP {
  sym_lib::ScheduleParameters Sp;
protected:
  Timer execute() override {
    OutTensor->reset();
    mkl_set_num_threads(1);
    float *intermediateResult = new float [InTensor->NumOfNodes*InTensor->FeatureDim]{};
    std::fill_n(intermediateResult, InTensor->NumOfNodes * InTensor->FeatureDim, 0.0);
    Timer t;
    t.start();
    SpMMGeMMInterleavedFusedVectorizedSP(
        InTensor->NumOfNodes, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AMValues, InTensor->FeatureDim,
        InTensor->EmbedDim, InTensor->FeatureMatrix, InTensor->Weight1,
        OutTensor->FirstLayerOutput, intermediateResult,
        InTensor->NumThreads, Sp.IterPerPartition);
    t.stop();
    delete[]  intermediateResult;
    return t;
  }

public:
  GCNSingleLayerSpMMVectorizedGeMMFusedSp(GnnTensorSpInputs *In1, Stats *Stat1, sym_lib::ScheduleParameters Sp1)
      : GCNSingleLayerUnFusedCSRMKLGeMMSP(In1, Stat1), Sp(Sp1) {
  }
  ~GCNSingleLayerSpMMVectorizedGeMMFusedSp() {}
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

class GCNSingleLayerSparseFusedP2PThreadWithGeMM_SP
    : public GCNSingleLayerUnFusedCSRMKLGeMMSP {
protected:
  sym_lib::ScheduleParameters Sp;
  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForAllFused *Inspector;
  int **Parents;
  int NumTasks;
  int *NPar;

  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet =
        Inspector->generateFusedScheduleForAllFused(InTensor->AdjacencyMatrix);
    createP2PPointers(FusedCompSet->ptr1_, FusedCompSet->ptr2_, FusedCompSet->ker_begin_, FusedCompSet->id_);
    t.stop();
    return t;
  }

  // only for two levels for now.
  void createP2PPointers(int *LevelPtr, int *ParPtr, int *MixPtr, int *Id){
    int **parents;
    int *nPar;
    int numLevels=2; //also used as numKernels
    int* ap = InTensor->AdjacencyMatrix->p;
    int* ai = InTensor->AdjacencyMatrix->i;
    NumTasks = LevelPtr[numLevels];
    parents = new int*[NumTasks];
    nPar = new int[NumTasks];
    for (int l1 = LevelPtr[0]; l1 < LevelPtr[1]; l1++){
      nPar[l1] = 0;
      parents[l1] = NULLPNTR;
    }
    for (int l1 = LevelPtr[1]; l1 < LevelPtr[2]; l1++){
//        int kBeginL1 = ParPtr[l1];
      int kBeginL2 = MixPtr[l1 * numLevels];
      int kEndL2 = MixPtr[l1 * numLevels + 1];
      std::set<int> parsVec;
      for (int i1=kBeginL2; i1 < kEndL2; i1++){
        int row = Id[i1];
        for (int j = ap[row]; j < ap[row + 1]; j++){
          int cInd = ai[j];
          for (int l2 = LevelPtr[0]; l2 < LevelPtr[1]; l2++) {
            if (cInd <=  Id[MixPtr[l2 * numLevels] - 1]) {
              parsVec.insert(l2);
              break;
            }
          }
        }
      }
      nPar[l1] = parsVec.size();
      parents[l1] = new int[nPar[l1]];
      int parCntr = 0;
      for (std::set<int>::iterator iter = parsVec.begin(); iter != parsVec.end(); iter++){
        parents[l1][parCntr] = *iter;
        parCntr++;
      }
    }
    Parents = parents;
    NPar = nPar;
  }

  Timer execute() override {
    float *intermediateResult = new float [InTensor->NumOfNodes*InTensor->EmbedDim];
    Timer t;
    St->OtherStats["FusedIterations"] = {(double)FusedCompSet->getNumberOfFusedNodes()};
    mkl_set_num_threads(1);
    OutTensor->reset();
    bool *taskFinished = new bool[NumTasks];
    for (int i = 0; i < NumTasks; i++){
      taskFinished[i] = false;
    }
    t.start();
    geMMSpMMFusedParallelSeparatedP2PThreadVectorizedSP(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AMValues,
        InTensor->FeatureDim, InTensor->EmbedDim,
        InTensor->FeatureMatrix, InTensor->Weight1,
        OutTensor->FirstLayerOutput, intermediateResult, InTensor->NumThreads, FusedCompSet->n1_,
        FusedCompSet->ptr1_, FusedCompSet->ptr2_, FusedCompSet->ker_begin_,
        FusedCompSet->id_, NPar, Parents, taskFinished);
    t.stop();
    delete[] taskFinished;
    delete[] intermediateResult;
    return t;
  }

public:

  GCNSingleLayerSparseFusedP2PThreadWithGeMM_SP(GnnTensorSpInputs *In1, Stats *Stat1,
                                               sym_lib::ScheduleParameters SpIn)
      : GCNSingleLayerUnFusedCSRMKLGeMMSP(In1, Stat1), Sp(SpIn) {
    Inspector = new InspectorForAllFused(SpIn, Stat1);
  }

  ~GCNSingleLayerSparseFusedP2PThreadWithGeMM_SP() {
    delete FusedCompSet;
    delete Inspector;
    for (int i = 0; i < NumTasks; i++){
      delete []Parents[i];
    }
    delete []Parents;
    delete []NPar;
  }
};

class GCNSingleLayerSparseFusedReorderedUnfusedWithGeMM_SP : public GCNSingleLayerUnFusedCSRMKLGeMMSP {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForAllFused *Inspector;
  int* UFAp;
  int* UFAi;
  float* UFAx;

  Timer analysis() override {
    Timer t;
    t.start();
    FusedCompSet =
        Inspector->generateFusedScheduleForAllFused(InTensor->AdjacencyMatrix);
    createUnfusedData(FusedCompSet->ptr1_, FusedCompSet->ker_begin_, FusedCompSet->id_);
    t.stop();
    return t;
  }

  void createUnfusedData(int *LevelPtr, int *MixPtr, int *Id){
    int numKernels = 2;
//    int unfusedStart = MixPtr[LevelPtr[1] * numKernels];
    int* ap = InTensor->AdjacencyMatrix->p;
    int* ai = InTensor->AdjacencyMatrix->i;
    float* ax = InTensor->AMValues;
    std::vector<int> ufApVec;
    std::vector<int> ufAiVec;
    std::vector<float> ufAxVec;
    ufApVec.push_back(0);
    int ufNnzCount = 0;
    for (int l1 = LevelPtr[1]; l1 < LevelPtr[2]; l1++){
      int kBeginL2 = MixPtr[l1 * numKernels];
      int kEndL2 = MixPtr[l1 * numKernels + 1];
      for (int i1=kBeginL2; i1 < kEndL2; i1++){
//        int newRow = i1 - unfusedStart;
        int oldRow = Id[i1];
        for (int j = ap[oldRow]; j < ap[oldRow + 1]; j++){
          ufAiVec.push_back(ai[j]);
          ufAxVec.push_back(ax[j]);
          ufNnzCount += 1;
        }
        ufApVec.push_back(ufNnzCount);
      }
    }
    UFAp = new int[ufApVec.size()];
    UFAi = new int[ufAiVec.size()];
    UFAx = new float[ufAxVec.size()];
    UFAp[0] = 0;
    for (int i = 1; i < ufApVec.size(); i++){
      UFAp[i] = ufApVec[i];
      for (int j = UFAp[i-1]; j < UFAp[i]; j++){
        UFAx[j] = ufAxVec[j];
        UFAi[j] = ufAiVec[j];
      }
    }
  }

  Timer execute() override {
    float *intermediateResult = new float [InTensor->NumOfNodes*InTensor->EmbedDim];
    Timer t;
    St->OtherStats["FusedIterations"] = {(double)FusedCompSet->getNumberOfFusedNodes()};
    mkl_set_num_threads(1);
    OutTensor->reset();
    t.start();
    forwardForOneLayerFusedParallelSeparatedReorderedUnfusedVectorizedSP(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AMValues, UFAp, UFAi, UFAx,
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
  GCNSingleLayerSparseFusedReorderedUnfusedWithGeMM_SP(GnnTensorSpInputs *In1, Stats *Stat1,
                                               sym_lib::ScheduleParameters SpIn)
      : GCNSingleLayerUnFusedCSRMKLGeMMSP(In1, Stat1) {
    Inspector = new InspectorForAllFused(SpIn, Stat1);
  }
  ~GCNSingleLayerSparseFusedReorderedUnfusedWithGeMM_SP() {
    delete FusedCompSet;
    delete Inspector;
    delete[] UFAp;
    delete[] UFAi;
    delete[] UFAx;
  }
};

class GCNSingleLayerSparseFusedRedundantParallelWithGeMM_SP : public GCNSingleLayerUnFusedCSRMKLGeMMSP {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  InspectorForAllFused *Inspector;
  sym_lib::ScheduleParameters Sp;

  Timer analysis() override {
    Timer t;
    t.start();
    Sp._num_w_partition = std::max<int>(InTensor->AdjacencyMatrix->m / Sp.IterPerPartition,
                                        2 * Sp._num_threads);
    auto *sf01 = new sym_lib::SparseFusionWithRedundancy(&Sp, 2);
    auto *mvDAG = sym_lib::diagonal(InTensor->AdjacencyMatrix->m, 1.0);
    auto *tmpCSCCSR = new sym_lib::CSC(InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->n,
                                       InTensor->AdjacencyMatrix->nnz, InTensor->AdjacencyMatrix->p,
                                       InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x);
    auto *Di = InTensor->AdjacencyMatrix;
    // sym_lib::print_csc(1, "Di", 6, Di->p, Di->i, Di->x);
    sf01->fuse(1, mvDAG, tmpCSCCSR);

    // sf01->print_final_list();
    sf01->fuse(0, mvDAG, tmpCSCCSR);
    // sf01->print_final_list();
    auto pt = St->OtherStats["PackingType"];
    FusedCompSet = sf01->getFusedCompressed((int)pt[0]);
//    sf01->measureRedundancy(tmpCSCCSR, SpInfo);
    // FusedCompSet->print_3d();
    delete sf01;
    delete mvDAG;
    delete tmpCSCCSR;

    t.stop();
    return t;
  }

  Timer execute() override {
    auto *ws = new float[InTensor->NumThreads * 2 * InTensor->NumOfNodes * Sp.TileN]();
    Timer t;
    St->OtherStats["FusedIterations"] = {(double)FusedCompSet->getNumberOfFusedNodes()};
    mkl_set_num_threads(1);
    OutTensor->reset();
    t.start();
    forwardForOneLayerFusedParallelRedundantSp(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AMValues,
        InTensor->FeatureDim, InTensor->EmbedDim,
        InTensor->FeatureMatrix, InTensor->Weight1,
        OutTensor->FirstLayerOutput, ws, InTensor->NumThreads, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->ker_begin_);
    t.stop();
    delete[] ws;
    return t;
  }

public:
  GCNSingleLayerSparseFusedRedundantParallelWithGeMM_SP(GnnTensorSpInputs *In1, Stats *Stat1,
                                               sym_lib::ScheduleParameters SpIn)
      : GCNSingleLayerUnFusedCSRMKLGeMMSP(In1, Stat1), Sp(SpIn) {
    Inspector = new InspectorForAllFused(SpIn, Stat1);
  }
  ~GCNSingleLayerSparseFusedRedundantParallelWithGeMM_SP() {
    delete FusedCompSet;
    delete Inspector;
  }
};

class GCNSingleLayerLNRSP : public GCNSingleLayerMKL_SP {

protected:
  Timer execute() override {
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::compute_LNR(
        InTensor->NumOfNodes, InTensor->EmbedDim, OutTensor->FirstLayerOutput,
        InTensor->AdjacencyMatrix->p, InTensor->AdjacencyMatrix->i,
        InTensor->AMValues, InTensor->NumOfNodes, InTensor->FeatureDim,
        InTensor->FeatureMatrix, InTensor->FeatureDim,
        InTensor->EmbedDim, InTensor->Weight1, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerLNRSP(GnnTensorSpInputs *In1, Stats *Stat1)
      : GCNSingleLayerMKL_SP(In1, Stat1) {}
};

class GCNSingleLayerTaco : public GCNSingleLayerMKL_SP {

protected:
  Timer execute() override {
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::compute_TACO(
        InTensor->NumOfNodes, InTensor->EmbedDim, OutTensor->FirstLayerOutput,
        InTensor->AdjacencyMatrix->p, InTensor->AdjacencyMatrix->i,
        InTensor->AMValues, InTensor->NumOfNodes, InTensor->FeatureDim,
        InTensor->FeatureMatrix, InTensor->FeatureDim,
        InTensor->EmbedDim, InTensor->Weight1, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  GCNSingleLayerTaco(GnnTensorSpInputs *In1, Stats *Stat1)
      : GCNSingleLayerMKL_SP(In1, Stat1) {}
};

#endif // SPARSE_FUSION_GCN_SINGLE_LAYER_SP_DEMO_UTILS_H

//
// Created by mehdi on 6/28/23.
//
#include "GCN_Layer_MKL_Demo.h"
#include "SWTensorBench.h"
#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include "sparse-fusion/SparseFusion.h"
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <cassert>
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

// void vecMatMul(int M, int N, double *Vec, double *Mat, double *result) {
//   for (int j = 0; j < N; j++) {
//     result[j] = 0;
//     for (int i = 0; i < M; i++) {
//       result[j] += Vec[i] * Mat[i * N + j];
//     }
//   }
// }
//
// void aggregateMessage(int Dim, double *Messages, double *NeighborMessage) {
//   for (int i = 0; i < Dim; i++) {
//     Messages[i] += NeighborMessage[i];
//   }
// }
//
// void normalizeMessage(int Dim, double DegI, double DegJ,
//                       double *NeighborMessage) {
//   for (int i = 0; i < Dim; i++) {
//     NeighborMessage[i] = NeighborMessage[i] / sqrt(DegI * DegJ);
//   }
// }
//
// void forward(int M, int *Ap, int *Ai, int InputChannelDim, int
// OutputChannelDim,
//              double *Degrees, double *Features, double *Weight,
//              double *Output) {
//   double *neighborMessage = new double[OutputChannelDim];
//   for (int i = 0; i < M; i++) {
//     double *messages = Output + OutputChannelDim * i;
//     for (int j = Ap[i]; j < Ap[i + 1]; j++) {
//       int n = Ai[j];
//       vecMatMul(InputChannelDim, OutputChannelDim,
//                 Features + (n * InputChannelDim), Weight, neighborMessage);
//       normalizeMessage(OutputChannelDim, Degrees[i], Degrees[Ai[j]],
//                        neighborMessage);
//       aggregateMessage(OutputChannelDim, messages, neighborMessage);
//     }
//   }
//   delete[] neighborMessage;
// }

struct GnnTensorInputs : public Inputs<double> {
  double *Weight1, *Weight2;
  int *Degrees;
  sym_lib::Dense *FeatureMatrix;
  sym_lib::CSR *AdjacencyMatrix;
  size_t EmbedDim, NumOfClasses;
  size_t NumOfNodes;
  size_t BatchSize;
  std::vector<std::set<int>> LayerMasks;
  std::vector<sym_lib::CSR *> LayerMaskedMatrices;

  void computeDegrees() {
    this->Degrees = new int[this->NumOfNodes];
    for (int i = 0; i < this->NumOfNodes; i++) {
      this->Degrees[i] +=
          this->AdjacencyMatrix->p[i + 1] - this->AdjacencyMatrix->p[i];
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
      for (int j = adjMtxP[node]; j < adjMtxP[node+1]; j++) {
        previousLayerMask.emplace(adjMtxIndex[j]);
      }
    }
    return previousLayerMask;
  }

  GnnTensorInputs(double *Weight1, double *Weight2,
                  sym_lib::Dense *FeatureMatrix, sym_lib::CSC *AdjMtxCSC,
                  size_t NumOfNodes, size_t EmbedDim, size_t NumOfClasses,
                  size_t BatchSize, int NumThreads1, int NumTrial1,
                  std::string ExpN)
      : Inputs<double>(NumTrial1, NumThreads1, ExpN), Weight1(Weight1),
        Weight2(Weight2), FeatureMatrix(FeatureMatrix), NumOfNodes(NumOfNodes),
        NumOfClasses(NumOfClasses), EmbedDim(EmbedDim), BatchSize(BatchSize) {
    this->CorrectSol = nullptr;
    this->AdjacencyMatrix = sym_lib::csc_to_csr(AdjMtxCSC);
    this->LayerMasks = generateLayerMasks();
    for (auto mask : LayerMasks) {
      LayerMaskedMatrices.emplace_back(
          this->generateMaskedMatrix(mask, this->AdjacencyMatrix));
    }
    this->computeDegrees();
  }

  ~GnnTensorInputs() {
    delete[] Weight1;
    delete[] Weight2;
    delete FeatureMatrix;
    delete AdjacencyMatrix;
    delete LayerMaskedMatrices[0];
    delete LayerMaskedMatrices[1];
    delete Degrees;
  }
};

struct GnnTensorOutputs : public Outputs<double> {
  double *FirstLayerOutput, *SecondLayerOutput;
  size_t EmbedDim, NumOfClasses, NumOfNodes;

  GnnTensorOutputs(size_t EmbedDim, size_t NumOfClasses, size_t NumOfNodes)
      : NumOfNodes(NumOfNodes), EmbedDim(EmbedDim), NumOfClasses(NumOfClasses) {
    this->FirstLayerOutput = new double[NumOfNodes * EmbedDim]{};
    this->SecondLayerOutput = new double[NumOfNodes * NumOfClasses]{};
  }
  ~GnnTensorOutputs() {
    delete[] FirstLayerOutput;
    delete[] SecondLayerOutput;
  }

  void reset() {
    std::fill_n(FirstLayerOutput, EmbedDim * NumOfNodes, 0.0);
    std::fill_n(SecondLayerOutput, NumOfNodes * NumOfClasses, 0.0);
  }
};

class GCNSequential : public SWTensorBench<double> {
protected:
  GnnTensorInputs *InTensor;
  void setup() override {
    this->St->OtherStats["Number of Sampled Nodes"] = {double(InTensor->LayerMasks[1].size())};
    this->St->OtherStats["Number of First Layer Nodes"] = {double(InTensor->LayerMasks[0].size())};
    this->St->OtherStats["Number of Fused Nodes"] = {0.};
  }

  bool verify(double &Error) override {
    bool retValue = true;
    if (In->CorrectSol == nullptr)
      return true;
    double infNorm = 0;
    for (int i = 0; i < InTensor->NumOfNodes * InTensor->NumOfClasses; ++i) {
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
    Timer t;
    t.start();
    forwardForOneLayer(InTensor->LayerMaskedMatrices[0]->m,
                       InTensor->LayerMaskedMatrices[0]->p,
                       InTensor->LayerMaskedMatrices[0]->i,
                       InTensor->FeatureMatrix->col, InTensor->EmbedDim,
                       InTensor->Degrees, InTensor->FeatureMatrix->a,
                       InTensor->Weight1, OutTensor->FirstLayerOutput);
    forwardForOneLayer(InTensor->LayerMaskedMatrices[1]->m,
                       InTensor->LayerMaskedMatrices[1]->p,
                       InTensor->LayerMaskedMatrices[1]->i, InTensor->EmbedDim,
                       InTensor->NumOfClasses, InTensor->Degrees,
                       OutTensor->FirstLayerOutput, InTensor->Weight2,
                       OutTensor->SecondLayerOutput);
    t.stop();
    return t;
  }

public:
  GnnTensorOutputs *OutTensor;
  GCNSequential(GnnTensorInputs *In1, Stats *Stat1)
      : SWTensorBench<double>(In1, Stat1) {
    OutTensor =
        new GnnTensorOutputs(In1->EmbedDim, In1->NumOfClasses, In1->NumOfNodes);
    InTensor = In1;
  }

  ~GCNSequential() { delete OutTensor; }
};

class GCNParallel : public GCNSequential {
protected:

  Timer execute() override {
    Timer t;
    OutTensor->reset();
    t.start();
    forwardForOneLayerParallel(
        InTensor->LayerMaskedMatrices[0]->m,
        InTensor->LayerMaskedMatrices[0]->p,
        InTensor->LayerMaskedMatrices[0]->i, InTensor->FeatureMatrix->col,
        InTensor->EmbedDim, InTensor->Degrees, InTensor->FeatureMatrix->a,
        InTensor->Weight1, OutTensor->FirstLayerOutput, InTensor->NumThreads);
    forwardForOneLayerParallel(
        InTensor->LayerMaskedMatrices[1]->m,
        InTensor->LayerMaskedMatrices[1]->p,
        InTensor->LayerMaskedMatrices[1]->i, InTensor->EmbedDim,
        InTensor->NumOfClasses, InTensor->Degrees, OutTensor->FirstLayerOutput,
        InTensor->Weight2, OutTensor->SecondLayerOutput, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  GCNParallel(GnnTensorInputs *In1, Stats *Stat1) : GCNSequential(In1, Stat1) {}
};

class GCNFused : public GCNSequential {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;


  Timer analysis() override {
    Timer t;
    t.start();
    //    // sym_lib::ScheduleParameters sp;
    //    // sp._num_threads = InTensor->NumThreads;
    //    //  create the fused set
    sym_lib::CSC *Di1 = sym_lib::csr_to_csc(InTensor->LayerMaskedMatrices[0]);
    sym_lib::CSC *Di2 = sym_lib::csr_to_csc(InTensor->LayerMaskedMatrices[1]);
    Sp._num_w_partition =
        std::max<int>(InTensor->AdjacencyMatrix->m / Sp.IterPerPartition,
                      2 * Sp._num_threads);
    auto *sf01 = new sym_lib::SparseFusion(&Sp, 2);
    auto *mvDAG = sym_lib::diagonal(InTensor->AdjacencyMatrix->m, 1.0);
    auto *tmpCSCCSR = new sym_lib::CSC(
        InTensor->AdjacencyMatrix->m, InTensor->AdjacencyMatrix->n,
        InTensor->AdjacencyMatrix->nnz, InTensor->AdjacencyMatrix->p,
        InTensor->AdjacencyMatrix->i, InTensor->AdjacencyMatrix->x);
    auto *Di = InTensor->AdjacencyMatrix;
    // sym_lib::print_csc(1, "Di", 6, Di->p, Di->i, Di->x);
    sf01->fuse(0, mvDAG, Di1);

    // sf01->print_final_list();
    sf01->fuse(1, mvDAG, Di2);
    // sf01->print_final_list();
    auto pt = St->OtherStats["PackingType"];
    FusedCompSet = sf01->getFusedCompressed((int)pt[0]);
    //    FusedCompSet->print_3d();
    St->OtherStats["Number of Fused Nodes"] = {double(FusedCompSet->getNumberOfFusedNodes(InTensor->LayerMaskedMatrices[1]))};
    delete sf01;
    delete mvDAG;
    delete tmpCSCCSR;
    delete Di1;
    delete Di2;
//        FusedCompSet->print_3d();
    t.stop();
    return t;
  }

  Timer execute() override {
    Timer t;
    OutTensor->reset();
    t.start();
    forwardForFusedLayersParallelWithBatching(
        InTensor->LayerMaskedMatrices[0]->m,
        InTensor->LayerMaskedMatrices[0]->p,
        InTensor->LayerMaskedMatrices[0]->i,
        InTensor->LayerMaskedMatrices[1]->p,
        InTensor->LayerMaskedMatrices[1]->i, InTensor->FeatureMatrix->col,
        InTensor->EmbedDim, InTensor->NumOfClasses, InTensor->Degrees,
        InTensor->FeatureMatrix->a, InTensor->Weight1, InTensor->Weight2,
        OutTensor->SecondLayerOutput, OutTensor->FirstLayerOutput,
        InTensor->NumThreads, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_);
    t.stop();
    return t;
  }

public:
  GCNFused(GnnTensorInputs *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn)
      : GCNSequential(In1, Stat1), Sp(SpIn) {}
  ~GCNFused() { delete FusedCompSet; }
};

class GCNFusedParallelWithOmittingEmptyRows : public GCNSequential {
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
        InTensor->LayerMaskedMatrices[1]->p,
        InTensor->LayerMaskedMatrices[1]->i, InTensor->FeatureMatrix->col,
        InTensor->EmbedDim, InTensor->NumOfClasses, InTensor->Degrees,
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
      : GCNSequential(In1, Stat1), Sp(SpIn) {}
  ~GCNFusedParallelWithOmittingEmptyRows() { delete FusedCompSet; }
};

class GCNFusedWithOmittingEmptyRows : public GCNSequential {
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
    int allNodesNum = InTensor->LayerMasks[0].size() +
                      InTensor->LayerMasks[1].size();
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
    for (int i = 0; i < l1->m; i+=TileSize) {
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
    for (int i = 0; i < l2->m; i++){
      if (l2->p[i] == l2->p[i+1] || fusedNodes.find(i) != fusedNodes.end()){
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
        InTensor->LayerMaskedMatrices[1]->p,
        InTensor->LayerMaskedMatrices[1]->i, InTensor->FeatureMatrix->col,
        InTensor->EmbedDim, InTensor->NumOfClasses, InTensor->Degrees,
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
      : GCNSequential(In1, Stat1), Sp(SpIn), TileSize(TileSize1) {}
  ~GCNFusedWithOmittingEmptyRows() { delete FusedCompSet; }
};
//
// Created by salehm32 on 08/12/23.
//

#ifndef SPARSE_FUSION_SPMV_SPMV_MKL_DEMO_UTILS_H
#define SPARSE_FUSION_SPMV_SPMV_MKL_DEMO_UTILS_H
#include "../gcn/Inspection/Fusion_Inspector.h"
#include "SWTensorBench.h"
#include "SpMV_SpMV.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Defs.h"
#include "sparse-fusion/SparseFusion.h"

using namespace swiftware::benchmark;

template <typename T> void printDense(int M, int N, T *X) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << X[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T> struct TensorInputs : public Inputs<T> {
  int M, K, L;
  sym_lib::CSC *A, *B;
  sym_lib::CSR *ACsr, *BCsr;

  T *Cx;
  T *CorrectMul;
  bool IsSolProvided;

  TensorInputs(int M1, int N1, int K1, int L1, sym_lib::CSC *A1,
               sym_lib::CSC *B1, int NumThreads1, int NumTrial1,
               std::string ExpN)
      : Inputs<T>(NumTrial1, NumThreads1, ExpN) {
    M = M1;
    K = K1;
    L = L1;
    A = sym_lib::copy_sparse(A1);
    B = sym_lib::copy_sparse(B1);
    ACsr = sym_lib::csc_to_csr(A);
    BCsr = sym_lib::csc_to_csr(B);
    Cx = new double[K]();
    // randomly initialize the input
    for (int i = 0; i < K; ++i) {
      Cx[i] = 1.0; //(double)rand()/RAND_MAX;
    }
    CorrectMul = new double[L](); // the correct solution
    IsSolProvided = false;
    Inputs<T>::Threshold = 1e-6;
  }

  ~TensorInputs() {
    delete[] Cx;
    delete[] CorrectMul;
    delete A;
    delete B;
    delete ACsr;
    delete BCsr;
  }
};

template <typename T> struct TensorOutputs : public Outputs<T> {
  int M, L;
  T *Dx, *ACx;

  TensorOutputs(int M, int L) : M(M), L(L) {
    Dx = new T[L]();
    ACx = new T[M]();
  }

  ~TensorOutputs() {
    delete[] Dx;
    delete[] ACx;
  }

  void printDx() {
    std::cout << "\n ACx:\n";
    printDense<T>(M, ACx);
    std::cout << "\n Dx:\n";
    printDense<T>(L, Dx);
    std::cout << "\n";
  }

  void reset() {
    std::fill_n(Dx, L, 0.0);
    std::fill_n(ACx, M, 0.0);
  }
};

class SpMVSpMVUnFusedSequential : public SWTensorBench<double> {
protected:
  TensorInputs<double> *InTensor;

  void setup() override {
    St->OtherStats["PackingType"] = {sym_lib::Separated};
    St->OtherStats["FusedIterations"] = {0.};
    St->OtherStats["Min Workload Size"] = {10};
  }

  void preExecute() override {}

  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spMVCsrSequential(InTensor->M, InTensor->K, InTensor->ACsr->p,
                      InTensor->ACsr->i, InTensor->ACsr->x, InTensor->Cx,
                      OutTensor->ACx);
    spMVCsrSequential(InTensor->L, InTensor->M, InTensor->BCsr->p,
                      InTensor->BCsr->i, InTensor->BCsr->x, OutTensor->ACx,
                      OutTensor->Dx);
    t.stop();
    return t;
  }

  bool verify(double &Error) override {
    bool retValue = true;
    if (!InTensor->IsSolProvided) {
      Error = 0;
      return true;
    }
    double infNorm = 0;
    for (int i = 0; i < InTensor->L; ++i) {
      if (std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]) > infNorm) {
        infNorm = std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]);
      }
    }
    Error = (double)infNorm;
    if (infNorm > InTensor->Threshold) {
      retValue = false;
    }
    return retValue;
  }

public:
  TensorOutputs<double> *OutTensor;
  SpMVSpMVUnFusedSequential(TensorInputs<double> *In1, Stats *Stat1)
      : SWTensorBench<double>(In1, Stat1) {
    OutTensor = new TensorOutputs<double>(In1->M, In1->L);
    InTensor = In1;
  }

  ~SpMVSpMVUnFusedSequential() { delete OutTensor; }
};

class SpMVSpMVUnFusedSegmentedSumSequential : public SpMVSpMVUnFusedSequential {

protected:
  double *WorkSpace;
  Timer execute() override {
    std::fill_n(WorkSpace, InTensor->ACsr->nnz, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spMVCsrSequentialSegmentedSum(InTensor->M, InTensor->K, InTensor->ACsr->p,
                                  InTensor->ACsr->i, InTensor->ACsr->x,
                                  InTensor->Cx, OutTensor->ACx, WorkSpace);
    spMVCsrSequentialSegmentedSum(InTensor->L, InTensor->M, InTensor->BCsr->p,
                                  InTensor->BCsr->i, InTensor->BCsr->x,
                                  OutTensor->ACx, OutTensor->Dx, WorkSpace);
    t.stop();
    return t;
  }
public:
  SpMVSpMVUnFusedSegmentedSumSequential(TensorInputs<double> *In1,
                                        Stats *Stat1)
      : SpMVSpMVUnFusedSequential(In1, Stat1) {
    WorkSpace = new double[InTensor->ACsr->nnz]();
  }

  ~SpMVSpMVUnFusedSegmentedSumSequential() { delete[] WorkSpace; }
};

class SpMVSpMVUnFusedSegmentedSumParallel : public SpMVSpMVUnFusedSegmentedSumSequential {
  protected:
  Timer execute() override {
    std::fill_n(WorkSpace, InTensor->ACsr->nnz, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spMVCsrSegmentedSumParallel(InTensor->M, InTensor->K, InTensor->ACsr->p,
                                InTensor->ACsr->i, InTensor->ACsr->x,
                                InTensor->Cx, OutTensor->ACx, WorkSpace,
                                InTensor->NumThreads);
    spMVCsrSegmentedSumParallel(InTensor->L, InTensor->M, InTensor->BCsr->p,
                                InTensor->BCsr->i, InTensor->BCsr->x,
                                OutTensor->ACx, OutTensor->Dx, WorkSpace,
                                InTensor->NumThreads);
    t.stop();
    return t;
  }

  public:
  SpMVSpMVUnFusedSegmentedSumParallel(TensorInputs<double> *In1,
                                      Stats *Stat1)
      : SpMVSpMVUnFusedSegmentedSumSequential(In1, Stat1) {}
};

class SpMVSpMVUnFusedParallel : public SpMVSpMVUnFusedSequential {
protected:
  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spMVCsrParallel(InTensor->M, InTensor->K, InTensor->ACsr->p,
                    InTensor->ACsr->i, InTensor->ACsr->x, InTensor->Cx,
                    OutTensor->ACx, InTensor->NumThreads);
    spMVCsrParallel(InTensor->L, InTensor->M, InTensor->BCsr->p,
                    InTensor->BCsr->i, InTensor->BCsr->x, OutTensor->ACx,
                    OutTensor->Dx, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMVSpMVUnFusedParallel(TensorInputs<double> *In1, Stats *Stat1)
      : SpMVSpMVUnFusedSequential(In1, Stat1) {}
};

class SpMVSpMVFused : public SpMVSpMVUnFusedSequential {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  Timer analysis() override {
    Timer t;
    t.start();
    // sym_lib::ScheduleParameters sp;
    // sp._num_threads = InTensor->NumThreads;
    //  create the fused set

    Sp._num_w_partition = std::max<int>(InTensor->ACsr->m / Sp.IterPerPartition,
                                        2 * Sp._num_threads);
    auto *sf01 = new sym_lib::SparseFusion(&Sp, 2);
    auto *mvDAG = sym_lib::diagonal(InTensor->ACsr->m, 1.0);
    auto *tmpCSCCSR = new sym_lib::CSC(InTensor->BCsr->m, InTensor->BCsr->n,
                                       InTensor->BCsr->nnz, InTensor->BCsr->p,
                                       InTensor->BCsr->i, InTensor->BCsr->x);
    auto *Di = InTensor->BCsr;
    // sym_lib::print_csc(1, "Di", 6, Di->p, Di->i, Di->x);
    sf01->fuse(0, mvDAG, tmpCSCCSR);

    // sf01->print_final_list();
    sf01->fuse(1, mvDAG, tmpCSCCSR);
    // sf01->print_final_list();
    auto pt = St->OtherStats["PackingType"];
    FusedCompSet = sf01->getFusedCompressed((int)pt[0]);
    // FusedCompSet->print_3d();
    delete sf01;
    delete mvDAG;
    delete tmpCSCCSR;

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spMVCsrSpMCsrFused(InTensor->M, InTensor->K, InTensor->L, InTensor->ACsr->p,
                       InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
                       InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Cx,
                       OutTensor->Dx, OutTensor->ACx, FusedCompSet->n1_,
                       FusedCompSet->ptr1_, FusedCompSet->ptr2_,
                       FusedCompSet->id_, FusedCompSet->type_,
                       InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMVSpMVFused(TensorInputs<double> *In1, Stats *Stat1,
                sym_lib::ScheduleParameters SpIn)
      : SpMVSpMVUnFusedSequential(In1, Stat1), Sp(SpIn) {}

  ~SpMVSpMVFused() { delete FusedCompSet; }
};

class SpMVSpMVFusedInterleaved : public SpMVSpMVUnFusedSequential {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  Timer analysis() override {
    Timer t;
    t.start();
    // sym_lib::ScheduleParameters sp;
    // sp._num_threads = InTensor->NumThreads;
    //  create the fused set
    FusedCompSet = new sym_lib::MultiDimensionalSet();
    FusedCompSet->n1_ = 2;
    FusedCompSet->ptr1_ = new int[3];
    FusedCompSet->ptr1_[0] = 0;
    FusedCompSet->ptr1_[1] = InTensor->NumThreads * 3;
    FusedCompSet->ptr1_[2] = InTensor->NumThreads * 2 * 3;
    FusedCompSet->ptr2_ = new int[InTensor->NumThreads * 6 + 1];
    FusedCompSet->ptr2_[0] = 0;
    FusedCompSet->id_ = new int[InTensor->M * 2];
    int iterPerPartition = InTensor->M / InTensor->NumThreads;
    int idCtr = 0;
    FusedCompSet->id_[0] = 0;
    for (int i = 0; i < InTensor->M; i += iterPerPartition) {
      for (int ii = 0; ii < iterPerPartition; ii++) {
        if ((i == 0 && ii > 0) || (i > 0 && ii > 1)) {
          FusedCompSet->id_[idCtr] = i + ii;
          idCtr++;
          FusedCompSet->id_[idCtr] = i + ii - 1;
          idCtr++;
          FusedCompSet->ptr2_[(i / iterPerPartition) * 3 + 2] = idCtr;
        } else {
          FusedCompSet->id_[idCtr] = i + ii;
          idCtr++;
          FusedCompSet->ptr2_[(i / iterPerPartition) * 3 + 1] = idCtr;
        }
      }
      FusedCompSet->ptr2_[(i / iterPerPartition) * 3 + 3] = idCtr;
    }
    for (int i = iterPerPartition; i < InTensor->M; i += iterPerPartition) {
      FusedCompSet->ptr2_[InTensor->NumThreads * 3 +
                          (i / iterPerPartition - 1) * 3 + 1] = idCtr;
      FusedCompSet->ptr2_[InTensor->NumThreads * 3 +
                          (i / iterPerPartition - 1) * 3 + 2] = idCtr;
      FusedCompSet->id_[idCtr] = i - 1;
      idCtr++;
      FusedCompSet->id_[idCtr] = i;
      idCtr++;
      FusedCompSet->ptr2_[InTensor->NumThreads * 3 +
                          (i / iterPerPartition - 1) * 3 + 3] = idCtr;
    }
    FusedCompSet->ptr2_[((InTensor->NumThreads * 2) - 1) * 3 + 1] = idCtr;
    FusedCompSet->ptr2_[((InTensor->NumThreads * 2) - 1) * 3 + 2] = idCtr;
    FusedCompSet->id_[idCtr] = InTensor->M - 1;
    FusedCompSet->ptr2_[((InTensor->NumThreads * 2) - 1) * 3 + 3] = idCtr + 1;
    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spMVCsrSpMCsrFusedRegisterReuseBanded(
        InTensor->M, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Cx, OutTensor->Dx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMVSpMVFusedInterleaved(TensorInputs<double> *In1, Stats *Stat1,
                           sym_lib::ScheduleParameters SpIn)
      : SpMVSpMVUnFusedSequential(In1, Stat1), Sp(SpIn) {}

  ~SpMVSpMVFusedInterleaved() { delete FusedCompSet; }
};

class SpMVSpMVFusedParallelSeparated : public SpMVSpMVFused {
protected:
  Timer execute() override {
    OutTensor->reset();
    St->OtherStats["FusedIterations"] = {
        (double)FusedCompSet->getNumberOfFusedNodes()};
    Timer t;
    t.start();
    spMVCsrSpMVCsrSeparatedFused(
        InTensor->M, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Cx, OutTensor->Dx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        FusedCompSet->ker_begin_, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMVSpMVFusedParallelSeparated(TensorInputs<double> *In1, Stats *Stat1,
                                 sym_lib::ScheduleParameters SpIn)
      : SpMVSpMVFused(In1, Stat1, SpIn) {}
};

class SpMVCSRSpMVCSCFusedColoring : public SpMVSpMVUnFusedSequential {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  InspectorForSingleLayerTiledFusedCSCParallel *Inspector;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  int TileSize;
  Timer analysis() override {
    Timer t;
    t.start();

    FusedCompSet =
        Inspector->generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            ConflictGraphColoring, InTensor->M, TileSize);

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spMVCsrSpMVCscFusedColored(
        InTensor->M, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->B->p, InTensor->B->i,
        InTensor->B->x, InTensor->Cx, OutTensor->Dx, OutTensor->ACx,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_, Sp.TileM,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMVCSRSpMVCSCFusedColoring(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn,
      int TileSize1, std::map<int, std::vector<int>> ConflictGraphColoring1)
      : SpMVSpMVUnFusedSequential(In1, Stat1), Sp(SpIn), TileSize(TileSize1),
        ConflictGraphColoring(ConflictGraphColoring1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }

  ~SpMVCSRSpMVCSCFusedColoring() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};

class SpMVCSRSpMVCSCPartialFusedColoring : public SpMVSpMVUnFusedSequential {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  InspectorForSingleLayerTiledFusedCSCCombined *Inspector;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  int MinWorkloadSize;
  int TileSize;
  Timer analysis() override {
    Timer t;
    t.start();

    FusedCompSet =
        Inspector->generateScheduleBasedOnConflictGraphColoring(
            ConflictGraphColoring, InTensor->M, TileSize, MinWorkloadSize);

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spmvCsrSpmvCscPartialFusedColored(
        InTensor->M, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->B->p, InTensor->B->i,
        InTensor->B->x, InTensor->Cx, OutTensor->Dx, OutTensor->ACx,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        FusedCompSet->type_, FusedCompSet->n3_, FusedCompSet->n2_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMVCSRSpMVCSCPartialFusedColoring(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn,
      int TileSize1, std::map<int, std::vector<int>> ConflictGraphColoring1, int MinWorkloadSize1)
      : SpMVSpMVUnFusedSequential(In1, Stat1), Sp(SpIn), TileSize(TileSize1),
        MinWorkloadSize(MinWorkloadSize1), ConflictGraphColoring(ConflictGraphColoring1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCCombined();
  }

  ~SpMVCSRSpMVCSCPartialFusedColoring() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};

class SpMVCSRSpMVCSCFusedColoringWithReduction
    : public SpMVSpMVUnFusedSequential {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  InspectorForSingleLayerTiledFusedCSCParallel *Inspector;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  int TileSize;
  Timer analysis() override {
    Timer t;
    t.start();

    FusedCompSet =
        Inspector->generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            ConflictGraphColoring, InTensor->M, TileSize);

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spMVCsrSpMVCscFusedColoredWithReduction(
        InTensor->M, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->B->p, InTensor->B->i,
        InTensor->B->x, InTensor->Cx, OutTensor->Dx, OutTensor->ACx,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_, Sp.TileM,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMVCSRSpMVCSCFusedColoringWithReduction(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn,
      int TileSize1, std::map<int, std::vector<int>> ConflictGraphColoring1)
      : SpMVSpMVUnFusedSequential(In1, Stat1), Sp(SpIn), TileSize(TileSize1),
        ConflictGraphColoring(ConflictGraphColoring1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }

  ~SpMVCSRSpMVCSCFusedColoringWithReduction() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};

class SpMVSpMVFusedTiledTri : public SpMVSpMVFused {

  void buildBandedFusedSchedule(
      std::vector<std::vector<sym_lib::FusedNode *>> &FusedSchedule,
      int BandWidth) {
    // one h-level
    FusedSchedule.resize(1);
    int hintTotLoops = 2, loopId = 0;
    int n = InTensor->ACsr->m;
    // create a list of consecutive integers to n
    std::vector<int> seqNode(n);
    for (int i = 0; i < n; ++i) {
      seqNode[i] = i;
    }
    int nParts = Sp._num_w_partition;
    // FusedSchedule[0].resize(nParts);
    int i = 0, j = 0;
    int halfBand = (BandWidth - 1) / 2;
    for (i = 0, j = 0; i < n; i += Sp.IterPerPartition, j++) {
      // iterations of first spmm
      int begin = std::max(0, i - halfBand),
          end = std::min(i + Sp.IterPerPartition + halfBand, n),
          nIters = end - begin;
      auto *curFn = new sym_lib::FusedNode(hintTotLoops, loopId, nIters,
                                           seqNode.data() + begin, j);
      // iterations of second spmm
      if (i + Sp.IterPerPartition > n)
        break;
      curFn->_list[1].resize(Sp.IterPerPartition);
      for (int k = i, kk = 0; k < i + Sp.IterPerPartition; ++k, ++kk) {
        curFn->_list[1][kk] = k;
      }
      FusedSchedule[0].emplace_back(curFn);
    }

    if (i < n) {
      int begin = std::max(0, i - BandWidth + 1), end = n, nIters = end - begin;
      auto *curFn = new sym_lib::FusedNode(hintTotLoops, loopId, nIters,
                                           seqNode.data() + begin, j);

      curFn->_list[1].resize(n - i);
      for (int k = i, kk = 0; k < n; ++k, ++kk) {
        curFn->_list[1][kk] = k;
      }
      FusedSchedule[0].emplace_back(curFn);
    }
  }

  Timer analysis() override {
    Timer t;
    t.start();
    // sym_lib::ScheduleParameters sp;
    // sp._num_threads = InTensor->NumThreads;
    //  create the fused set

    int numParts = std::max<int>(InTensor->ACsr->m / Sp.IterPerPartition,
                                 2 * Sp._num_threads);
    Sp._num_w_partition = numParts;
    std::vector<std::vector<sym_lib::FusedNode *>> fusedSchedule;
    int band = 2 * (InTensor->ACsr->p[1] - InTensor->ACsr->p[0] - 1) + 1;
    buildBandedFusedSchedule(fusedSchedule, band);
    auto pt = St->OtherStats["PackingType"];
    FusedCompSet = new sym_lib::MultiDimensionalSet(fusedSchedule, (int)pt[0]);
    FusedCompSet->n3_ = band - 1;
    //    FusedCompSet->print_3d();

    t.stop();
    return t;
  }
  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    auto *ws = new double[InTensor->NumThreads *
                          (Sp.IterPerPartition + FusedCompSet->n3_)]();
    OutTensor->reset();
    Timer t;
    t.start();
    spmvCsrSpmvCsrTiledFusedRedundantBanded(
        InTensor->M, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Cx, OutTensor->Dx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        FusedCompSet->ker_begin_, InTensor->NumThreads, Sp.IterPerPartition, ws,
        FusedCompSet->n3_);

    t.stop();
    delete[] ws;
    return t;
  }

public:
  SpMVSpMVFusedTiledTri(TensorInputs<double> *In1, Stats *Stat1,
                        sym_lib::ScheduleParameters SpIn)
      : SpMVSpMVFused(In1, Stat1, SpIn) {}
  ~SpMVSpMVFusedTiledTri() {}
};

class SpMVSpMVFusedTiledRedundant : public SpMVSpMVFused {

  Timer analysis() override {
    Timer t;
    t.start();
    int mTilesNum = std::ceil((double)InTensor->ACsr->m / Sp.TileM);
    FusedCompSet = new sym_lib::MultiDimensionalSet();
    FusedCompSet->ptr1_ = new int[mTilesNum];
    FusedCompSet->ptr2_ = new int[mTilesNum];
    int maxL1TileSize = 0;
    for (int i = 0; i < InTensor->ACsr->m; i += Sp.TileM) {
      FusedCompSet->ptr1_[i / Sp.TileM] = InTensor->ACsr->m;
      FusedCompSet->ptr2_[i / Sp.TileM] = 0;
      for (int ii = 0; ii < Sp.TileM; ++ii) {
        if (i + ii < InTensor->ACsr->m) {
          int lowBound = InTensor->ACsr->i[InTensor->ACsr->p[i + ii]];
          int highBound =
              InTensor->ACsr->i[InTensor->ACsr->p[i + ii + 1] - 1] + 1;
          if (lowBound < FusedCompSet->ptr1_[i / Sp.TileM]) {
            FusedCompSet->ptr1_[i / Sp.TileM] = lowBound;
          }
          if (highBound > FusedCompSet->ptr2_[i / Sp.TileM]) {
            FusedCompSet->ptr2_[i / Sp.TileM] = highBound;
          }
        }
      }
      if (FusedCompSet->ptr2_[i / Sp.TileM] -
              FusedCompSet->ptr1_[i / Sp.TileM] >
          maxL1TileSize) {
        maxL1TileSize = FusedCompSet->ptr2_[i / Sp.TileM] -
                        FusedCompSet->ptr1_[i / Sp.TileM];
      }
    }
    FusedCompSet->n1_ = maxL1TileSize;
    t.stop();
    return t;
  }
  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    auto *ws = new double[InTensor->NumThreads * FusedCompSet->n1_]();
    OutTensor->reset();
    Timer t;
    t.start();
    spmvCsrSpmvCsrTiledFusedRedundantGeneral(
        InTensor->M, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Cx, OutTensor->Dx,
        InTensor->NumThreads, Sp.TileM, ws, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->n1_);

    t.stop();
    delete[] ws;
    return t;
  }

public:
  SpMVSpMVFusedTiledRedundant(TensorInputs<double> *In1, Stats *Stat1,
                              sym_lib::ScheduleParameters SpIn)
      : SpMVSpMVFused(In1, Stat1, SpIn) {}
  ~SpMVSpMVFusedTiledRedundant() {}
};

#ifdef MKL

#include <mkl.h>
class SpMVSpMVMkl : public SpMVSpMVUnFusedSequential {
protected:
  sparse_matrix_t A;
  sparse_matrix_t B;
  MKL_INT *LLI_A;
  MKL_INT *LLI_B;
  matrix_descr d;
  Timer execute() override {
    Timer t;
    t.start();
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, this->A, this->d,
                    this->InTensor->Cx, 0, this->OutTensor->ACx);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, this->B, this->d,
                    this->OutTensor->ACx, 0, this->OutTensor->Dx);
    t.stop();
    return t;
  }

public:
  SpMVSpMVMkl(TensorInputs<double> *In1, Stats *Stat1)
      : SpMVSpMVUnFusedSequential(In1, Stat1) {
    d.type = SPARSE_MATRIX_TYPE_GENERAL;

    LLI_A = new MKL_INT[this->InTensor->M + 1]();
    for (int l = 0; l < this->InTensor->M + 1; ++l) {
      LLI_A[l] = this->InTensor->ACsr->p[l];
    }

    LLI_B = new MKL_INT[this->InTensor->L + 1]();
    for (int l = 0; l < this->InTensor->L + 1; ++l) {
      LLI_B[l] = this->InTensor->BCsr->p[l];
    }

    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, this->InTensor->M,
                            this->InTensor->K, LLI_A, LLI_A + 1,
                            this->InTensor->ACsr->i, this->InTensor->ACsr->x);
    mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO, this->InTensor->L,
                            this->InTensor->M, LLI_B, LLI_B + 1,
                            this->InTensor->BCsr->i, this->InTensor->BCsr->x);
    mkl_set_num_threads(this->InTensor->NumThreads);
  }

  ~SpMVSpMVMkl() {
    mkl_free(A);
    mkl_free(B);
  }
};
#endif // MKL
#endif // SPARSE_FUSION_SPMV_SPMV_MKL_DEMO_UTILS_H

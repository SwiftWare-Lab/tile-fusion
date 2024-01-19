//
// Created by kazem on 08/05/23.
//

#ifndef SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H
#define SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H

#include "Inspection/Fusion_Inspector.h"
#include "SWTensorBench.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include "sparse-fusion/SpMM_SpMM.h"
#include "sparse-fusion/SparseFusion.h"
#include "sparse-fusion/SparseFusionWithRedundancy.h"
#include <omp.h>

using namespace swiftware::benchmark;

// print a dense matrix with dimension MxN
template <typename T> void printDense(int M, int N, T *X) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << X[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T> struct TensorInputs : public Inputs<T> {
  int M, N, K, L;
  sym_lib::CSC *A, *B;
  sym_lib::CSR *ACsr, *BCsr;

  T *Bx;
  T *CorrectSol;
  bool IsSolProvided;

  TensorInputs(int M1, int N1, int K1, int L1, sym_lib::CSC *A1,
               sym_lib::CSC *B1, int NumThreads1, int NumTrial1,
               std::string ExpN)
      : Inputs<T>(NumTrial1, NumThreads1, ExpN) {
    M = M1;
    N = N1;
    K = K1;
    L = L1;
    A = sym_lib::copy_sparse(A1);
    B = sym_lib::copy_sparse(B1);
    ACsr = sym_lib::csc_to_csr(A);
    BCsr = sym_lib::csc_to_csr(B);
    Bx = new double[K * N]();
    // randomly initialize the input
    for (int i = 0; i < K * N; ++i) {
      Bx[i] = 1.0; //(double)rand()/RAND_MAX;
    }
    CorrectSol = new double[L * N](); // the correct solution
    IsSolProvided = false;
    Inputs<T>::Threshold = 1e-6;
  }

  ~TensorInputs() {
    delete[] Bx;
    delete[] CorrectSol;
    delete A;
    delete B;
    delete ACsr;
    delete BCsr;
  }
};

template <typename T> struct TensorOutputs : public Outputs<T> {
  int M, N, L;
  T *Xx, *ACx;

  TensorOutputs(int M, int N, int L) : M(M), N(N), L(L) {
    Xx = new T[L * N]();
    ACx = new T[M * N]();
  }

  ~TensorOutputs() {
    delete[] Xx;
    delete[] ACx;
  }

  void printDx() {
    std::cout << "\n ACx:\n";
    printDense<T>(M, N, ACx);
    std::cout << "\n Xx:\n";
    printDense<T>(L, N, Xx);
    std::cout << "\n";
  }

  void reset() {
    std::fill_n(Xx, L * N, 0.0);
    std::fill_n(ACx, M * N, 0.0);
  }
};

class SpMMSpMMUnFused : public SWTensorBench<double> {
protected:
  TensorInputs<double> *InTensor;

  void setup() override {
    this->St->OtherStats["NTile"] = {4};
  }

  void preExecute() override {}

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSequential(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->Bx, OutTensor->ACx);
    swiftware::sparse::spmmCsrSequential(
        InTensor->L, InTensor->N, InTensor->M, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, OutTensor->ACx, OutTensor->Xx);
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
    for (int i = 0; i < InTensor->L * InTensor->N; ++i) {
      if (std::abs(OutTensor->Xx[i] - InTensor->CorrectSol[i]) > infNorm) {
        infNorm = std::abs(OutTensor->Xx[i] - InTensor->CorrectSol[i]);
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
  SpMMSpMMUnFused(TensorInputs<double> *In1, Stats *Stat1)
      : SWTensorBench<double>(In1, Stat1) {
    OutTensor = new TensorOutputs<double>(In1->M, In1->N, In1->L);
    InTensor = In1;
  }

  ~SpMMSpMMUnFused() { delete OutTensor; }
};

class SpMMSpMMUnFusedParallelTiled : public SpMMSpMMUnFused {
protected:
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrParallel(InTensor->M, InTensor->N, InTensor->K,
                                       InTensor->ACsr->p, InTensor->ACsr->i,
                                       InTensor->ACsr->x, InTensor->Bx,
                                       OutTensor->ACx, InTensor->NumThreads);
    swiftware::sparse::spmmCsrParallel(InTensor->L, InTensor->N, InTensor->M,
                                       InTensor->BCsr->p, InTensor->BCsr->i,
                                       InTensor->BCsr->x, OutTensor->ACx,
                                       OutTensor->Xx, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMSpMMUnFusedParallelTiled(TensorInputs<double> *In1, Stats *Stat1)
      : SpMMSpMMUnFused(In1, Stat1) {}
};

class SpMMSpMMUnFusedParallel : public SpMMSpMMUnFused {
protected:
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrParallel(InTensor->M, InTensor->N, InTensor->K,
                                       InTensor->ACsr->p, InTensor->ACsr->i,
                                       InTensor->ACsr->x, InTensor->Bx,
                                       OutTensor->ACx, InTensor->NumThreads);
    swiftware::sparse::spmmCsrParallel(InTensor->L, InTensor->N, InTensor->M,
                                       InTensor->BCsr->p, InTensor->BCsr->i,
                                       InTensor->BCsr->x, OutTensor->ACx,
                                       OutTensor->Xx, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMSpMMUnFusedParallel(TensorInputs<double> *In1, Stats *Stat1)
      : SpMMSpMMUnFused(In1, Stat1) {}
};

class SpMMSpMMUnFusedInnerParallel : public SpMMSpMMUnFused {
protected:
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrInnerProductParallel(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->Bx, OutTensor->ACx,
        InTensor->NumThreads);
    swiftware::sparse::spmmCsrInnerProductParallel(
        InTensor->L, InTensor->N, InTensor->M, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, OutTensor->ACx, OutTensor->Xx,
        InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMSpMMUnFusedInnerParallel(TensorInputs<double> *In1, Stats *Stat1)
      : SpMMSpMMUnFused(In1, Stat1) {}
};

class SpMMSpMMUnFusedCTiledParallel : public SpMMSpMMUnFused {
protected:
  sym_lib::ScheduleParameters Sp;
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    int tileM = Sp.TileM, tileN = Sp.TileN;

    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrParallelTiled(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->Bx, OutTensor->ACx,
        InTensor->NumThreads, tileM, tileN);
    swiftware::sparse::spmmCsrParallelTiled(
        InTensor->L, InTensor->N, InTensor->M, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, OutTensor->ACx, OutTensor->Xx,
        InTensor->NumThreads, tileM, tileN);
    t.stop();
    return t;
  }

public:
  SpMMSpMMUnFusedCTiledParallel(TensorInputs<double> *In1, Stats *Stat1,
                                sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn) {}
};

class SpMMSpMMFusedInterLayer : public SpMMSpMMUnFused {
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
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrFused(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMSpMMFusedInterLayer(TensorInputs<double> *In1, Stats *Stat1,
                          sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn) {}

  ~SpMMSpMMFusedInterLayer() { delete FusedCompSet; }
};

class SpMMSpMMFusedInterLayerKTiled : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  int KTileSize;
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
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrFusedKTiled(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, KTileSize, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMSpMMFusedInterLayerKTiled(TensorInputs<double> *In1, Stats *Stat1,
                          sym_lib::ScheduleParameters SpIn, int KTileSize1)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn), KTileSize(KTileSize1) {}

  ~SpMMSpMMFusedInterLayerKTiled() { delete FusedCompSet; }
};

class SpMMSpMMFusedInterLayerRedundant : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  Timer analysis() override {
    Timer t;
    t.start();
    // sym_lib::ScheduleParameters sp;
    // sp._num_threads = InTensor->NumThreads;
    //  create the fused set

    Sp._num_w_partition = std::max<int>(InTensor->ACsr->m / Sp.IterPerPartition,
                                        2 * Sp._num_threads);
    auto *sf01 = new sym_lib::SparseFusionWithRedundancy(&Sp, 2);
    auto *mvDAG = sym_lib::diagonal(InTensor->ACsr->m, 1.0);
    auto *tmpCSCCSR = new sym_lib::CSC(InTensor->BCsr->m, InTensor->BCsr->n,
                                       InTensor->BCsr->nnz, InTensor->BCsr->p,
                                       InTensor->BCsr->i, InTensor->BCsr->x);
    auto *Di = InTensor->BCsr;
    // sym_lib::print_csc(1, "Di", 6, Di->p, Di->i, Di->x);
    sf01->fuse(1, mvDAG, tmpCSCCSR);

    // sf01->print_final_list();
    sf01->fuse(0, mvDAG, tmpCSCCSR);
    // sf01->print_final_list();
    auto pt = St->OtherStats["PackingType"];
    FusedCompSet = sf01->getFusedCompressed((int)pt[0]);
    sf01->measureRedundancy(tmpCSCCSR, SpInfo);
    // FusedCompSet->print_3d();
    delete sf01;
    delete mvDAG;
    delete tmpCSCCSR;

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    auto *ws = new double[InTensor->NumThreads * 2 * InTensor->M * Sp.TileN]();
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrTiledFusedRedundantGeneral(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        FusedCompSet->ker_begin_, InTensor->NumThreads, Sp.TileM, Sp.TileN, ws);

    t.stop();
    delete[] ws;
    return t;
  }

public:
  SpMMSpMMFusedInterLayerRedundant(TensorInputs<double> *In1, Stats *Stat1,
                                   sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn) {}

  ~SpMMSpMMFusedInterLayerRedundant() { delete FusedCompSet; }

  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};

// mixed redundant and non redundant
class SpMMSpMMFusedInterLayerMixed : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
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
    std::vector<std::vector<sym_lib::FusedNode *>> updatedFinalList(2);
    sym_lib::BalanceWithRedundantComputation(sf01->getFinalNodeList(),
                                             updatedFinalList, tmpCSCCSR,
                                             0.1 * Sp._lbc_agg);
    // sf01->print_final_list();
    auto pt = St->OtherStats["PackingType"];
    // FusedCompSet = sf01->getFusedCompressed((int) pt[0]);
    FusedCompSet = new sym_lib::MultiDimensionalSet(updatedFinalList, pt[0]);
    sym_lib::measureRedundancy(tmpCSCCSR, SpInfo, updatedFinalList);
    // FusedCompSet->print_3d();
    delete sf01;
    delete mvDAG;
    delete tmpCSCCSR;

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrFused(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMSpMMFusedInterLayerMixed(TensorInputs<double> *In1, Stats *Stat1,
                               sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn) {}

  ~SpMMSpMMFusedInterLayerMixed() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};

class SpMMSpMMFusedInnerProdInterLayer : public SpMMSpMMFusedInterLayer {
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrInnerProductFused(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMSpMMFusedInnerProdInterLayer(TensorInputs<double> *In1, Stats *Stat1,
                                   sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedInterLayer(In1, Stat1, SpIn) {}
  ~SpMMSpMMFusedInnerProdInterLayer() {}
};

class SpMMSpMMFusedTiled : public SpMMSpMMFusedInterLayer {
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    auto *ws = new double[InTensor->NumThreads * Sp.TileM * Sp.TileN];
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrTiledFused(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads, Sp.TileM, Sp.TileN, ws);

    t.stop();
    delete[] ws;
    return t;
  }

public:
  SpMMSpMMFusedTiled(TensorInputs<double> *In1, Stats *Stat1,
                     sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedInterLayer(In1, Stat1, SpIn) {}
  ~SpMMSpMMFusedTiled() {}
};

class SpMMSpMMFusedTiledTri : public SpMMSpMMFusedInterLayer {

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
      int begin = std::max(0, i - BandWidth + 1),
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
    // FusedCompSet->print_3d();

    t.stop();
    return t;
  }
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    auto *ws = new double[InTensor->NumThreads * 2 * Sp.TileM * Sp.TileN]();
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrTiledFusedRedundantBanded(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        FusedCompSet->ker_begin_, InTensor->NumThreads, Sp.TileM, Sp.TileN, ws);

    t.stop();
    delete[] ws;
    return t;
  }

public:
  SpMMSpMMFusedTiledTri(TensorInputs<double> *In1, Stats *Stat1,
                        sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedInterLayer(In1, Stat1, SpIn) {}
  ~SpMMSpMMFusedTiledTri() {}
};

class SpMMSpMMFusedSepInterLayer : public SpMMSpMMFusedInterLayer {
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrSeparatedFused(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        FusedCompSet->ker_begin_, InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMSpMMFusedSepInterLayer(TensorInputs<double> *In1, Stats *Stat1,
                             sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedInterLayer(In1, Stat1, SpIn) {}
};

class SpMMSpMMFusedMixedInterLayer : public SpMMSpMMFusedInterLayer {
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrMixedScheduleFused(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMSpMMFusedMixedInterLayer(TensorInputs<double> *In1, Stats *Stat1,
                               sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedInterLayer(In1, Stat1, SpIn) {}
};

// class SpMMSpMMUnFusedCTiledParallel : public SpMMSpMMUnFused {
//   protected:
//     Timer execute() override {
//         //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
//         //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
//         int tileM = 128, tileN = 32;
//         OutTensor->reset();
//         Timer t;
//         t.start();
//         swiftware::sparse::spmmCsrParallelTiled(
//             InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
//             InTensor->ACsr->i, InTensor->ACsr->x, InTensor->Bx,
//             OutTensor->ACx, InTensor->NumThreads, tileM, tileN);
//         swiftware::sparse::spmmCsrParallelTiled(
//             InTensor->L, InTensor->N, InTensor->M, InTensor->BCsr->p,
//             InTensor->BCsr->i, InTensor->BCsr->x, OutTensor->ACx,
//             OutTensor->Xx, InTensor->NumThreads, tileM, tileN);
//         t.stop();
//         return t;
//     }
//   public:
//     SpMMSpMMUnFusedCTiledParallel(TensorInputs<double> *In1, Stats *Stat1) :
//     SpMMSpMMUnFused(In1, Stat1){
//     }
// };

class SpMMSpMMFusionProfiler : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  Timer analysis() override {
    Timer t;
    t.start();
    // sym_lib::ScheduleParameters sp;
    // sp._num_threads = InTensor->NumThreads;
    //  create the fused set
    Sp._num_w_partition = std::max<int>(InTensor->ACsr->m / Sp._num_w_partition,
                                        2 * Sp._num_threads);
    auto *sf01 = new sym_lib::SparseFusion(&Sp, 2);
    auto *mvDAG = sym_lib::diagonal(InTensor->ACsr->m, 1.0);
    sf01->fuse(0, mvDAG, NULLPNTR);
    auto *tmpCSCCSR = new sym_lib::CSC(InTensor->BCsr->m, InTensor->BCsr->n,
                                       InTensor->BCsr->nnz, InTensor->BCsr->p,
                                       InTensor->BCsr->i, InTensor->BCsr->x);
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
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    //        swiftware::sparse::spmmCsrSpmmCsrFused(InTensor->M, InTensor->N,
    //                                               InTensor->K, InTensor->L,
    //                                               InTensor->ACsr->p,
    //                                               InTensor->ACsr->i,
    //                                               InTensor->ACsr->x,
    //                                               InTensor->BCsr->p,
    //                                               InTensor->BCsr->i,
    //                                               InTensor->BCsr->x,
    //                                               InTensor->Bx,
    //                                               OutTensor->Xx,
    //                                               OutTensor->ACx,
    //                                               FusedCompSet->n1_,
    //                                               FusedCompSet->ptr1_,
    //                                               FusedCompSet->ptr2_,
    //                                               FusedCompSet->id_,
    //                                               FusedCompSet->type_,
    //                                               InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMSpMMFusionProfiler(TensorInputs<double> *In1, Stats *Stat1,
                         sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn) {}

  ~SpMMSpMMFusionProfiler() { delete FusedCompSet; }

  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};

/// Testing SpMM CSR - SpMM CSC

class SpMMCSRSpMMCSCFusedAtomic : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  Timer analysis() override {
    Timer t;
    t.start();

    Sp._num_w_partition = std::max<int>(InTensor->ACsr->m / Sp.IterPerPartition,
                                        2 * Sp._num_threads);
    auto *sf01 = new sym_lib::SparseFusion(&Sp, 2);
    auto *mvDAG = sym_lib::diagonal(InTensor->ACsr->m, 1.0);
    auto *tmpCSCCSR = sym_lib::diagonal(InTensor->ACsr->m, 1.0);
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
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCscFused(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->B->p, InTensor->B->i,
        InTensor->B->x, InTensor->Bx, OutTensor->Xx, OutTensor->ACx,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->ptr2_,
        FusedCompSet->id_, FusedCompSet->type_, InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMCSRSpMMCSCFusedAtomic(TensorInputs<double> *In1, Stats *Stat1,
                            sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn) {}

  ~SpMMCSRSpMMCSCFusedAtomic() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};

class SpMMCSRSpMMCSCFusedAtomicInterleaved : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  Timer analysis() override {
    Timer t;
    t.start();

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCscFusedAffine(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->B->p, InTensor->B->i,
        InTensor->B->x, InTensor->Bx, OutTensor->Xx, OutTensor->ACx,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMCSRSpMMCSCFusedAtomicInterleaved(TensorInputs<double> *In1, Stats *Stat1,
                                       sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn) {}

  ~SpMMCSRSpMMCSCFusedAtomicInterleaved() {}
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};

class SpMMCSRSpMMCSCFusedColoring : public SpMMSpMMUnFused {
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
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCscFusedColored(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->B->p, InTensor->B->i,
        InTensor->B->x, InTensor->Bx, OutTensor->Xx, OutTensor->ACx,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        Sp.TileM, InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMCSRSpMMCSCFusedColoring(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn,
      int TileSize1, std::map<int, std::vector<int>> ConflictGraphColoring1)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn), TileSize(TileSize1),
        ConflictGraphColoring(ConflictGraphColoring1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }

  ~SpMMCSRSpMMCSCFusedColoring() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};

class SpMMCSRSpMMCSCFusedColoringWithScheduledKTiling : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  InspectorForSingleLayerTiledFusedCSCParallelWithSchedulingKTiles *Inspector;
  int TileSize;
  int KTileSize;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  Timer analysis() override {
    Timer t;
    t.start();

    FusedCompSet =
        Inspector->generateScheduleBasedOnConflictGraphColoring(
            ConflictGraphColoring, InTensor->M, TileSize, InTensor->N, KTileSize);

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCscFusedColoredWithScheduledKTiles(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->B->p, InTensor->B->i,
        InTensor->B->x, InTensor->Bx, OutTensor->Xx, OutTensor->ACx,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        TileSize, KTileSize, InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMCSRSpMMCSCFusedColoringWithScheduledKTiling(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn, int TileSize1,
      std::map<int, std::vector<int>> ConflictGraphColoring1, int KTileSize1)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn), ConflictGraphColoring(ConflictGraphColoring1),
    TileSize(TileSize1), KTileSize(KTileSize1){
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallelWithSchedulingKTiles();
  }

  ~SpMMCSRSpMMCSCFusedColoringWithScheduledKTiling() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};

class SpMMCSRSpMMCSCFusedColoringWithReplicatedKTiling
    : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  InspectorForSingleLayerTiledFusedCSCParallelWithReplicatedKTiles *Inspector;
  int TileSize;
  int KTileSize;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  Timer analysis() override {
    Timer t;
    t.start();

    FusedCompSet =
        Inspector->generateScheduleBasedOnConflictGraphColoring(
            ConflictGraphColoring, InTensor->M, TileSize, InTensor->N, KTileSize);

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCscFusedColoredWithReplicatedKTiles(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->B->p, InTensor->B->i,
        InTensor->B->x, InTensor->Bx, OutTensor->Xx, OutTensor->ACx,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        FusedCompSet->type_, TileSize, KTileSize, InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMCSRSpMMCSCFusedColoringWithReplicatedKTiling(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn, int TileSize1,
      std::map<int, std::vector<int>> ConflictGraphColoring1, int KTileSize1)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn), ConflictGraphColoring(ConflictGraphColoring1),
        TileSize(TileSize1), KTileSize(KTileSize1){
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallelWithReplicatedKTiles();
  }

  ~SpMMCSRSpMMCSCFusedColoringWithReplicatedKTiling() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};


#ifdef MKL

#include <mkl.h>
class SpMMSpMMMKL : public SpMMSpMMUnFused {
protected:
  sparse_matrix_t A;
  sparse_matrix_t B;
  MKL_INT *LLI_A;
  MKL_INT *LLI_B;
  matrix_descr d;
  Timer execute() override {
    Timer t;
    t.start();
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, this->A, this->d,
                    SPARSE_LAYOUT_ROW_MAJOR, this->InTensor->Bx,
                    this->InTensor->N, this->InTensor->N, 0,
                    this->OutTensor->ACx, this->InTensor->N);
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, this->B, this->d,
                    SPARSE_LAYOUT_ROW_MAJOR, this->OutTensor->ACx,
                    this->InTensor->N, this->InTensor->N, 0,
                    this->OutTensor->Xx, this->InTensor->N);
    t.stop();
    return t;
  }

public:
  SpMMSpMMMKL(TensorInputs<double> *In1, Stats *Stat1)
      : SpMMSpMMUnFused(In1, Stat1) {
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

  ~SpMMSpMMMKL() {
    mkl_free(A);
    mkl_free(B);
  }
};
#endif
#endif // SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H

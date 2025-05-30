//
// Created by kazem on 08/05/23.
//

#ifndef SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H
#define SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H

#include "SWTensorBench.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Inspector.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include "sparse-fusion/SpMM_SpMM.h"
#include "sparse-fusion/SpMM_SpMM_vectorized.h"
#include "sparse-fusion/SparseFusion.h"
#include "sparse-fusion/SparseFusionWithRedundancy.h"
#include <cmath>
#include <numeric>
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

  TensorInputs<T>(int M1, int N1, int K1, int L1, sym_lib::CSC *A1,
               sym_lib::CSC *B1, int NumThreads1, int NumTrial1,
               std::string ExpN)
      : Inputs<T>(NumTrial1, NumThreads1, ExpN) {
    M = M1;
    N = N1;
    K = K1;
    L = L1;
//    A = sym_lib::copy_sparse(A1);
//    B = sym_lib::copy_sparse(B1);
    ACsr = sym_lib::csc_to_csr(A1);
    BCsr = ACsr;
    Bx = new T[K * N]();
    // randomly initialize the input
    for (int i = 0; i < K * N; ++i) {
      Bx[i] = 1.0; //(double)rand()/RAND_MAX;
    }
    CorrectSol = new T[L * N](); // the correct solution
    IsSolProvided = false;
    Inputs<T>::Threshold = 1e-6;
  }

  ~TensorInputs() {
    delete[] Bx;
    delete[] CorrectSol;
//    delete A;
//    delete B;
    delete ACsr;
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

  virtual void reset() {
    std::fill_n(Xx, L * N, 0.0);
    std::fill_n(ACx, M * N, 0.0);
  }
};

class SpMMSpMMUnFused : public SWTensorBench<double> {
protected:
  TensorInputs<double> *InTensor;

  void setup() override {
    this->St->OtherStats["NTile"] = {4};
    this->St->OtherStats["Number of Fused Nodes"] = {0.};
    this->St->OtherStats["Number of Fused nnz"] = {0.};
    this->St->OtherStats["Tile Size Mean"] = {0.};
    this->St->OtherStats["Tile Size STD"] = {0.};
  }

  void preExecute() override {}

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSequential<double>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->Bx, OutTensor->ACx);
    swiftware::sparse::spmmCsrSequential<double>(
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
    OutTensor->reset();
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
    delete[] LLI_A;
    delete[] LLI_B;
  }
};


class SpMMMKLImpl : public SpMMSpMMUnFused {
protected:
  sparse_matrix_t A;
  MKL_INT *LLI_A;
  matrix_descr d;
  Timer execute() override {
    Timer t;
    t.start();
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, this->A, this->d,
                    SPARSE_LAYOUT_ROW_MAJOR, this->InTensor->Bx,
                    this->InTensor->N, this->InTensor->N, 0,
                    this->OutTensor->ACx, this->InTensor->N);
    t.stop();
    return t;
  }

public:
  SpMMMKLImpl(TensorInputs<double> *In1, Stats *Stat1)
      : SpMMSpMMUnFused(In1, Stat1) {
    d.type = SPARSE_MATRIX_TYPE_GENERAL;

    LLI_A = new MKL_INT[this->InTensor->M + 1]();
    for (int l = 0; l < this->InTensor->M + 1; ++l) {
      LLI_A[l] = this->InTensor->ACsr->p[l];
    }

    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, this->InTensor->M,
                            this->InTensor->K, LLI_A, LLI_A + 1,
                            this->InTensor->ACsr->i, this->InTensor->ACsr->x);
    mkl_set_num_threads(this->InTensor->NumThreads);
  }

  ~SpMMMKLImpl() {
    mkl_free(A);
  }
};



#endif


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
    int fusedNodesNum = FusedCompSet->getNumberOfFusedNodes();
    int fusedNnzNum = FusedCompSet->getFusedNnzNum(InTensor->ACsr);
    this->St->OtherStats["Number of Fused Nodes"] = {(double)fusedNodesNum};
    this->St->OtherStats["Number of Fused nnz"] = {(double)fusedNnzNum};
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
    swiftware::sparse::spmmCsrSpmmCsrFused<double>(
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

class SpMMSpMMFusedVariableTileSize: public SpMMSpMMFusedInterLayer{
protected:
  InspectorForTileFusedCSRVariableTileSize *inspector;

  Timer analysis() override{
    auto tm = St->OtherStats["TilingMethod"];
    if(tm[0] == sym_lib::Fixed){
      return SpMMSpMMFusedInterLayer::analysis();
    }
    else {
      Timer t1;
      t1.start();
      FusedCompSet = inspector->generateVariableTileSizeSchedule(InTensor->ACsr,InTensor->N);
      //    FusedCompSet->print_3d();
      t1.stop();
      return t1;
    }
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrFused<double>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

  SpMMSpMMFusedVariableTileSize(TensorInputs<double> *In1, Stats *Stat1,
                                sym_lib::ScheduleParameters SpIn,
                                InspectorForTileFusedCSRVariableTileSize *Inspector1)
      : SpMMSpMMFusedInterLayer(In1, Stat1, SpIn){
    inspector = Inspector1;
  }
public:
  SpMMSpMMFusedVariableTileSize(TensorInputs<double> *In1, Stats *Stat1,
                                sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedInterLayer(In1, Stat1, SpIn){
    inspector = new InspectorForTileFusedCSRVariableTileSize(SpIn, Stat1);
  }

  ~SpMMSpMMFusedVariableTileSize(){
    delete inspector;
  }
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
    swiftware::sparse::spmmCsrSpmmCsrFused<double>(
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
    int fusedNodesNum = FusedCompSet->getNumberOfFusedNodes();
    int fusedNnzNum = FusedCompSet->getFusedNnzNum(InTensor->ACsr);
    this->St->OtherStats["Number of Fused Nodes"] = {(double)fusedNodesNum};
    this->St->OtherStats["Number of Fused nnz"] = {(double)fusedNnzNum};
    // FusedCompSet->print_3d();
    delete sf01;
    delete mvDAG;
    delete tmpCSCCSR;
    delete FusedCompSet;
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

//  ~SpMMSpMMFusionProfiler() { delete FusedCompSet; }

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
  Timer analysis() override {
    Timer t;
    t.start();

    FusedCompSet =
        Inspector->generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            ConflictGraphColoring, InTensor->M, Sp.IterPerPartition);

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
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p, InTensor->BCsr->i,
        InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx, OutTensor->ACx,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        Sp.IterPerPartition, InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMCSRSpMMCSCFusedColoring(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn,
      std::map<int, std::vector<int>> ConflictGraphColoring1)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn),
        ConflictGraphColoring(ConflictGraphColoring1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }

  ~SpMMCSRSpMMCSCFusedColoring() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};

class SpMMCSRSpMMCSCFusedColoringNTiling : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  InspectorForSingleLayerTiledFusedCSCParallel *Inspector;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  Timer analysis() override {
    Timer t;
    t.start();

    FusedCompSet =
        Inspector->generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            ConflictGraphColoring, InTensor->M, Sp.IterPerPartition);

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCscFusedColoredNTiling(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p, InTensor->BCsr->i,
        InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx, OutTensor->ACx,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        Sp.IterPerPartition, Sp.TileN,InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMCSRSpMMCSCFusedColoringNTiling(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn,
      std::map<int, std::vector<int>> ConflictGraphColoring1)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn),
        ConflictGraphColoring(ConflictGraphColoring1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }

  ~SpMMCSRSpMMCSCFusedColoringNTiling() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};


#ifdef __AVX2__

class SpMMSpMMUnFusedParallelVectorizedAVX2 : public SpMMSpMMUnFused {
protected:
  sym_lib::ScheduleParameters Sp;
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrVectorized2_16(InTensor->M, InTensor->N,
                                       InTensor->ACsr->p, InTensor->ACsr->i,
                                       InTensor->ACsr->x, InTensor->Bx,
                                       OutTensor->ACx, Sp.IterPerPartition,InTensor->NumThreads);
    swiftware::sparse::spmmCsrVectorized2_16(InTensor->M, InTensor->N,
                                       InTensor->BCsr->p, InTensor->BCsr->i,
                                       InTensor->BCsr->x, OutTensor->ACx,
                                       OutTensor->Xx, Sp.IterPerPartition,InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMSpMMUnFusedParallelVectorizedAVX2(TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn) {}
};


class SpMMSpMMFusedInterLayerKTiled8VectorizedAvx256: public SpMMSpMMFusedVariableTileSize{
protected:
  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrFusedKTiled8Vectorized(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->ACsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMSpMMFusedInterLayerKTiled8VectorizedAvx256(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedVariableTileSize(
            In1, Stat1, SpIn,
            new InspectorForTileFusedCSRVariableTileSizeWithKTiles8(SpIn,
                                                                    Stat1)) {}
};

class SpMMSpMMFusedInterLayerVectorizedAvx256 : public SpMMSpMMFusedVariableTileSize {
protected:
  void (*spmmCsrSpmmCsrFusedVectorizedFunc)(int , int , int , int ,
                                            const int *, const int *, const double *,
                                            const int *, const int *,const double *,
                                            const double *,
                                            double *,
                                            double *,
                                            int , const int *, const int *,
                                            const int *, const int *,
                                            int );
  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spmmCsrSpmmCsrFusedVectorizedFunc(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->ACsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMSpMMFusedInterLayerVectorizedAvx256(TensorInputs<double> *In1, Stats *Stat1,
                                          sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedVariableTileSize(In1, Stat1, SpIn) {
    if(this->InTensor->N == 8){
      this->spmmCsrSpmmCsrFusedVectorizedFunc = swiftware::sparse::spmmCsrSpmmCsrFusedVectorized2_8;
    }
    else{
      this->spmmCsrSpmmCsrFusedVectorizedFunc = swiftware::sparse::spmmCsrSpmmCsrFusedVectorized2_16;
    }
  }

};

class SpMMCSRSpMMCSCFusedColoringVectorized : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  InspectorForSingleLayerTiledFusedCSCParallel *Inspector;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  Timer analysis() override {
    Timer t;
    t.start();

    FusedCompSet =
        Inspector->generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            ConflictGraphColoring, InTensor->M, Sp.IterPerPartition);

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCscFusedColoredAvx256(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->id_, Sp.IterPerPartition, InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMCSRSpMMCSCFusedColoringVectorized(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn,
      std::map<int, std::vector<int>> ConflictGraphColoring1)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn),
        ConflictGraphColoring(ConflictGraphColoring1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }

  ~SpMMCSRSpMMCSCFusedColoringVectorized() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};


class SpMMCSRSpMMCSCFusedColoringVectorizedPacked : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  InspectorForSingleLayerTiledFusedCSCParallel *Inspector;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  Timer analysis() override {
    Timer t;
    t.start();

    FusedCompSet =
        Inspector->generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            ConflictGraphColoring, InTensor->M, Sp.IterPerPartition);

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    double* packedDx = new double[InTensor->L * InTensor->N]{};
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCscFusedColoredAvx256Packed(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, packedDx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->id_, Sp.IterPerPartition, InTensor->NumThreads);
    t.stop();
    unpack(packedDx, 16);
    delete[] packedDx;
    return t;
  }

  void unpack(double *packedResult, int NPack){
    for (int i = 0; i < InTensor->L; i++){
      for (int k = 0; k < InTensor->N; k+=NPack){
        for (int kk = 0; kk < NPack; kk+=1){
          OutTensor->Xx[i * InTensor->N + k + kk] = packedResult[k * InTensor->L + i * NPack + kk];
        }
      }
    }
  }

public:
  SpMMCSRSpMMCSCFusedColoringVectorizedPacked(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn,
      std::map<int, std::vector<int>> ConflictGraphColoring1)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn),
        ConflictGraphColoring(ConflictGraphColoring1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }

  ~SpMMCSRSpMMCSCFusedColoringVectorizedPacked() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};
#endif

#ifdef __AVX512F__

class SpMMSpMMUnFusedParallelVectorizedAvx512 : public SpMMSpMMUnFused {
protected:
  sym_lib::ScheduleParameters Sp;
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrVectorized2_32Avx512(InTensor->M, InTensor->N,
                                             InTensor->ACsr->p, InTensor->ACsr->i,
                                             InTensor->ACsr->x, InTensor->Bx,
                                             OutTensor->ACx, Sp.IterPerPartition,InTensor->NumThreads);
    swiftware::sparse::spmmCsrVectorized2_32Avx512(InTensor->M, InTensor->N,
                                             InTensor->BCsr->p, InTensor->BCsr->i,
                                             InTensor->BCsr->x, OutTensor->ACx,
                                             OutTensor->Xx, Sp.IterPerPartition,InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMSpMMUnFusedParallelVectorizedAvx512(TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn) {}
};

class SpMMSpMMFusedInterLayerKTiled8VectorizedAvx512 : public SpMMSpMMFusedVariableTileSize {
protected:
  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrFusedVectorizedKTiled8Avx512(
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
  SpMMSpMMFusedInterLayerKTiled8VectorizedAvx512(TensorInputs<double> *In1, Stats *Stat1,
                                          sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedVariableTileSize(In1, Stat1, SpIn, new InspectorForTileFusedCSRVariableTileSizeWithKTiles8(SpIn, Stat1)) {
  }

};

class SpMMSpMMFusedInterLayerVectorizedAvx512 : public SpMMSpMMFusedVariableTileSize {
protected:
  void (*spmmCsrSpmmCsrFusedVectorizedFunc)(int , int , int , int ,
                                            const int *, const int *, const double *,
                                            const int *, const int *,const double *,
                                            const double *,
                                            double *,
                                            double *,
                                            int , const int *, const int *,
                                            const int *, const int *,
                                            int );
  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spmmCsrSpmmCsrFusedVectorizedFunc(
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
  SpMMSpMMFusedInterLayerVectorizedAvx512(TensorInputs<double> *In1, Stats *Stat1,
                                          sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedVariableTileSize(In1, Stat1, SpIn) {
    if(In1->N==8) {
      spmmCsrSpmmCsrFusedVectorizedFunc = swiftware::sparse::spmmCsrSpmmCsrFusedVectorized8Avx512;
    }
    else {
      spmmCsrSpmmCsrFusedVectorizedFunc = swiftware::sparse::spmmCsrSpmmCsrFusedVectorized2_32Avx512;
    }
  }

};

class SpMMCSRSpMMCSCFusedColoringAvx512 : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  InspectorForSingleLayerTiledFusedCSCParallel *Inspector;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  Timer analysis() override {
    Timer t;
    t.start();

    FusedCompSet =
        Inspector->generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            ConflictGraphColoring, InTensor->M, Sp.IterPerPartition);

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCscFusedColoredAvx512(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->id_, Sp.IterPerPartition, InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMCSRSpMMCSCFusedColoringAvx512(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn,
      std::map<int, std::vector<int>> ConflictGraphColoring1)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn),
        ConflictGraphColoring(ConflictGraphColoring1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }

  ~SpMMCSRSpMMCSCFusedColoringAvx512() { delete FusedCompSet; }
  sym_lib::SparsityProfileInfo getSpInfo() { return SpInfo; }
};
#endif

class SpMMCSRSpMMCSCFusedColoringRowTiling: public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  sym_lib::SparsityProfileInfo SpInfo;
  InspectorForSingleLayerTiledFusedCSCParallel *Inspector;
  std::map<int, std::vector<int>> ConflictGraphColoring;
  Timer analysis() override {
    Timer t;
    t.start();

    FusedCompSet =
        Inspector->generateFusedScheduleForSingleLayerTiledFusedCSCParallel(
            ConflictGraphColoring, InTensor->M, Sp.IterPerPartition);

    t.stop();
    return t;
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCscFusedColoredIterationTiled(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p, InTensor->BCsr->i,
        InTensor->BCsr->x, InTensor->Bx, OutTensor->Xx, OutTensor->ACx,
        FusedCompSet->n1_, FusedCompSet->ptr1_, FusedCompSet->id_,
        Sp.IterPerPartition, Sp.IterPerPartition, InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMCSRSpMMCSCFusedColoringRowTiling(
      TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters SpIn,
      std::map<int, std::vector<int>> ConflictGraphColoring1)
      : SpMMSpMMUnFused(In1, Stat1), Sp(SpIn),
        ConflictGraphColoring(ConflictGraphColoring1) {
    Inspector = new InspectorForSingleLayerTiledFusedCSCParallel();
  }

  ~SpMMCSRSpMMCSCFusedColoringRowTiling() { delete FusedCompSet; }
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

//class SpMMParallelVectorized : public SpMMSpMMUnFused {
//protected:
//  Timer execute() override {
//    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
//    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
//    OutTensor->reset();
//    Timer t;
//    t.start();
//    swiftware::sparse::spmmCsrParallel(InTensor->M, InTensor->N, InTensor->K,
//                                       InTensor->ACsr->p, InTensor->ACsr->i,
//                                       InTensor->ACsr->x, InTensor->Cx,
//                                       OutTensor->ACx, InTensor->NumThreads);
//    t.stop();
//    return t;
//  }
//
//public:
//  SpMMParallelVectorized(TensorInputs<double> *In1, Stats *Stat1)
//      : SpMMSpMMUnFused(In1, Stat1) {}
//};

#ifdef __AVX2__
class SpMMParallelVectorizedUnroll48: public SpMMSpMMUnFused {
protected:
  sym_lib::ScheduleParameters Sp;
  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmm8CsrVectorizedUnrollJ4(InTensor->M, InTensor->N,
                                       InTensor->ACsr->p, InTensor->ACsr->i,
                                       InTensor->ACsr->x, InTensor->Bx,
                                       OutTensor->ACx, Sp.TileM, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMParallelVectorizedUnroll48(TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters Sp1)
      : SpMMSpMMUnFused(In1, Stat1), Sp(Sp1) {}
};

class SpMMParallelVectorizedUnroll216: public SpMMParallelVectorizedUnroll48 {
protected:
  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrVectorized2_16(
        InTensor->M, InTensor->N, InTensor->ACsr->p, InTensor->ACsr->i,
        InTensor->ACsr->x, InTensor->Bx, OutTensor->ACx, Sp.TileM,
        InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMParallelVectorizedUnroll216(TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters Sp1)
      : SpMMParallelVectorizedUnroll48(In1, Stat1, Sp1) {}
};

class SpMMParallelVectorizedUnroll16: public SpMMParallelVectorizedUnroll48 {
protected:
  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmm16CsrVectorized(InTensor->M, InTensor->N,
                                       InTensor->ACsr->p, InTensor->ACsr->i,
                                       InTensor->ACsr->x, InTensor->Bx,
                                       OutTensor->ACx, Sp.TileM ,InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMParallelVectorizedUnroll16(TensorInputs<double> *In1, Stats *Stat1, sym_lib::ScheduleParameters Sp1)
      : SpMMParallelVectorizedUnroll48(In1, Stat1, Sp1) {}
};

#endif
#endif // SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H
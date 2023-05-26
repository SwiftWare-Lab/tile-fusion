//
// Created by mehdi on 5/25/23.
//

#include "SpMM_SpMM_Demo_Utils.h"
#include "SWTensorBench.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/SpMM_SpMM.h"
#include "sparse-fusion/SparseFusion.h"

#ifdef MKL
#include <cmath>
#include <mkl.h>
#indlude <mkl_spblas.h>
#endif


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

/// implementation of TensorInputs<T>
template <typename T> TensorInputs<T>::TensorInputs(int M1, int N1, int K1, int L1, sym_lib::CSC *A1,
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
  Cx = new double[K * N]();
  // randomly initialize the input
  for (int i = 0; i < K * N; ++i) {
    Cx[i] = 1.0; //(double)rand()/RAND_MAX;
  }
  CorrectMul = new double[L * N](); // the correct solution
  IsSolProvided = false;
  Inputs<T>::Threshold = 1e-6;
}

template <typename T> TensorInputs<T>::~TensorInputs() {
  delete[] Cx;
  delete[] CorrectMul;
  delete A;
  delete B;
  delete ACsr;
  delete BCsr;
}


/// implementation of TensorOutputs<T>
template <typename T> TensorOutputs<T>::TensorOutputs(int M, int N, int L) : M(M), N(N), L(L) {
  Dx = new T[L * N]();
  ACx = new T[M * N]();
}

template <typename T> TensorOutputs<T>::~TensorOutputs() {
  delete[] Dx;
  delete[] ACx;
}

template <typename T> void TensorOutputs<T>::printDx() {
  std::cout << "\n ACx:\n";
  printDense<T>(M, N, ACx);
  std::cout << "\n Dx:\n";
  printDense<T>(L, N, Dx);
  std::cout << "\n";
}

template <typename T> void TensorOutputs<T>::reset() {
  std::fill_n(Dx, L * N, 0.0);
  std::fill_n(ACx, M * N, 0.0);
}

/// implementation of SpMMSpMMUnFused
void SpMMSpMMUnFused::setup(){}

void SpMMSpMMUnFused::preExecute(){}

Timer SpMMSpMMUnFused::execute(){
  //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
  //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
  OutTensor->reset();
  Timer t;
  t.start();
  swiftware::sparse::spmmCsrSequential(
      InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
      InTensor->ACsr->i, InTensor->ACsr->x, InTensor->Cx, OutTensor->ACx);
  swiftware::sparse::spmmCsrSequential(
      InTensor->L, InTensor->N, InTensor->M, InTensor->BCsr->p,
      InTensor->BCsr->i, InTensor->BCsr->x, OutTensor->ACx, OutTensor->Dx);
  t.stop();
  return t;
}

bool SpMMSpMMUnFused::verify(double &Error){
  bool retValue = true;
  if (!InTensor->IsSolProvided) {
    Error = 0;
    return true;
  }
  double infNorm = 0;
  for (int i = 0; i < InTensor->L * InTensor->N; ++i) {
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

SpMMSpMMUnFused::SpMMSpMMUnFused(TensorInputs<double> *In1, Stats *Stat1)
    : SWTensorBench<double>(In1, Stat1) {
  OutTensor = new TensorOutputs<double>(In1->M, In1->N, In1->L);
  InTensor = In1;
}

SpMMSpMMUnFused::~SpMMSpMMUnFused() { delete OutTensor; }


/// implementation of SpMMSpMMUnFusedParallel
Timer SpMMSpMMUnFusedParallel::execute(){
  //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
  //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
  OutTensor->reset();
  Timer t;
  t.start();
  swiftware::sparse::spmmCsrParallel(InTensor->M, InTensor->N, InTensor->K,
                                     InTensor->ACsr->p, InTensor->ACsr->i,
                                     InTensor->ACsr->x, InTensor->Cx,
                                     OutTensor->ACx, InTensor->NumThreads);
  swiftware::sparse::spmmCsrParallel(InTensor->L, InTensor->N, InTensor->M,
                                     InTensor->BCsr->p, InTensor->BCsr->i,
                                     InTensor->BCsr->x, OutTensor->ACx,
                                     OutTensor->Dx, InTensor->NumThreads);
  t.stop();
  return t;
}

SpMMSpMMUnFusedParallel::SpMMSpMMUnFusedParallel(TensorInputs<double> *In1,
                                                 Stats *Stat1)
    : SpMMSpMMUnFused(In1, Stat1) {}

/// implementation of SpMMSpMMFusedInterLayer
Timer SpMMSpMMFusedInterLayer::analysis(){
  Timer t;
  t.start();
  sym_lib::ScheduleParameters sp;
  sp._num_threads = InTensor->NumThreads;
  // create the fused set
  auto *sf01 = new sym_lib::SparseFusion(&sp, 2);
  auto *mvDAG = sym_lib::diagonal(InTensor->ACsr->m, 1.0);
  sf01->fuse(0, mvDAG, NULLPNTR);
  auto *tmpCSCCSR = new sym_lib::CSC(InTensor->BCsr->m, InTensor->BCsr->n,
                                     InTensor->BCsr->nnz, InTensor->BCsr->p,
                                     InTensor->BCsr->i, InTensor->BCsr->x);
  // sf01->print_final_list();
  sf01->fuse(1, mvDAG, tmpCSCCSR);
  // sf01->print_final_list();
  FusedCompSet = sf01->getFusedCompressed();
  // FusedCompSet->print_3d();
  delete sf01;
  delete mvDAG;

  t.stop();
  return t;
}

Timer SpMMSpMMFusedInterLayer::execute(){
  //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
  //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
  OutTensor->reset();
  Timer t;
  t.start();
  swiftware::sparse::spmmCsrSpmmCsrFused(
      InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
      InTensor->ACsr->i, InTensor->ACsr->x, InTensor->BCsr->p,
      InTensor->BCsr->i, InTensor->BCsr->x, InTensor->Cx, OutTensor->Dx,
      OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
      FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
      InTensor->NumThreads);

  t.stop();
  return t;
}

SpMMSpMMFusedInterLayer::SpMMSpMMFusedInterLayer(TensorInputs<double> *In1,
                                                 Stats *Stat1)
    : SpMMSpMMUnFused(In1, Stat1) {}

SpMMSpMMFusedInterLayer::~SpMMSpMMFusedInterLayer() { delete FusedCompSet; }

/// implementation of SpMMSpMMMKL
SpMMSpMMMKL::SpMMSpMMMKL(TensorInputs<double> *In1, Stats *Stat1)
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

  mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO,
                          this->InTensor->M, this->InTensor->K, LLI_A,
                          LLI_A + 1, this->InTensor->ACsr->i,
                          this->InTensor->ACsr->x);
  mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO,
                          this->InTensor->L, this->InTensor->M, LLI_B,
                          LLI_B + 1, this->InTensor->BCsr->i,
                          this->InTensor->BCsr->x);
  mkl_set_num_threads(this->InTensor->NumThreads);
  mkl_set_num_threads_local(this->InTensor->NumThreads);
}

Timer SpMMSpMMMKL::execute() {
  Timer t;
  t.start();
  mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1,
                  this->A, this->d,
                  SPARSE_LAYOUT_ROW_MAJOR,
                  this->InTensor->Cx, this->InTensor->N,
                  this->InTensor->N, 0,
                  this->OutTensor->ACx, this->InTensor->N);
  mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1,
                  this->B, this->d,
                  SPARSE_LAYOUT_ROW_MAJOR,
                  this->OutTensor->ACx, this->InTensor->N,
                  this->InTensor->N, 0,
                  this->OutTensor->Dx, this->InTensor->N);
  t.stop();
  return t;
}
SpMMSpMMMKL::~SpMMSpMMMKL() {
  mkl_free(A);
  mkl_free(B);
  mkl_free(LLI_A);
  mkl_free(LLI_B);
}

//
// Created by kazem on 08/05/23.
//

#include "SWTensorBench.h"
#include "aggregation/sparse_utilities.h"
#include "papi_wrapper.h"
#include "sparse-fusion/SpMM_SpMM.h"
#include <omp.h>

#ifndef SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H
#define SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H
using namespace swiftware::benchmark;
template<typename T>
struct TensorInputs : public Inputs<T>{
  int M, N, K, L;
  sym_lib::CSC *A, *B;
  sym_lib::CSR *ACsr, *BCsr;

  T *Cx;

  TensorInputs(int M1, int N1, int K1, int L1,
               sym_lib::CSC *A1, sym_lib::CSC *B1,
               int NumThreads1) : Inputs<T>(){
    M = M1;
    N = N1;
    K = K1;
    L = L1;
    A = sym_lib::copy_sparse(A1);
    B = sym_lib::copy_sparse(B1);
    ACsr = sym_lib::csc_to_csr(A);
    BCsr = sym_lib::csc_to_csr(B);
    Cx = new double[K * N]();
    Inputs<T>::NumThreads = NumThreads1;
  }
};

template<typename T>
struct TensorOutputs : public Outputs<T>{
  T *Dx, *ACx;

  TensorOutputs(int M, int N, int L){
    Dx = new double[L * N]();
    ACx = new double[M * N]();
  }

  ~TensorOutputs(){
    delete[] Dx;
    delete[] ACx;
  }
};

class SpMMSpMMUnFused : public SWTensorBench<double> {
protected:
  TensorInputs<double> *InTensor;
  TensorOutputs<double> *OutTensor;
  void setup() override {
  }

  void preExecute() override {
  }

  Timer execute() override {
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrParallel(InTensor->M, InTensor->N,
                                       InTensor->K,
                    InTensor->ACsr->p, InTensor->ACsr->i, InTensor->ACsr->x,
                    InTensor->Cx, OutTensor->ACx, InTensor->NumThreads);
    swiftware::sparse::spmmCsrParallel(InTensor->L, InTensor->N,
                                       InTensor->M,
                                       InTensor->BCsr->p, InTensor->BCsr->i,
                                       InTensor->BCsr->x,
                                       OutTensor->ACx, OutTensor->Dx,
                                       InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMSpMMUnFused(TensorInputs<double> *In1, Stats *Stat1) : SWTensorBench<double>(In1, Stat1){
    OutTensor = new TensorOutputs<double>(In1->M, In1->N, In1->L);
    InTensor = In1;
  }

  ~SpMMSpMMUnFused(){
    delete OutTensor;
  }
};

class SpMMSpMMInterLayer : public SpMMSpMMUnFused {
protected:
  Timer execute() override {
    Timer t;
    t.start();
    //GEMVVec(In->Dim1, In->Dim2, In->A, In->x, In->y);
    t.stop();
    return t;
  }
public:
  SpMMSpMMInterLayer(TensorInputs<double> *In1, Stats *Stat1) : SpMMSpMMUnFused(In1, Stat1){

  }
};




#endif // SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H

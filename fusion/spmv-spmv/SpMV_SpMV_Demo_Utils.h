//
// Created by salehm32 on 08/12/23.
//

#ifndef SPARSE_FUSION_SPMV_SPMV_MKL_DEMO_UTILS_H
#define SPARSE_FUSION_SPMV_SPMV_MKL_DEMO_UTILS_H
#include "SWTensorBench.h"
#include "SpMV_SpMV.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"

using namespace swiftware::benchmark;

template <typename T> void printDense(int M, int N, T *X) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << X[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

template<typename T>
struct TensorInputs : public Inputs<T>{
  int M, K, L;
  sym_lib::CSC *A, *B;
  sym_lib::CSR *ACsr, *BCsr;

  T *Cx;
  T *CorrectMul;
  bool IsSolProvided;

  TensorInputs(int M1, int N1, int K1, int L1,
               sym_lib::CSC *A1, sym_lib::CSC *B1,
               int NumThreads1, int NumTrial1, std::string ExpN):Inputs<T>(NumTrial1, NumThreads1, ExpN){
    M = M1;
    K = K1;
    L = L1;
    A = sym_lib::copy_sparse(A1);
    B = sym_lib::copy_sparse(B1);
    ACsr = sym_lib::csc_to_csr(A);
    BCsr = sym_lib::csc_to_csr(B);
    Cx = new double[K]();
    // randomly initialize the input
    for(int i=0; i<K; ++i){
      Cx[i] = 1.0; //(double)rand()/RAND_MAX;
    }
    CorrectMul = new double[L](); // the correct solution
    IsSolProvided = false;
    Inputs<T>::Threshold = 1e-6;
  }

  ~TensorInputs(){
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

class SpMVSpMVSequential: public SWTensorBench<double> {
  TensorInputs<double> *InTensor;

  void setup() override {
    this->St->OtherStats["NTile"] = {4};
  }

  void preExecute() override {}

  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    spMVCsrSequential(
        InTensor->M, InTensor->K, InTensor->ACsr->p,
        InTensor->ACsr->i, InTensor->ACsr->x, InTensor->Cx, OutTensor->ACx);
    spMVCsrSequential(
        InTensor->L, InTensor->M, InTensor->BCsr->p,
        InTensor->BCsr->i, InTensor->BCsr->x, OutTensor->ACx, OutTensor->Dx);
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
  SpMVSpMVSequential(TensorInputs<double> *In1, Stats *Stat1)
      : SWTensorBench<double>(In1, Stat1) {
    OutTensor = new TensorOutputs<double>(In1->M, In1->L);
    InTensor = In1;
  }

  ~SpMVSpMVSequential() { delete OutTensor; }
};

#endif // SPARSE_FUSION_SPMV_SPMV_MKL_DEMO_UTILS_H

//
// Created by kazem on 08/05/23.
//

#include "SWTensorBench.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include "sparse-fusion/SpMM_SpMM.h"
#include <math.h>
#include <omp.h>

#include <cmath>
#include <mkl.h>


#ifndef SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H
#define SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H
using namespace swiftware::benchmark;

// print a dense matrix with dimension MxN
template <typename T> void printDense(int M, int N, T *X);

template <typename T> struct TensorInputs : public Inputs<T> {
  int M, N, K, L;
  sym_lib::CSC *A, *B;
  sym_lib::CSR *ACsr, *BCsr;

  T *Cx;
  T *CorrectMul;
  bool IsSolProvided;

  TensorInputs(int M1, int N1, int K1, int L1, sym_lib::CSC *A1,
               sym_lib::CSC *B1, int NumThreads1, int NumTrial1,
               std::string ExpN);

  ~TensorInputs();
};


template <typename T> struct TensorOutputs : public Outputs<T> {
  int M, N, L;
  T *Dx, *ACx;

  TensorOutputs(int M, int N, int L);

  ~TensorOutputs();

  void printDx();

  void reset();
};


class SpMMSpMMUnFused : public SWTensorBench<double> {
protected:
  TensorInputs<double> *InTensor;
  void setup() override;
  void preExecute() override;
  Timer execute() override;
  bool verify(double &Error) override;

public:
  TensorOutputs<double> *OutTensor;
  SpMMSpMMUnFused(TensorInputs<double> *In1, Stats *Stat1);
  ~SpMMSpMMUnFused();
};


class SpMMSpMMUnFusedParallel : public SpMMSpMMUnFused {
protected:
  Timer execute() override;

public:
  SpMMSpMMUnFusedParallel(TensorInputs<double> *In1, Stats *Stat1);
};


class SpMMSpMMFusedInterLayer : public SpMMSpMMUnFused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  Timer analysis() override;
  Timer execute() override;

public:
  SpMMSpMMFusedInterLayer(TensorInputs<double> *In1, Stats *Stat1);
  ~SpMMSpMMFusedInterLayer();
};

class SpMMSpMMMKL: public SpMMSpMMUnFused {
protected:
  sparse_matrix_t A;
  sparse_matrix_t B;
  MKL_INT *LLI_A;
  MKL_INT *LLI_B;
  matrix_descr d;
  Timer execute() override;
public:
  SpMMSpMMMKL(TensorInputs<double> *In1, Stats *Stat1);
  ~SpMMSpMMMKL();
};

#endif // SPARSE_FUSION_SPMM_SPMM_DEMO_UTILS_H

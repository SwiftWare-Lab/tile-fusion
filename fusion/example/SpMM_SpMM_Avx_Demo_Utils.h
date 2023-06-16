//
// Created by mehdi on 6/16/23.
//

#include "SpMM_SpMM_Demo_Utils.h"

#ifndef SPARSE_FUSION_SPMM_SPMM_AVX_DEMO_UTILS_H
#define SPARSE_FUSION_SPMM_SPMM_AVX_DEMO_UTILS_H

#endif // SPARSE_FUSION_SPMM_SPMM_AVX_DEMO_UTILS_H

class SpMMSpMMAvx : public SpMMSpMMUnFused {
protected:
  void (*spmm)(int M, int N, int K, const int *Ap, const int *Ai,
               const double *Ax, const double *Bx, double *Cx, int NThreads);
  Timer execute() override {
    //    std::fill_n(OutTensor->Dx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    this->spmm(InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
               InTensor->ACsr->i, InTensor->ACsr->x, InTensor->Cx,
               OutTensor->ACx, InTensor->NumThreads);
    this->spmm(InTensor->L, InTensor->N, InTensor->M, InTensor->BCsr->p,
               InTensor->BCsr->i, InTensor->BCsr->x, OutTensor->ACx,
               OutTensor->Dx, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMSpMMAvx(TensorInputs<double> *In1, Stats *Stat1,
              void (*spmmAvx)(int, int, int, const int *, const int *,
                              const double *, const double *, double *, int))
      : SpMMSpMMUnFused(In1, Stat1) {
    this->spmm = spmmAvx;
  }
};

class SpmmSpmmAvxFirstSparseRow : public SpMMSpMMAvx {
public:
  SpmmSpmmAvxFirstSparseRow(TensorInputs<double> *In1, Stats *Stat1)
      : SpMMSpMMAvx(In1, Stat1,
                    swiftware::sparse::spmmCsrAvxFirstSparseRow) {}
};

class SpmmSpmmAvxFirstDenseRow : public SpMMSpMMAvx {
public:
  SpmmSpmmAvxFirstDenseRow(TensorInputs<double> *In1, Stats *Stat1)
      : SpMMSpMMAvx(In1, Stat1,
                    swiftware::sparse::spmmCsrAvxFirstDenseRow) {}
};

class SpmmSpmmFirstDenseRowSecondSparseRow : public SpMMSpMMAvx {
public:
  SpmmSpmmFirstDenseRowSecondSparseRow(TensorInputs<double> *In1, Stats *Stat1)
      : SpMMSpMMAvx(In1, Stat1,
                    swiftware::sparse::spmmCsrAvxFirstDenseRowSecondSparseRow) {}
};
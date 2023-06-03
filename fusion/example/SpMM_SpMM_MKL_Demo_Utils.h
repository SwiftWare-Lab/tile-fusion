//
// Created by mehdi on 6/3/23.
//

#include "SpMM_SpMM_Demo_Utils.h"
#ifdef MKL
#include <mkl.h>
#endif

#ifndef SPARSE_FUSION_SPMM_SPMM_MKL_DEMO_UTILS_H
#define SPARSE_FUSION_SPMM_SPMM_MKL_DEMO_UTILS_H

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
                    SPARSE_LAYOUT_ROW_MAJOR, this->InTensor->Cx,
                    this->InTensor->N, this->InTensor->N, 0,
                    this->OutTensor->ACx, this->InTensor->N);
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, this->B, this->d,
                    SPARSE_LAYOUT_ROW_MAJOR, this->OutTensor->ACx,
                    this->InTensor->N, this->InTensor->N, 0,
                    this->OutTensor->Dx, this->InTensor->N);
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
    mkl_set_num_threads_local(this->InTensor->NumThreads);
  }

  ~SpMMSpMMMKL() {
    mkl_free(A);
    mkl_free(B);
  }
};

#endif // SPARSE_FUSION_SPMM_SPMM_MKL_DEMO_UTILS_H

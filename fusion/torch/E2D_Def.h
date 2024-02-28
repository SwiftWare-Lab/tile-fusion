//
// Created by salehm32 on 10/01/24.
//
#include <cstddef>
#include <cassert>
#ifndef SPARSE_FUSION_E2D_DEF_H
#define SPARSE_FUSION_E2D_DEF_H

struct FloatDense {
  size_t row;
  size_t col;
  size_t lda;
  float *a;

  FloatDense(size_t M, size_t N, size_t LDA) : row(M), col(N), lda(LDA) {
    assert(lda == 1 || lda == N);
    if (row > 0 && col > 0)
      a = new float[row * col]();
  }

  ~FloatDense() {
    if (row > 0 && col > 0)
      delete[]a;
  }
};

#endif // SPARSE_FUSION_E2D_DEF_H

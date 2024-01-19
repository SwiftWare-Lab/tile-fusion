//
// Created by kazem on 1/19/24.
//

#ifndef SPARSE_FUSION_JACOBI_H
#define SPARSE_FUSION_JACOBI_H

namespace sym_lib {


inline double ResidualMutipleCols(int M, int K, const double *X0, const double *X1) {
  double max = 0.0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; ++j) {
      if (std::fabs(X0[i * K + j] - X1[i * K + j]) > max)
        max = std::fabs(X0[i * K + j] - X1[i * K + j]);
    }
  }
  return max;
}


inline void jacobiIterationCsr(int m, const int *Ap, const int *Ai, const double *Ax, double *Diags,
                        double *Xx, const double *B, int BCol){
  // x = (b - dot(R,x)) / D
#pragma omp parallel for
  for (int j = 0; j < m; ++j) {
    for(int i=0; i<BCol; ++i) {
      long double sum = 0.0;
      auto *x = Xx + i; //Xx[0][i];
      auto *b = B + i; //B[0][i];
      for (int k = Ap[j]; k < Ap[j + 1]; ++k) {
        sum += Ax[k] * x[Ai[k]*BCol];
      }
      assert(!std::isnan(sum));
      x[j*BCol] = (b[j*BCol] - sum) / Diags[j];
    }
  }
}

///
/// \param m
/// \param n
/// \param Ap
/// \param Ai
/// \param Ax
/// \param x
/// \param b
/// \param eps
/// \param n_iter
/// \param WS temp array of size m+nnzA + m*k
/// \return
inline int jacobiCSR(int m, int k, const int *Ap, const int *Ai, const double *Ax,
                    double *X, const double *B, int BCol,
                    double &Eps, int NIter, double *WS){
  double *diagVals = WS;
  double *xp = WS+m; // previous X
  double *r = WS+ m + (m*k);

  for (int i = 0; i < m; ++i) {
    for (int j = Ap[i]; j < Ap[i+1]; ++j) {
      if(Ai[j] == i) {
        // D = diag(A)
        diagVals[i] = Ax[j];
        r[j] = 0;
      } else {
        // R = A - diagflat(D)
        r[j] = Ax[j];
      }
    }
  }
  // x = (b - dot(R,x)) / D
  jacobiIterationCsr(m, Ap, Ai, r, diagVals, xp, B, BCol);
  // copy xp to X
  std::memcpy(X, xp, sizeof(double)*m*k);
  for (int i = 0; i < NIter; ++i) {
    jacobiIterationCsr(m, Ap, Ai, r, diagVals, X, B, BCol);
    double res = ResidualMutipleCols(m, k, X, xp);
    if(res < Eps)
      return i+1;
    std::memcpy(xp, X, sizeof(double)*m*k);
  }
  return 0;
}


}


#endif // SPARSE_FUSION_JACOBI_H

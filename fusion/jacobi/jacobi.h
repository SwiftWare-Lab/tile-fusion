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
                        double *Xx, double *XxIn, const double *B, int BCol){
  // x = (b - dot(R,x)) / D
#pragma omp parallel for
  for (int j = 0; j < m; ++j) {
    for(int i=0; i<BCol; ++i) {
      long double sum = 0.0;
      auto *xIn = XxIn + i; //Xx[0][i];
      auto *xOut = Xx + i;
      auto *b = B + i; //B[0][i];
      for (int k = Ap[j]; k < Ap[j + 1]; ++k) {
        sum += Ax[k] * xIn[Ai[k]*BCol];
      }
      assert(!std::isnan(sum));
      xOut[j*BCol] = (b[j*BCol] - sum) / Diags[j];
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
  jacobiIterationCsr(m, Ap, Ai, r, diagVals, X, xp, B, BCol);
  //sym_lib::print_dense(m, BCol, 1, B);
  //std::cout<<" \n ==== \n";
  //sym_lib::print_dense(m, BCol, 1, X);
  // copy X to xp
  std::memcpy(xp, X, sizeof(double)*m*k);
  for (int i = 0; i < NIter; ++i) {
    jacobiIterationCsr(m, Ap, Ai, r, diagVals, X, xp, B, BCol);
    double res = ResidualMutipleCols(m, k, xp, X);
    if(res < Eps)
      return i+1;
    std::memcpy(xp, X, sizeof(double)*m*k);
  }
  return 0;
}


}


#endif // SPARSE_FUSION_JACOBI_H

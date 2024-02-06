//
// Created by kazem on 1/19/24.
//

#include <immintrin.h>

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

void jacobiTwoIterationsFusedCsr(int M, const int *Ap, const int *Ai,
                                 const double *Ax, const double *Bx, int BCol, double* XIn,
                                 double *X1, double *X2, double *Diags,
                                 int LevelNo, const int *LevelPtr,
                                 const int *ParPtr, const int *Partition,
                                 const int *ParType, int NThreads) {
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int j = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            for (int i = 0; i < BCol; ++i) {
              long double sum = 0.0;
              auto *xIn = XIn + i;
              auto *x1 = X1 + i;// Xx2[0][i];
              auto *b = Bx + i; // B[0][i];
              for (int k = Ap[j]; k < Ap[j + 1]; ++k) {
                sum += Ax[k] * xIn[Ai[k] * BCol];
              }
              assert(!std::isnan(sum));
              x1[j * BCol] = (b[j * BCol] - sum) / Diags[j];
            }
          } else {
            for (int i = 0; i < BCol; ++i) {
              long double sum = 0.0;
              auto *xIn = X1 + i;// Xx2[0][i];
              auto *x2 = X2 + i;
              auto *b = Bx + i; // B[0][i];
              for (int k = Ap[j]; k < Ap[j + 1]; ++k) {
                sum += Ax[k] * xIn[Ai[k] * BCol];
              }
              assert(!std::isnan(sum));
              x2[j * BCol] = (b[j * BCol] - sum) / Diags[j];
            }
          }
        }
      }
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

inline int jacobiBiIterationFusedCSR(int m, int k, const int *Ap, const int *Ai,
                                     const double *Ax, double *X1, double *X2,
                                     const double *B, int BCol, double &Eps,
                                     int NIter, double *WS, int LevelNo,
                                     const int *LevelPtr, const int *ParPtr,
                                     const int *Partition, const int *ParType,
                                     int NThreads) {
  double *diagVals = WS;
  double *xp = WS + m; // previous X2
  double *r = WS + m + (m * k);

  for (int i = 0; i < m; ++i) {
    for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
      if (Ai[j] == i) {
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
  jacobiIterationCsr(m, Ap, Ai, r, diagVals, X2, xp, B, BCol);
  //sym_lib::print_dense(m, BCol, 1, B);
  //std::cout<<" \n ==== \n";
  //sym_lib::print_dense(m, BCol, 1, X);
  // copy X to xp
  std::memcpy(xp, X2, sizeof(double)*m*k);
  for (int i = 0; i < NIter; i+=2) {
    jacobiTwoIterationsFusedCsr(m, Ap, Ai, r, B, BCol, xp,X1, X2, diagVals, LevelNo,
                                LevelPtr, ParPtr, Partition, ParType, NThreads);

    double res = ResidualMutipleCols(m, k, X2, X1);
    if (res < Eps)
      return i + 2;
    std::memcpy(xp, X2, sizeof(double) * m * k);
//    std::memcpy(X2, X1, sizeof(double) * m * k);
  }
  return 0;
}

} // namespace sym_lib

#endif // SPARSE_FUSION_JACOBI_H

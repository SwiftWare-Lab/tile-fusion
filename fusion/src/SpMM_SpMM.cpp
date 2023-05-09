//
// Created by kazem on 02/05/23.
//

namespace swiftware {
namespace sparse {

/// C = A*B, where A is sparse CSR MxK and B (K x N) and C (MxN) are Dense
void spmmCsrSequential(int M, int N, int K,
                       const int *Ap, const int *Ai, const double *Ax,
                       const double *Bx, double *Cx){
  for (int i = 0; i < M; ++i) {
    for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
      for (int k = 0; k < K; ++k) {
        Cx[i * K + k] += Ax[j] * Bx[Ai[j] * K + k];
      }
    }
  }
}

void spmmCsrParallel(int M, int N, int K,
                     const int *Ap, const int *Ai, const double *Ax,
                     const double *Bx, double *Cx, int NThreads) {
#pragma omp parallel for num_threads(NThreads)
  for (int i = 0; i < M; ++i) {
    for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
      for (int k = 0; k < K; ++k) {
        Cx[i * K + k] += Ax[j] * Bx[Ai[j] * K + k];
      }
    }
  }
}


/// D = B*A*C
void spmmCsrSpmmCsrFused(int M, int N, int K, int L,
                         const int *Ap, const int *Ai, const double *Ax,
                         const int *Bp, const int *Bi,const double *Bx,
                         const double *Cx,
                         double *Dx,
                         double *ACx,
                         int LevelNo, const int *LevelPtr, const int *ParPtr,
                         const int *Partition, const int *ParType) {
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel
    {
#pragma omp for schedule(auto) nowait
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int aij = Ai[j] * M;
              for (int kk = 0; kk < M; ++kk) {
                ACx[i * M + kk] += Ax[j] * Cx[aij + kk];
              }
            }
          } else {
            for (int k = Bp[i]; k < Bp[i + 1]; k++) {
              int bij = Bi[k] * M;
              for (int kk = 0; kk < M; ++kk) {
                Dx[i * M + kk] += Bx[k] * ACx[bij + kk];
              }
            }
          }
        }
      }
    }
  }
}


} // namespace sparse
} // namespace swiftware
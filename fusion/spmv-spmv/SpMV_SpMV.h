//
// Created by salehm32 on 08/12/23.
//

#ifndef SPARSE_FUSION_SPMV_SPMV_UTILS_H
#define SPARSE_FUSION_SPMV_SPMV_UTILS_H

#include <omp.h>
void spMVCsrSequential(int M, int K, const int *Ap, const int *Ai,
                       const double *Ax, const double *Bx, double *Cx);
void spMVCsrParallel(int M, int K, const int *Ap, const int *Ai,
                     const double *Ax, const double *Bx, double *Cx,
                     int NumThreads);

void spMVCsrSequential(int M, int K, const int *Ap, const int *Ai,
                       const double *Ax, const double *Bx, double *Cx) {
  for (int i = 0; i < M; i++) {
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      Cx[i] += Ax[j] * Bx[Ai[j]];
    }
  }
}

void spMVCsrParallel(int M, int K, const int *Ap, const int *Ai,
                     const double *Ax, const double *Bx, double *Cx,
                     int NumThreads) {
#pragma omp parallel num_threads(NumThreads)
  {
#pragma omp for
    for (int i = 0; i < M; i++) {
      for (int j = Ap[i]; j < Ap[i + 1]; j++) {
        Cx[i] += Ax[j] * Bx[Ai[j]];
      }
    }
  }
}

void spMVCsrSpMCsrFused(int M, int K, int L, const int *Ap, const int *Ai,
                        const double *Ax, const int *Bp, const int *Bi,
                        const double *Bx, const double *Cx, double *Dx,
                        double *ACx, int LevelNo, const int *LevelPtr,
                        const int *ParPtr, const int *Partition,
                        const int *ParType, int NThreads) {

  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              ACx[i] += Ax[j] * Cx[Ai[j]];
            }
          } else {
            for (int k = Bp[i]; k < Bp[i + 1]; k++) {
              Dx[i] += Bx[k] * ACx[Bi[k]];
            }
          }
        }
      }
    }
  }
}

void spMVCsrSpMVCscFusedColored(int M, int K, int L, const int *Ap,
                                const int *Ai, const double *Ax, const int *Bp,
                                const int *Bi, const double *Bx,
                                const double *Cx, double *Dx, double *ACx,
                                int LevelNo, const int *LevelPtr, const int *Id,
                                int TileSize, int NThreads) {
  int lastTileSize = M % TileSize;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        int id = Id[j1];
        int i = id * TileSize;
        for (int ii = 0; ii < TileSize; ++ii) {
          auto ipii = i + ii;
          // first SpMV
          for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
            ACx[ipii] += Ax[j] * Cx[Ai[j]];
          }
          // second SpMV CSC
          for (int k = Bp[ipii]; k < Bp[ipii + 1];
               k++) { // for each column of B
            int bij = Bi[k];
            Dx[Bi[k]] += Bx[k] * ACx[ipii];
          }
        }
      }
    }
  }
  int i = M - lastTileSize;
  for (int ii = 0; ii < lastTileSize; ++ii) {
    auto ipii = i + ii;
    // first SpMV
    for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
      ACx[ipii] += Ax[j] * Cx[Ai[j]];
    }
    // second SpMV CSC
    for (int k = Bp[ipii]; k < Bp[ipii + 1]; k++) { // for each column of B
      int bij = Bi[k];
      Dx[Bi[k]] += Bx[k] * ACx[ipii];
    }
  }
}

void spMVCsrSpMVCscFusedColoredWithReduction(int M, int K, int L, const int *Ap,
                                const int *Ai, const double *Ax, const int *Bp,
                                const int *Bi, const double *Bx,
                                const double *Cx, double *Dx, double *ACx,
                                int LevelNo, const int *LevelPtr, const int *Id,
                                int TileSize, int NThreads) {
  int lastTileSize = M % TileSize;
  double *tempResults = new double[LevelNo*L];
#pragma omp parallel num_threads(NThreads)
  {
#pragma omp for
  for (int i1 = 0; i1 < LevelNo; ++i1) {
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        int id = Id[j1];
        int i = id * TileSize;
        for (int ii = 0; ii < TileSize; ++ii) {
          auto ipii = i + ii;
          // first SpMV
          for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
            ACx[ipii] += Ax[j] * Cx[Ai[j]];
          }
          // second SpMV CSC
          for (int k = Bp[ipii]; k < Bp[ipii + 1];
               k++) { // for each column of B
            int bij = Bi[k];
            tempResults[i1*L + Bi[k]] += Bx[k] * ACx[ipii];
          }
        }
      }
    }
  }
  for (int i1 = 0; i1 < LevelNo; ++i1) {
    for(int j = 0; j < L; j++){
      Dx[j] += tempResults[i1*L + j];
    }
  }
  delete[] tempResults;
  int i = M - lastTileSize;
  for (int ii = 0; ii < lastTileSize; ++ii) {
    auto ipii = i + ii;
    // first SpMV
    for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
      ACx[ipii] += Ax[j] * Cx[Ai[j]];
    }
    // second SpMV CSC
    for (int k = Bp[ipii]; k < Bp[ipii + 1]; k++) { // for each column of B
      int bij = Bi[k];
      Dx[Bi[k]] += Bx[k] * ACx[ipii];
    }
  }
}

void spMVCsrSpMVCsrSeparatedFused(int M, int K, int L, const int *Ap,
                                  const int *Ai, const double *Ax,
                                  const int *Bp, const int *Bi,
                                  const double *Bx, const double *Cx,
                                  double *Dx, double *ACx, int LevelNo,
                                  const int *LevelPtr, const int *ParPtr,
                                  const int *Partition, const int *ParType,
                                  const int *MixPtr, int NThreads) {
  int numKer = 2;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {

        // Loop 1
        for (int k1 = ParPtr[j1]; k1 < MixPtr[j1 * numKer]; ++k1) {
          int i = Partition[k1];
          for (int j = Ap[i]; j < Ap[i + 1]; j++) {
            ACx[i] += Ax[j] * Cx[Ai[j]];
          }
        } // end loop 1

        // Loop 2
        for (int k1 = MixPtr[j1 * numKer]; k1 < MixPtr[j1 * numKer + 1]; ++k1) {
          int i = Partition[k1];
          for (int k = Bp[i]; k < Bp[i + 1]; k++) {
            Dx[i] += Bx[k] * ACx[Bi[k]];
          }
        } // end loop 2
      }
    }
  }
}

#endif // SPARSE_FUSION_SPMV_SPMV_UTILS_H

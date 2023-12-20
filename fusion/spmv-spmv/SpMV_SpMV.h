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

void spMVCsrSpMCsrFusedRegisterReuseBanded(
    int M, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads) {
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; j1+=3) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          for (int j = Ap[i]; j < Ap[i + 1]; j++) {
            ACx[i] += Ax[j] * Cx[Ai[j]];
          }
        }
        for (int k1 = ParPtr[j1 + 1]; k1 < ParPtr[j1 + 2]; ++k1) {
          double temp = 0;
          int i = Partition[k1];
          int i2 = Partition[k1 + 1];
          for (int j = Ap[i]; j < Ap[i + 1]; j++) {
            temp += Ax[j] * Cx[Ai[j]];
          }
          for (int k = Bp[i2]; k < Bp[i2 + 1] - 1; k++) {
            Dx[i2] += Bx[k] * ACx[Bi[k]];
          }
          Dx[i2] += Bx[Bp[i2 + 1] - 1] * temp;
          ACx[i] = temp;
          k1++;
        }
        for (int k1 = ParPtr[j1 + 2]; k1 < ParPtr[j1 + 3]; ++k1) {
          int i = Partition[k1];
          for (int k = Bp[i]; k < Bp[i + 1]; k++) {
            Dx[i] += Bx[k] * ACx[Bi[k]];
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
          double ACxi = 0;
          for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
            ACxi += Ax[j] * Cx[Ai[j]];
          }
          // second SpMV CSC
          for (int k = Bp[ipii]; k < Bp[ipii + 1];
               k++) { // for each column of B
            int bij = Bi[k];
            Dx[Bi[k]] += Bx[k] * ACxi;
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

void spMVCsrSpMVCscFusedColoredWithReduction(
    int M, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr, const int *Id,
    int TileSize, int NThreads) {
  int lastTileSize = M % TileSize;
  double *tempResults = new double[LevelNo * L];
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
            tempResults[i1 * L + Bi[k]] += Bx[k] * ACx[ipii];
          }
        }
      }
    }
  }
  for (int i1 = 0; i1 < LevelNo; ++i1) {
    for (int j = 0; j < L; j++) {
      Dx[j] += tempResults[i1 * L + j];
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

void spmvCsrSpmvCsrTiledFusedRedundantBanded(
    int M, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType,
    const int *MixPtr, int NThreads, int MTile, double *Ws, int L1MarginSize) {
  int numKer = 2;
  int mBound = M - M % MTile;
  auto *cxBufAll = Ws; // new double[(MTile+2) * NThreads]();
  // First level benefits from Fusion
  int iBoundBeg = LevelPtr[0], iBoundEnd = LevelPtr[1];
#pragma omp parallel num_threads(NThreads)
  {
#pragma omp for
    for (int j1 = iBoundBeg; j1 < iBoundEnd; ++j1) {
      auto *cxBuf = cxBufAll + omp_get_thread_num() * (MTile + L1MarginSize);
      int kBegin = ParPtr[j1], kEnd = MixPtr[j1 * numKer];
      int ii = Partition[kBegin]; // first iteration of tile
      int mTileLoc = kEnd - kBegin;
      // if(ii >= mBound) continue;
      for (int i = 0; i < mTileLoc; ++i) {
        int iipi = ii + i;
        for (int j = Ap[iipi]; j < Ap[iipi + 1]; ++j) {
          cxBuf[i] += Ax[j] * Cx[Ai[j]];
        }
      }
      // second loop
      int kEndL2 = MixPtr[j1 * numKer + 1];
      for (int k1 = kEnd; k1 < kEndL2; k1++) { // i-loop
        int i = Partition[k1];
        for (int j = Bp[i]; j < Bp[i + 1]; j++) {
          int bij = Bi[j] - ii;
          //            assert(bij < mTileLoc + 1 && bij >= 0); // stays within
          //            the tile i
          Dx[i] += Bx[j] * cxBuf[bij];
          // cxBuf[bij + k] = 0;
        }
      }
      std::fill_n(cxBuf, mTileLoc, 0.0);
    }
  }
}


void spmvCsrSpmvCsrTiledFusedRedundantBandedV2(
    int M, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, int NThreads, int MTile, double *Ws) {
  auto *cxBufAll = Ws; // new double[(MTile+2) * NThreads]();
  // First level benefits from Fusion
#pragma omp parallel num_threads(NThreads)
  {
#pragma omp for
    for (int i = 0; i < M; i+=MTile) {
      auto *cxBuf = cxBufAll + omp_get_thread_num() * (MTile + 2);
      int l1Begin = std::max(i-1, 0), l1End = std::min(i + MTile + 1, M);
      int l2Begin = i, l2End = std::min(i + MTile, M);
      int mTileLoc = l1End - l1Begin;
      // if(ii >= mBound) continue;
      for (int ii = l1Begin; ii < l1End; ++ii) {
        for (int j = Ap[ii]; j < Ap[ii + 1]; ++j) {
          cxBuf[ii - l1Begin] += Ax[j] * Cx[Ai[j]];
        }
      }
      // second loop
      for (int ii = l2Begin; ii < l2End; ii++) { // i-loop
        for (int j = Bp[ii]; j < Bp[ii + 1]; j++) {
          int bij = Bi[j] - l1Begin;
          Dx[ii] += Bx[j] * cxBuf[bij];
        }
      }
      std::fill_n(cxBuf, mTileLoc, 0.0);
    }
  }
}

void spmvCsrSpmvCsrTiledFusedRedundantGeneral(
    int M, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, int NThreads, int MTile, double *Ws, int *L1TileLowBounds, int *L1TileHighBounds, int MaxL1TileSize) {
  auto *cxBufAll = Ws; // new double[MaxL1TileSize * NThreads]();
  // First level benefits from Fusion
#pragma omp parallel num_threads(NThreads)
  {
#pragma omp for
    for (int i = 0; i < M; i+=MTile) {
      auto *cxBuf = cxBufAll + omp_get_thread_num() * MaxL1TileSize;
      int l1Begin = L1TileLowBounds[i/MTile], l1End = L1TileHighBounds[i/MTile];
      int l2Begin = i, l2End = std::min(i + MTile, M);
      int mTileLoc = l1End - l1Begin;
      // if(ii >= mBound) continue;
      for (int ii = l1Begin; ii < l1End; ++ii) {
        for (int j = Ap[ii]; j < Ap[ii + 1]; ++j) {
          cxBuf[ii - l1Begin] += Ax[j] * Cx[Ai[j]];
        }
      }
      // second loop
      for (int ii = l2Begin; ii < l2End; ii++) { // i-loop
        for (int j = Bp[ii]; j < Bp[ii + 1]; j++) {
          int bij = Bi[j] - l1Begin;
          Dx[ii] += Bx[j] * cxBuf[bij];
        }
      }
      std::fill_n(cxBuf, MaxL1TileSize, 0.0);
    }
  }
}

void spmmCsrSpmmCsrTiledFusedRedundantGeneral(int M, int K, int L,
                                              const int *Ap, const int *Ai, const double *Ax,
                                              const int *Bp, const int *Bi,const double *Bx,
                                              const double *Cx,
                                              double *Dx,
                                              double *ACx,
                                              int LevelNo, const int *LevelPtr, const int *ParPtr,
                                              const int *Partition, const int *ParType, const int*MixPtr,
                                              int NThreads, double *Ws) {
  int numKer=2;
  auto *cxBufAll = Ws;//new double[MTile * NTile * NThreads]();
  // First level benefits from Fusion
  int iBoundBeg = LevelPtr[0], iBoundEnd = LevelPtr[1];
#pragma omp parallel num_threads(NThreads)
  {
#pragma omp  for
    for (int j1 = iBoundBeg; j1 < iBoundEnd; ++j1) {
      auto *cxBuf = cxBufAll + omp_get_thread_num() * 2 * M;
      int kBegin = ParPtr[j1], kEnd = MixPtr[j1 * numKer];
      int ii = Partition[kBegin]; // first iteration of tile
      int mTileLoc = kEnd - kBegin;
        // first loop, for every k-tile
        for(int k1 = kBegin; k1 < kEnd; k1++) { // i-loop
          int i = Partition[k1];
          // reset cxBuf, I used dot product to avoid the following
            double acc = 0;
            for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
              int aij = Ai[j];
              acc += Ax[j] * Cx[aij];
              //ACx[iipi * N + k + kk] = tmp;
            }
            cxBuf[i] = acc;
        }
        // second loop
        int kEndL2 = MixPtr[j1 * numKer + 1];
        for(int k1 = kEnd; k1 < kEndL2; k1++) { // i-loop
          int i = Partition[k1];
          for (int j = Bp[i]; j < Bp[i + 1]; j++) {
            int bij = Bi[j];
            int inkk = i;
              Dx[inkk] += Bx[j] * cxBuf[bij];
          }
        }
        //std::fill_n(cxBuf, M * NTile, 0.0);

        //            std::cout<<"\n=============\n";
        //            for(int i = 0; i < mTileLoc; ++i) {
        //              for(int k = 0; k < NTile; ++k) {
        //                std::cout << Dx[i * NTile + k] << " ";
        //              }
        //              std::cout << std::endl;
        //            }
    }
  }
}

#endif // SPARSE_FUSION_SPMV_SPMV_UTILS_H

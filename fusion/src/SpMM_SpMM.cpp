//
// Created by kazem on 02/05/23.
//
#ifdef PROF_WITH_PAPI
#include "papi_wrapper.h"
#else
#define pw_init_instruments
#define pw_start_instruments_loop(th)
#define pw_stop_instruments_loop(th)
#endif
#include <cassert>
#include <iostream>
#include <omp.h>
namespace swiftware {
namespace sparse {

/// C = A*B, where A is sparse CSR MxK and B (K x N) and C (MxN) are Dense
void spmmCsrSequential(int M, int N, int K,
                       const int *Ap, const int *Ai, const double *Ax,
                       const double *Bx, double *Cx){
  for (int i = 0; i < M; ++i) {
    for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
      int aij = Ai[j] * N;
      for (int k = 0; k < N; ++k) {
        assert(i * N + k < M * N);
        Cx[i * N + k] += Ax[j] * Bx[aij + k];
      }
    }
  }
}



void spmmCsrParallel(int M, int N, int K,
                     const int *Ap, const int *Ai, const double *Ax,
                     const double *Bx, double *Cx, int NThreads) {
  pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
  {
    pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
    for (int i = 0; i < M; ++i) {
      for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
        int aij = Ai[j] * N;
        for (int k = 0; k < N; ++k) {
          Cx[i * N + k] += Ax[j] * Bx[aij + k];
        }
      }
    }
    pw_stop_instruments_loop(omp_get_thread_num());
  }
}

void spmmCsrParallelTiled(int M, int N, int K,
                          const int *Ap, const int *Ai, const double *Ax,
                          const double *Bx, double *Cx, int NThreads,
                          int MTile, int NTile) {
  int mBound = M - M % MTile;
  int nt=0;
#pragma omp parallel num_threads(NThreads)
  { nt = omp_get_num_threads(); }
  auto *cxBufAll = new double[MTile * NTile * nt ]();
  //omp_set_nested(1);

  pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
  {
    pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
    for (int ii = 0; ii < mBound; ii+=MTile) {
      for (int kk = 0; kk < N; kk+=NTile) {
        // print the thread id
        //std::cout << "------------- Thread " << omp_get_thread_num() << " is doing " << ii << " " << kk << std::endl;
        //assert(omp_get_thread_num() < nt);
        auto *cxBuf = cxBufAll + omp_get_thread_num() * MTile * NTile;
        for (int i = 0; i < MTile; ++i) {
          for (int j = Ap[ii + i]; j < Ap[ii + i + 1]; ++j) {
            int aij = Ai[j] * N;
            for (int k = 0; k < NTile; ++k) {
              cxBuf[i * NTile + k] += Ax[j] * Bx[aij + k];
            }
          }
        }
        // copy to C
        for (int i = ii, ti = 0; i < ii + MTile; ++i, ++ti) {
          for (int k = kk, tk = 0; k < kk + NTile; ++k, ++tk) {
            Cx[i * N + k] += cxBuf[ti * NTile + tk];
            cxBuf[ti * NTile + tk] = 0;
          }
        }

        //        for (int i = ii; i < ii + MTile; ++i) {
        //          for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
        //            int aij = Ai[j] * N;
        //            for (int k = kk; k < kk + NTile; ++k) {
        //              Cx[i * N + k] += Ax[j] * Bx[aij + k];
        //            }
        //          }
        //        }
      }
    } // end ii
    pw_stop_instruments_loop(omp_get_thread_num());
  }
  // Remaining rows
  for (int i = mBound; i < M; ++i) {
    for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
      int aij = Ai[j] * N;
      for (int k = 0; k < N; ++k) {
        Cx[i * N + k] += Ax[j] * Bx[aij + k];
      }
    }
  }
  delete [] cxBufAll;
}


void spmmCsrInnerProductParallel(int M, int N, int K,
                                 const int *Ap, const int *Ai, const double *Ax,
                                 const double *Bx, double *Cx, int NThreads) {
  pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
  {
    pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < N; ++k) {
        auto cik = Cx[i * N + k];
        for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
          int aij = Ai[j] * N;
          cik += Ax[j] * Bx[aij + k];// C[i][k] += A[i][j] * B[j][k];
        }
        Cx[i * N + k] = cik;
      }
    }
    pw_stop_instruments_loop(omp_get_thread_num());
  }
}


void spmmCsrInnerProductTiledCParallel(int M, int N, int K,
                                       const int *Ap, const int *Ai, const double *Ax,
                                       const double *Bx, double *Cx, int NThreads
                                       ,int MTile, int NTile) {
  auto *cTile = new double[MTile * NTile]();
  int nTailBeg = N - (N % NTile), mTailBeg = M - (M % MTile);
  pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
  {
    pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
    for (int i = 0; i < mTailBeg; i += MTile) {
      for (int k = 0; k < nTailBeg; k += NTile) {
        // perform computation per C tile
        for (int ii = 0; ii < MTile; ii++) {
          for (int kk = 0; kk < NTile; kk++) {
            auto cik = 0;//cTile[ii * NTile + kk];
            for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; ++j) {
              int aij = Ai[j] * N;
              cik += Ax[j] * Bx[aij + k + kk]; // C[i][k] += A[i][j] * B[j][k];
            }
            Cx[(i + ii) * N + (k + kk)] += cik;
            //cTile[ii * NTile + kk] += cik;
          }
        }
        // copy ctile to C
        //        for (int ii = 0; ii < MTile; ii++) {
        //          for (int kk = 0; kk < NTile; kk++) {
        //            Cx[(i + ii) * N + (k + kk)] += cTile[ii * NTile + kk];
        //            cTile[ii * NTile + kk] = 0;
        //          }
        //        }
      }
      // tail iterations for k
      for (int ii = 0; ii < MTile; ii++) {
        for (int k = nTailBeg; k < N; ++k) {
          auto cik = 0;//Cx[(i+ii) * N + k];
          for (int j = Ap[i+ii]; j < Ap[i + ii + 1]; ++j) {
            int aij = Ai[j] * N;
            cik += Ax[j] * Bx[aij + k]; // C[i][k] += A[i][j] * B[j][k];
          }
          Cx[(i+ii) * N + k] += cik;
        }
      }
    }
    pw_stop_instruments_loop(omp_get_thread_num());
  }
  // tail iterations for i
#pragma omp parallel for num_threads(NThreads)
  for (int i = mTailBeg; i < M; i++){
    for (int k = 0; k < N; ++k) {
      auto cik = Cx[i * N + k];
      for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
        int aij = Ai[j] * N;
        cik += Ax[j] * Bx[aij + k]; // C[i][k] += A[i][j] * B[j][k];
      }
      Cx[i * N + k] += cik;
    }
  }
  delete[] cTile;
}

/// D = B*A*C
void spmmCsrSpmmCsrFused(int M, int N, int K, int L,
                         const int *Ap, const int *Ai, const double *Ax,
                         const int *Bp, const int *Bi,const double *Bx,
                         const double *Cx,
                         double *Dx,
                         double *ACx,
                         int LevelNo, const int *LevelPtr, const int *ParPtr,
                         const int *Partition, const int *ParType,
                         int NThreads) {
  pw_init_instruments;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp  for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int aij = Ai[j] * N;
              for (int kk = 0; kk < N; ++kk) {
                ACx[i * N + kk] += Ax[j] * Cx[aij + kk];
              }
            }
          } else {
            for (int k = Bp[i]; k < Bp[i + 1]; k++) {
              int bij = Bi[k] * N;
              for (int kk = 0; kk < N; ++kk) {
                Dx[i * N + kk] += Bx[k] * ACx[bij + kk];
              }
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}


void spmmCsrSpmmCsrTiledFused(int M, int N, int K, int L,
                              const int *Ap, const int *Ai, const double *Ax,
                              const int *Bp, const int *Bi,const double *Bx,
                              const double *Cx,
                              double *Dx,
                              double *ACx,
                              int LevelNo, const int *LevelPtr, const int *ParPtr,
                              const int *Partition, const int *ParType,
                              int NThreads, int MTile, int NTile, double *Ws) {
  pw_init_instruments;
  int mBound = M - M % MTile;
  auto *cxBufAll = Ws;//new double[MTile * NTile * NThreads]();
  // First level benefits from Fusion
  int iBoundBeg = LevelPtr[0], iBoundEnd = LevelPtr[1];
  //#pragma omp parallel num_threads(NThreads)
  {
    //#pragma omp  for
    for (int j1 = iBoundBeg; j1 < iBoundEnd; ++j1) {
      auto *cxBuf = cxBufAll + omp_get_thread_num() * MTile * NTile;

      int kBegin = ParPtr[j1], kEnd = ParPtr[j1 + 1];
      if(kEnd - kBegin == 0) continue;
      if(kEnd - kBegin < MTile)
        continue ;
      int ii = Partition[kBegin]; // first iteration of tile
      if(ii >= mBound) continue;

      for (int kk = 0; kk < N; kk += NTile) {
        // first loop, for every k-tile
        for (int i = 0; i < MTile; ++i) {
          int iipi = ii + i;
          for (int j = Ap[iipi]; j < Ap[iipi + 1]; ++j) {
            int aij = Ai[j] * N;
            for (int k = 0; k < NTile; ++k) {
              //auto tmp = Ax[j] * Cx[aij + k];
              cxBuf[i * NTile + k] += Ax[j] * Cx[aij + k];
              //ACx[iipi * N + k + kk] = tmp;
            }
          }
        }
        //}

        // second loop
        //for (int kk = 0; kk < N; kk += NTile) {
        for(int k1 = kBegin+MTile; k1 < kEnd; k1++) {// i-loop
          int i = Partition[k1];

          for (int j = Bp[i]; j < Bp[i + 1]; j++) {
            int bij = Bi[j]-ii;
            assert(bij < MTile && bij >= 0); // stays within the tile i
            bij *= NTile;
            int inkk = i * N + kk;
            for (int k = 0; k < NTile; ++k) {
              Dx[inkk + k] += Bx[j] * cxBuf[bij + k];
            }
          }
        }
        // }

        // copy to ACx for the next wavefront
        //for (int kk = 0; kk < N; kk += NTile) {
        //        for (int i = ii, ti = 0; i < ii + MTile; ++i, ++ti) {
        //          for (int k = kk, tk = 0; k < kk + NTile; ++k, ++tk) {
        //            ACx[i * N + k] = cxBuf[ti * NTile + tk];
        //            cxBuf[ti * NTile + tk] = 0;
        //          }
        //        }
      }
    }
  }
  //delete[] cxBufAll;
  int loopBeg = ParPtr[LevelPtr[1]], loopEnd = ParPtr[LevelPtr[LevelNo]];
  //for (int i1 = 1; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
  {
#pragma omp  for
    //for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
    //for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
    for(int k1 = loopBeg; k1 < loopEnd; ++k1) {
      int i = Partition[k1];
      //          int t = ParType[k1];
      //          if (t == 0) {
      //            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      //              int aij = Ai[j] * N;
      //              for (int kk = 0; kk < N; ++kk) {
      //                ACx[i * N + kk] += Ax[j] * Cx[aij + kk];
      //              }
      //            }
      //          } else {
      //#pragma omp parallel for
      for (int kk = 0; kk < N; kk += NTile) {
        for (int j = Bp[i]; j < Bp[i + 1]; j++) {
          int bij = Bi[j] * N + kk, dik = i * N + kk;
          for (int k = 0; k < NTile; ++k) {
            Dx[dik + k] += Bx[j] * ACx[bij + k];
          }
        }
      }

      //          for (int k = Bp[i]; k < Bp[i + 1]; k++) {
      //            int bij = Bi[k] * N;
      //            for (int kk = 0; kk < N; ++kk) {
      //              Dx[i * N + kk] += Bx[k] * ACx[bij + kk];
      //            }
      //          }

    }
    //}
  }
  //}
}


/// This works for tri-diagonal only
void spmmCsrSpmmCsrTiledFusedBanded(int M, int N, int K, int L,
                                    const int *Ap, const int *Ai, const double *Ax,
                                    const int *Bp, const int *Bi,const double *Bx,
                                    const double *Cx,
                                    double *Dx,
                                    double *ACx,
                                    int LevelNo, const int *LevelPtr, const int *ParPtr,
                                    const int *Partition, const int *ParType, const int*MixPtr,
                                    int NThreads, int MTile, int NTile, double *Ws) {
  pw_init_instruments;
  // 0, 1, 2 of loop 1 : 0 and 1 loop 2
  int i = 0;
  // i = i+0
  int iIdx = i;
  double acxI0 = 0, acxI1 = 0, acxI2 = 0, acxI3 = 0;
  double acxI4 = 0, acxI5 = 0, acxI6 = 0, acxI7 = 0;
  double acxI8 = 0, acxI9 = 0, acxI10 = 0, acxI11 = 0;
  for (int kk = 0; kk < N; kk+=NTile) {
    iIdx = i; // i = i+0, first dependent iteration
    for (int j = Ap[iIdx]; j < Ap[iIdx + 1]; j++) {
      int aij = Ai[j] * N;
      //ACx[i * N + kk] += Ax[j] * Cx[aij + kk];
      acxI0 += Ax[j] * Cx[aij + kk];
      acxI1 += Ax[j] * Cx[aij + kk + 1];
      acxI2 += Ax[j] * Cx[aij + kk + 2];
      acxI3 += Ax[j] * Cx[aij + kk + 3];
    }

    iIdx = i+1; // i = i+1, second dependent iteration
    for (int j = Ap[iIdx]; j < Ap[iIdx + 1]; j++) {
      int aij = Ai[j] * N;
      //ACx[i * N + kk] += Ax[j] * Cx[aij + kk];
      acxI4 += Ax[j] * Cx[aij + kk];
      acxI5 += Ax[j] * Cx[aij + kk + 1];
      acxI6 += Ax[j] * Cx[aij + kk + 2];
      acxI7 += Ax[j] * Cx[aij + kk + 3];
    }

    iIdx = i+2; // i = i+1, third dependent iteration
    for (int j = Ap[iIdx]; j < Ap[iIdx + 1]; j++) {
      int aij = Ai[j] * N;
      //ACx[i * N + kk] += Ax[j] * Cx[aij + kk];
      acxI8 += Ax[j] * Cx[aij + kk];
      acxI9 += Ax[j] * Cx[aij + kk + 1];
      acxI10 += Ax[j] * Cx[aij + kk + 2];
      acxI11 += Ax[j] * Cx[aij + kk + 3];
    }
    // copy acx to the output (not needed here)
    ACx[i * N + kk + 0] = acxI0; ACx[i * N + kk + 1] = acxI1; ACx[i * N + kk + 2] = acxI2; ACx[i * N + kk + 3] = acxI3;
    ACx[(i+1) * N + kk + 0] = acxI4; ACx[(i+1) * N + kk + 1] = acxI5; ACx[(i+1) * N + kk + 2] = acxI6; ACx[(i+1) * N + kk + 3] = acxI7;
    ACx[(i+2) * N + kk + 0] = acxI8; ACx[(i+2) * N + kk + 1] = acxI9; ACx[(i+2) * N + kk + 2] = acxI10; ACx[(i+2) * N + kk + 3] = acxI11;

    int jIdx = 0; // loop 2
    for (int j = Bp[jIdx]; j < Bp[jIdx + 1]; j++) {
      int bij = Bi[j] * N;
      for(int k = 0; k < NTile; ++k){
        Dx[jIdx * N + kk + k] += Bx[j] * ACx[bij + kk + k];
      }
    }

    jIdx = 1;
    int inkk = jIdx * N + kk;
    int j = Bp[i];
    //ACx[i * N + kk] += Ax[j] * Cx[aij + kk];
    Dx[inkk + 0] += Bx[j] * acxI0;
    Dx[inkk + 1] += Bx[j] * acxI1;
    Dx[inkk + 2] += Bx[j] * acxI2;
    Dx[inkk + 3] += Bx[j] * acxI3;
    j++;
    Dx[inkk + 0] += Bx[j] * acxI4;
    Dx[inkk + 1] += Bx[j] * acxI5;
    Dx[inkk + 2] += Bx[j] * acxI6;
    Dx[inkk + 3] += Bx[j] * acxI7;
    j++;
    Dx[inkk + 0] += Bx[j] * acxI8;
    Dx[inkk + 1] += Bx[j] * acxI9;
    Dx[inkk + 2] += Bx[j] * acxI10;
    Dx[inkk + 3] += Bx[j] * acxI11;
  }

  // remaining iters
  for (int jj = 3; jj < M; ++jj) {
    for (int kk = 0; kk < N; kk+=NTile) {
      // 2 overlapping iterations
      acxI0 = acxI4; acxI1 = acxI5; acxI2 = acxI6; acxI3 = acxI7;
      acxI4 = acxI8; acxI5 = acxI9; acxI6 = acxI10; acxI7 = acxI11;
      acxI8 = 0; acxI9 = 0; acxI10 = 0; acxI11 = 0;
      // 1 new iteration
      for (int j = Ap[jj]; j < Ap[jj + 1]; j++) {
        int aij = Ai[j] * N;
        // ACx[i * N + kk] += Ax[j] * Cx[aij + kk];
        acxI8 += Ax[j] * Cx[aij + kk];
        acxI9 += Ax[j] * Cx[aij + kk + 1];
        acxI10 += Ax[j] * Cx[aij + kk + 2];
        acxI11 += Ax[j] * Cx[aij + kk + 3];
      }
      // copy acx to the output (not needed here)
      ACx[(jj) * N + kk + 0] = acxI8; ACx[(jj) * N + kk + 1] = acxI9; ACx[(jj) * N + kk + 2] = acxI10; ACx[(jj) * N + kk + 3] = acxI11;

      int j =jj - 1;
      int inkk = j * N + kk;
      Dx[inkk + 0] += Bx[j] * acxI0;
      Dx[inkk + 1] += Bx[j] * acxI1;
      Dx[inkk + 2] += Bx[j] * acxI2;
      Dx[inkk + 3] += Bx[j] * acxI3;
      j++;
      Dx[inkk + 0] += Bx[j] * acxI4;
      Dx[inkk + 1] += Bx[j] * acxI5;
      Dx[inkk + 2] += Bx[j] * acxI6;
      Dx[inkk + 3] += Bx[j] * acxI7;
      j++;
      Dx[inkk + 0] += Bx[j] * acxI8;
      Dx[inkk + 1] += Bx[j] * acxI9;
      Dx[inkk + 2] += Bx[j] * acxI10;
      Dx[inkk + 3] += Bx[j] * acxI11;
    }
  }
  
  pw_stop_instruments_loop(omp_get_thread_num());
}







void spmmCsrSpmmCsrTiledFusedRedundantBanded(int M, int N, int K, int L,
                                             const int *Ap, const int *Ai, const double *Ax,
                                             const int *Bp, const int *Bi,const double *Bx,
                                             const double *Cx,
                                             double *Dx,
                                             double *ACx,
                                             int LevelNo, const int *LevelPtr, const int *ParPtr,
                                             const int *Partition, const int *ParType, const int*MixPtr,
                                             int NThreads, int MTile, int NTile, double *Ws) {
  pw_init_instruments;
  int numKer=2;
  int mBound = M - M % MTile;
  auto *cxBufAll = Ws;//new double[MTile * NTile * NThreads]();
  // First level benefits from Fusion
  int iBoundBeg = LevelPtr[0], iBoundEnd = LevelPtr[1];
#pragma omp parallel num_threads(NThreads)
  {
#pragma omp  for
    for (int j1 = iBoundBeg; j1 < iBoundEnd; ++j1) {
      auto *cxBuf = cxBufAll + omp_get_thread_num() * 2 * MTile * NTile;


      int kBegin = ParPtr[j1], kEnd = MixPtr[j1 * numKer];
      int ii = Partition[kBegin]; // first iteration of tile
      int mTileLoc = kEnd - kBegin;
      //if(ii >= mBound) continue;

      for (int kk = 0; kk < N; kk += NTile) {
        // first loop, for every k-tile
        for (int i = 0; i < mTileLoc; ++i) {
          int iipi = ii + i;
          //            for (int k1 = ParPtr[j1]; k1 < MixPtr[j1 * numKer]; ++k1) {
          //              int i = Partition[k1];
          for (int j = Ap[iipi]; j < Ap[iipi + 1]; ++j) {
            int aij = Ai[j] * N;
            //std::fill_n(cxBuf + i * NTile, NTile, 0.0);
            for (int k = 0; k < NTile; ++k) {
              //auto tmp = Ax[j] * Cx[aij + k];
              //cxBuf[i * NTile + k] = 0;
              cxBuf[i * NTile + k] += Ax[j] * Cx[aij + k];
              //ACx[iipi * N + k + kk] = tmp;
            }
          }
        }
        // print cxBuf
        //            for(int i = 0; i < mTileLoc; ++i) {
        //              for(int k = 0; k < NTile; ++k) {
        //                std::cout << cxBuf[i * NTile + k] << " ";
        //              }
        //              std::cout << std::endl;
        //            }

        // second loop
        int kEndL2 = MixPtr[j1 * numKer + 1];
        for(int k1 = kEnd; k1 < kEndL2; k1++) { // i-loop
          int i = Partition[k1];
          for (int j = Bp[i]; j < Bp[i + 1]; j++) {
            int bij = Bi[j]-ii;
            assert(bij < mTileLoc + 1 && bij >= 0); // stays within the tile i
            bij *= NTile;
            int inkk = i * N + kk;
            for (int k = 0; k < NTile; ++k) {
              Dx[inkk + k] += Bx[j] * cxBuf[bij + k];
              //cxBuf[bij + k] = 0;
            }
          }
        }
        std::fill_n(cxBuf, mTileLoc * NTile, 0.0);

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
}



void spmmCsrSpmmCsrTiledFusedRedundantGeneral(int M, int N, int K, int L,
                                              const int *Ap, const int *Ai, const double *Ax,
                                              const int *Bp, const int *Bi,const double *Bx,
                                              const double *Cx,
                                              double *Dx,
                                              double *ACx,
                                              int LevelNo, const int *LevelPtr, const int *ParPtr,
                                              const int *Partition, const int *ParType, const int*MixPtr,
                                              int NThreads, int MTile, int NTile, double *Ws) {
  pw_init_instruments;
  int numKer=2;
  int mBound = M - M % MTile;
  auto *cxBufAll = Ws;//new double[MTile * NTile * NThreads]();
  // First level benefits from Fusion
  int iBoundBeg = LevelPtr[0], iBoundEnd = LevelPtr[1];
#pragma omp parallel num_threads(NThreads)
  {
#pragma omp  for
    for (int j1 = iBoundBeg; j1 < iBoundEnd; ++j1) {
      auto *cxBuf = cxBufAll + omp_get_thread_num() * 2 * M * NTile;
      int kBegin = ParPtr[j1], kEnd = MixPtr[j1 * numKer];
      int ii = Partition[kBegin]; // first iteration of tile
      int mTileLoc = kEnd - kBegin;
      for (int kk = 0; kk < N; kk += NTile) {
        // first loop, for every k-tile
        for(int k1 = kBegin; k1 < kEnd; k1++) { // i-loop
          int i = Partition[k1];
          // reset cxBuf, I used dot product to avoid the following
          //std::fill_n(cxBuf + i * NTile,  NTile, 0.0);
          for (int k = 0; k < NTile; ++k) {
            double acc = 0;
            for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
              int aij = Ai[j] * N;
              //std::fill_n(cxBuf + i * NTile, NTile, 0.0);

              //auto tmp = Ax[j] * Cx[aij + k];
              //cxBuf[i * NTile + k] = 0;
              acc += Ax[j] * Cx[aij + k];
              //ACx[iipi * N + k + kk] = tmp;
            }
            cxBuf[i * NTile + k] = acc;
          }
        }
        // print cxBuf
        //            for(int i = 0; i < mTileLoc; ++i) {
        //              for(int k = 0; k < NTile; ++k) {
        //                std::cout << cxBuf[i * NTile + k] << " ";
        //              }
        //              std::cout << std::endl;
        //            }

        // second loop
        int kEndL2 = MixPtr[j1 * numKer + 1];
        for(int k1 = kEnd; k1 < kEndL2; k1++) { // i-loop
          int i = Partition[k1];
          for (int j = Bp[i]; j < Bp[i + 1]; j++) {
            int bij = Bi[j];
            bij *= NTile;
            int inkk = i * N + kk;
            for (int k = 0; k < NTile; ++k) {
              Dx[inkk + k] += Bx[j] * cxBuf[bij + k];
              //cxBuf[bij + k] = 0;
            }
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
}


// TODO: this is WIP, we want to tile kk to improve reuse
void spmmCsrSpmmCsrTiledFused(int M, int N, int K, int L,
                              const int *Ap, const int *Ai, const double *Ax,
                              const int *Bp, const int *Bi,const double *Bx,
                              const double *Cx,
                              double *Dx,
                              double *ACx,
                              int LevelNo, const int *LevelPtr, const int *ParPtr,
                              const int *Partition, const int *ParType,
                              int NThreads) {
  pw_init_instruments;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp  for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int aij = Ai[j] * N;
              for (int kk = 0; kk < N; ++kk) {
                ACx[i * N + kk] += Ax[j] * Cx[aij + kk];
              }
            }
          } else {
            for (int k = Bp[i]; k < Bp[i + 1]; k++) {
              int bij = Bi[k] * N;
              for (int kk = 0; kk < N; ++kk) {
                Dx[i * N + kk] += Bx[k] * ACx[bij + kk];
              }
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}

/// D = B*A*C
void spmmCsrSpmmCsrInnerProductFused(int M, int N, int K, int L,
                                     const int *Ap, const int *Ai, const double *Ax,
                                     const int *Bp, const int *Bi,const double *Bx,
                                     const double *Cx,
                                     double *Dx,
                                     double *ACx,
                                     int LevelNo, const int *LevelPtr, const int *ParPtr,
                                     const int *Partition, const int *ParType,
                                     int NThreads) {
  pw_init_instruments;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp  for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            for (int kk = 0; kk < N; ++kk) {
              auto acxik = ACx[i * N + kk];
              for (int j = Ap[i]; j < Ap[i + 1]; j++) {
                int aij = Ai[j] * N;
                acxik += Ax[j] * Cx[aij + kk];
              }
              ACx[i * N + kk] = acxik;
            }
          } else {
            for (int kk = 0; kk < N; ++kk) {
              auto dxik = Dx[i * N + kk];
              for (int k = Bp[i]; k < Bp[i + 1]; k++) {
                int bij = Bi[k] * N;
                dxik += Bx[k] * ACx[bij + kk];
              }
              Dx[i * N + kk] = dxik;
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}


void spmmCsrSpmmCsrMixedScheduleFused(int M, int N, int K, int L,
                                      const int *Ap, const int *Ai, const double *Ax,
                                      const int *Bp, const int *Bi,const double *Bx,
                                      const double *Cx,
                                      double *Dx,
                                      double *ACx,
                                      int LevelNo, const int *LevelPtr, const int *ParPtr,
                                      const int *Partition, const int *ParType,
                                      int NThreads) {
  pw_init_instruments;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp  for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            for (int kk = 0; kk < N; ++kk) {
              auto acxik = ACx[i * N + kk];
              for (int j = Ap[i]; j < Ap[i + 1]; j++) {
                int aij = Ai[j] * N;
                acxik += Ax[j] * Cx[aij + kk];
              }
              ACx[i * N + kk] = acxik; // final AC value after this iteration
            }
          } else {
            for (int k = Bp[i]; k < Bp[i + 1]; k++) {
              int bij = Bi[k] * N;
              for (int kk = 0; kk < N; ++kk) {
                Dx[i * N + kk] += Bx[k] * ACx[bij + kk];
              }
            }
            //                            for (int kk = 0; kk < N; ++kk) {
            //                                auto dxik = Dx[i * N + kk];
            //                                for (int k = Bp[i]; k < Bp[i + 1]; k++) {
            //                                    int bij = Bi[k] * N;
            //                                    dxik += Bx[k] * ACx[bij + kk];
            //                                }
            //                                Dx[i * N + kk] = dxik;
            //                            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}

void spmmCsrSpmmCsrSeparatedFused(int M, int N, int K, int L,
                                  const int *Ap, const int *Ai, const double *Ax,
                                  const int *Bp, const int *Bi,const double *Bx,
                                  const double *Cx,
                                  double *Dx,
                                  double *ACx,
                                  int LevelNo, const int *LevelPtr, const int *ParPtr,
                                  const int *Partition, const int *ParType,
                                  const int *MixPtr,
                                  int NThreads) {
  int numKer = 2;
  pw_init_instruments;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp  for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {

        // Loop 1
        for (int k1 = ParPtr[j1]; k1 < MixPtr[j1 * numKer]; ++k1) {
          int i = Partition[k1];
          for (int kk = 0; kk < N; ++kk) {
            auto acxik = ACx[i * N + kk];
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              int aij = Ai[j] * N;
              acxik += Ax[j] * Cx[aij + kk];
            }
            ACx[i * N + kk] = acxik;
          }
        } // end loop 1

        // Loop 2
        for (int k1 = MixPtr[j1 * numKer ]; k1 < MixPtr[j1 * numKer + 1]; ++k1) {
          int i = Partition[k1];
          for (int kk = 0; kk < N; ++kk) {
            auto dxik = Dx[i * N + kk];
            for (int k = Bp[i]; k < Bp[i + 1]; k++) {
              int bij = Bi[k] * N;
              dxik += Bx[k] * ACx[bij + kk];
            }
            Dx[i * N + kk] = dxik;
          }
        } // end loop 2

      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}


} // namespace sparse
} // namespace swiftware
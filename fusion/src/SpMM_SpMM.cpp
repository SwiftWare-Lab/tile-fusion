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
#include <cstring>
#include <iostream>
#include <omp.h>
#include <immintrin.h>
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


#ifdef __AVX512F__

void vectorCrossProduct128Avx512(double Ax, int Ai, const double* B, const double* C, int N, int I){
  int bij = Ai * N;
  auto bxV = _mm512_set1_pd(Ax);
  int offset = N * I;

  for (int kk = 0; kk < N; kk+=128) {
    auto acxV1 = _mm512_loadu_pd(B + bij + kk);
    auto dxV1 = _mm512_loadu_pd(C + offset + kk);
    auto acxV2 = _mm512_loadu_pd(B + bij + kk + 8);
    auto dxV2 = _mm512_loadu_pd(C + offset + kk + 8);
    auto acxV3 = _mm512_loadu_pd(B + bij + kk + 16);
    auto dxV3 = _mm512_loadu_pd(C + offset + kk + 16);
    auto acxV4 = _mm512_loadu_pd(B + bij + kk + 24);
    auto dxV4 = _mm512_loadu_pd(C + offset + kk + 24);
    auto acxV5 = _mm512_loadu_pd(B + bij + kk + 32);
    auto dxV5 = _mm512_loadu_pd(C + offset + kk+ 32);
    auto acxV6 = _mm512_loadu_pd(B + bij + kk + 40);
    auto dxV6 = _mm512_loadu_pd(C + offset + kk + 40);
    auto acxV7 = _mm512_loadu_pd(B + bij + kk + 48);
    auto dxV7 = _mm512_loadu_pd(C + offset + kk + 48);
    auto acxV8 = _mm512_loadu_pd(B + bij + kk + 56);
    auto dxV8 = _mm512_loadu_pd(C + offset + kk + 56);
    auto acxV9 = _mm512_loadu_pd(B + bij + kk + 64);
    auto dxV9 = _mm512_loadu_pd(C + offset + kk + 64);
    auto acxV10 = _mm512_loadu_pd(B + bij + kk + 72);
    auto dxV10 = _mm512_loadu_pd(C + offset + kk + 72);
    auto acxV11 = _mm512_loadu_pd(B + bij + kk + 80);
    auto dxV11 = _mm512_loadu_pd(C + offset + kk + 80);
    auto acxV12 = _mm512_loadu_pd(B + bij + kk + 88);
    auto dxV12 = _mm512_loadu_pd(C + offset + kk + 88);
    auto acxV13 = _mm512_loadu_pd(B + bij + kk + 96);
    auto dxV13 = _mm512_loadu_pd(C + offset + kk + 96);
    auto acxV14 = _mm512_loadu_pd(B + bij + kk + 104);
    auto dxV14 = _mm512_loadu_pd(C + offset + kk + 104);
    auto acxV15 = _mm512_loadu_pd(B + bij + kk + 112);
    auto dxV15 = _mm512_loadu_pd(C + offset + kk + 112);
    auto acxV16 = _mm512_loadu_pd(B + bij + kk + 120);
    auto dxV16 = _mm512_loadu_pd(C + offset + kk + 120);
    dxV1 = _mm512_fmadd_pd(bxV, acxV1, dxV1);
    dxV2 = _mm512_fmadd_pd(bxV, acxV2, dxV2);
    dxV3 = _mm512_fmadd_pd(bxV, acxV3, dxV3);
    dxV4 = _mm512_fmadd_pd(bxV, acxV4, dxV4);
    dxV5 = _mm512_fmadd_pd(bxV, acxV5, dxV5);
    dxV6 = _mm512_fmadd_pd(bxV, acxV6, dxV6);
    dxV7 = _mm512_fmadd_pd(bxV, acxV7, dxV7);
    dxV8 = _mm512_fmadd_pd(bxV, acxV8, dxV8);
    dxV9 = _mm512_fmadd_pd(bxV, acxV9, dxV9);
    dxV10 = _mm512_fmadd_pd(bxV, acxV10, dxV10);
    dxV11 = _mm512_fmadd_pd(bxV, acxV11, dxV11);
    dxV12 = _mm512_fmadd_pd(bxV, acxV12, dxV12);
    dxV13 = _mm512_fmadd_pd(bxV, acxV13, dxV13);
    dxV14 = _mm512_fmadd_pd(bxV, acxV14, dxV14);
    dxV15 = _mm512_fmadd_pd(bxV, acxV15, dxV15);
    dxV16 = _mm512_fmadd_pd(bxV, acxV16, dxV16);
    _mm512_storeu_pd(C + offset + kk, dxV1);
    _mm512_storeu_pd(C + offset + kk + 8, dxV2);
    _mm512_storeu_pd(C + offset + kk + 16, dxV3);
    _mm512_storeu_pd(C + offset + kk + 24, dxV4);
    _mm512_storeu_pd(C + offset + kk + 32, dxV5);
    _mm512_storeu_pd(C + offset + kk + 40, dxV6);
    _mm512_storeu_pd(C + offset + kk + 48, dxV7);
    _mm512_storeu_pd(C + offset + kk + 56, dxV8);
    _mm512_storeu_pd(C + offset + kk + 64, dxV9);
    _mm512_storeu_pd(C + offset + kk + 72, dxV10);
    _mm512_storeu_pd(C + offset + kk + 80, dxV11);
    _mm512_storeu_pd(C + offset + kk + 88, dxV12);
    _mm512_storeu_pd(C + offset + kk + 96, dxV13);
    _mm512_storeu_pd(C + offset + kk + 104, dxV14);
    _mm512_storeu_pd(C + offset + kk + 112, dxV15);
    _mm512_storeu_pd(C + offset + kk + 120, dxV16);
  }
}

void spmmCsrSpmmCsrFusedVectorized(int M, int N, int K, int L,
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
              vectorCrossProduct128Avx512(Ax[j], Ai[j], Cx, ACx, N, i );
              }
            }
          } else {
            for (int k = Bp[i]; k < Bp[i + 1]; k++) {
              vectorCrossProduct128Avx512(Bx[k], Bi[k], ACX, Dx, N, i );
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}
#elif __AVX2__

void spmmCsrSpmmCsrFusedVectorized(int M, int N, int K, int L,
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
              auto axV = _mm256_set1_pd(Ax[j]);
              for (int kk = 0; kk < N; kk+=4) {
                auto cxV = _mm256_loadu_pd(Cx + aij + kk);
                auto acxV = _mm256_loadu_pd(ACx + i*N + kk);
                acxV = _mm256_fmadd_pd(axV, cxV, acxV);
                _mm256_storeu_pd(ACx + i*N + kk, acxV);
              }
            }
          } else {
            for (int k = Bp[i]; k < Bp[i + 1]; k++) {
              int bij = Bi[k] * N;
              auto bxV = _mm256_set1_pd(Bx[k]);
              for (int kk = 0; kk < N; kk+=4) {
                auto acxV = _mm256_loadu_pd(ACx + bij + kk);
                auto dxV = _mm256_loadu_pd(Dx + i*N + kk);
                dxV = _mm256_fmadd_pd(bxV, acxV, dxV);
                _mm256_storeu_pd(Dx + i*N + kk, dxV);
              }
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}

#endif

/// D = B*A*C
void spmmCsrSpmmCsrFusedKTiled(int M, int N, int K, int L,
                         const int *Ap, const int *Ai, const double *Ax,
                         const int *Bp, const int *Bi,const double *Bx,
                         const double *Cx,
                         double *Dx,
                         double *ACx, int KTileSize,
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
            for (int k = 0; k < N; k += KTileSize) {
              for (int j = Ap[i]; j < Ap[i + 1]; j++) {
                int aij = Ai[j] * N;
                for (int kk = 0; kk < KTileSize; ++kk) {
                  ACx[i * N + kk + k] += Ax[j] * Cx[aij + kk + k];
                }
              }
            }
          } else {
            for (int k = 0; k < N; k += KTileSize) {
              for (int j = Bp[i]; j < Bp[i + 1]; j++) {
                int bij = Bi[j] * N;
                for (int kk = 0; kk < KTileSize; ++kk) {
                  Dx[i * N + kk + k] += Bx[j] * ACx[bij + kk + k];
                }
              }
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}

void spmmCsrSpmmCscFused(int M, int N, int K, int L,
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
            for (int k = Bp[i]; k < Bp[i + 1]; k++) { // for each column of B

              for (int kk = 0; kk < N; ++kk) {
                int bij = Bi[k] * N;
#pragma omp atomic
                Dx[bij + kk] += Bx[k] * ACx[i*N + kk];
              }
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}


void spmmCsrSpmmCscFusedAffine(int M, int N, int K, int L,
                               const int *Ap, const int *Ai, const double *Ax,
                               const int *Bp, const int *Bi,const double *Bx,
                               const double *Cx,
                               double *Dx,
                               double *ACx,
                               int NThreads) {
#pragma omp parallel  for num_threads(NThreads)
    for (int i = 0; i < M; ++i) {
      for (int j = Ap[i]; j < Ap[i + 1]; j++) {
        int aij = Ai[j] * N;
        for (int kk = 0; kk < N; ++kk) {
          ACx[i * N + kk] += Ax[j] * Cx[aij + kk];
        }
      }
      for (int k = Bp[i]; k < Bp[i + 1]; k++) { // for each column of B
        for (int kk = 0; kk < N; ++kk) {
          int bij = Bi[k] * N;
          auto tmp = Bx[k] * ACx[i*N + kk];
#pragma omp atomic
          Dx[bij + kk] += tmp;
        }
      }
    }
}


void spmmCsrSpmmCscFusedColored(int M, int N, int K, int L,
                                const int *Ap, const int *Ai, const double *Ax,
                                const int *Bp, const int *Bi,const double *Bx,
                                const double *Cx,
                                double *Dx,
                                double *ACx,
                                int LevelNo, const int *LevelPtr,
                                const int *Id,
                                int TileSize, int NThreads) {
    int lastTileSize = M% TileSize;
    double* aCxi = new double[NThreads * TileSize * N];
    pw_init_instruments;
    for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
      {
        pw_start_instruments_loop(omp_get_thread_num());
        int threadId = omp_get_thread_num();
#pragma omp for
        for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
          int id = Id[j1];
          int i = id * TileSize;
          memset(aCxi + threadId*TileSize*N, 0.,sizeof (double) *N*TileSize);
          for (int ii = 0; ii < TileSize; ++ii) {
          auto ipii = i + ii;
          double* tAcxi = aCxi + threadId*N*TileSize + ii*N;
          // first SpMM
          for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
            int aij = Ai[j] * N;
            for (int kk = 0; kk < N; ++kk) {
              tAcxi[kk] += Ax[j] * Cx[aij + kk];
            }
          }
          // second SpMM CSC
          for (int k = Bp[ipii]; k < Bp[ipii + 1];
               k++) { // for each column of B
            for (int kk = 0; kk < N; ++kk) {
              int bij = Bi[k] * N;
              Dx[bij + kk] += Bx[k] * tAcxi[kk];
            }
          }
          }
        }
        pw_stop_instruments_loop(omp_get_thread_num());
      }
    }
    delete[] aCxi;
    int i = M-lastTileSize;
    for (int ii = 0; ii < lastTileSize; ++ii) {
      auto ipii = i + ii;
      // first SpMM
      for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
        int aij = Ai[j] * N;
        for (int kk = 0; kk < N; ++kk) {
          ACx[ipii * N + kk] += Ax[j] * Cx[aij + kk];
        }
      }
      // second SpMM CSC
      for (int k = Bp[ipii]; k < Bp[ipii + 1];
           k++) { // for each column of B
        for (int kk = 0; kk < N; ++kk) {
          int bij = Bi[k] * N;
          Dx[bij + kk] += Bx[k] * ACx[ipii * N + kk];
        }
      }
    }

}


void spmmCsrSpmmCscFusedColoredWithScheduledKTiles(int M, int N, int K, int L,
                                const int *Ap, const int *Ai, const double *Ax,
                                const int *Bp, const int *Bi,const double *Bx,
                                const double *Cx,
                                double *Dx,
                                double *ACx,
                                int LevelNo, const int *LevelPtr,
                                const int *Id,
                                int TileSize, int KTileSize,
                                int NThreads) {
    int numOfKTiles = N / KTileSize;
    int lastTileSize = M % TileSize;
    pw_init_instruments;
    for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
      {
        pw_start_instruments_loop(omp_get_thread_num());
#pragma omp  for
        for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
          int id = Id[j1];
          int tile = (id/numOfKTiles);
          int i = tile * TileSize;
          int k = (id % numOfKTiles) * KTileSize;
          for(int ii = 0; ii <  TileSize; ++ii) {
            auto ipii = i + ii;
            // first SpMM
            for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
              int aij = Ai[j] * N;
              for (int kk = 0; kk < KTileSize; ++kk) {
                ACx[ipii * N + kk + k] += Ax[j] * Cx[aij + kk + k];
              }
            }
            // second SpMM CSC
            for (int j = Bp[ipii]; j < Bp[ipii + 1]; j++) { // for each column of B
              for (int kk = 0; kk < KTileSize; ++kk) {
                int bij = Bi[j] * N;
                Dx[bij + kk + k] += Bx[j] * ACx[ipii * N + kk + k];
              }
            }
          }
        }
        pw_stop_instruments_loop(omp_get_thread_num());
      }
    }
    int i = M-lastTileSize;
    for (int ii = 0; ii < lastTileSize; ++ii) {
      auto ipii = i + ii;
      // first SpMM
      for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
        int aij = Ai[j] * N;
        for (int kk = 0; kk < N; ++kk) {
          ACx[ipii * N + kk] += Ax[j] * Cx[aij + kk];
        }
      }
      // second SpMM CSC
      for (int k = Bp[ipii]; k < Bp[ipii + 1];
           k++) { // for each column of B
        for (int kk = 0; kk < N; ++kk) {
          int bij = Bi[k] * N;
          Dx[bij + kk] += Bx[k] * ACx[ipii * N + kk];
        }
      }
    }
}

void spmmCsrSpmmCscFusedColoredWithReplicatedKTiles(int M, int N, int K, int L,
                                                        const int *Ap, const int *Ai, const double *Ax,
                                                        const int *Bp, const int *Bi,const double *Bx,
                                                        const double *Cx,
                                                        double *Dx,
                                                        double *ACx,
                                                        int LevelNo, const int *LevelPtr,
                                                        const int *Id, const int *TileSizes,
                                                        int TileSize, int KTileSize,
                                                        int NThreads) {
    int numOfKTiles = N / KTileSize;
    int lastTileSize = M% TileSize;
    pw_init_instruments;
    for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
      {
        pw_start_instruments_loop(omp_get_thread_num());
#pragma omp  for
        for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
          int id = Id[j1];
          int i = id * TileSize;
          int k = (j1 % numOfKTiles) * KTileSize;
          for(int ii = 0; ii <  TileSize; ++ii) {
            auto ipii = i + ii;
            // first SpMM
            for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
              int aij = Ai[j] * N;
              for (int kk = 0; kk < KTileSize; ++kk) {
                ACx[ipii * N + kk + k] += Ax[j] * Cx[aij + kk + k];
              }
            }
            // second SpMM CSC
            for (int j = Bp[ipii]; j < Bp[ipii + 1]; j++) { // for each column of B
              for (int kk = 0; kk < KTileSize; ++kk) {
                int bij = Bi[j] * N;
                Dx[bij + kk + k] += Bx[j] * ACx[ipii * N + kk + k];
              }
            }
          }
        }
        pw_stop_instruments_loop(omp_get_thread_num());
      }
    }
    int i = M-lastTileSize;
    for (int ii = 0; ii < lastTileSize; ++ii) {
      auto ipii = i + ii;
      // first SpMM
      for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
        int aij = Ai[j] * N;
        for (int kk = 0; kk < N; ++kk) {
          ACx[ipii * N + kk] += Ax[j] * Cx[aij + kk];
        }
      }
      // second SpMM CSC
      for (int k = Bp[ipii]; k < Bp[ipii + 1];
           k++) { // for each column of B
        for (int kk = 0; kk < N; ++kk) {
          int bij = Bi[k] * N;
          Dx[bij + kk] += Bx[k] * ACx[ipii * N + kk];
        }
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
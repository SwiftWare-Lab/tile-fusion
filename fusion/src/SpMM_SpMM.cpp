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
#include <immintrin.h>

namespace swiftware {
namespace sparse {

/// C = A*B, where A is sparse CSR MxK and B (K x N) and C (MxN) are Dense
void spmmCsrSequential(int M, int N, int K, const int *Ap, const int *Ai,
                       const double *Ax, const double *Bx, double *Cx) {
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

void spmmCsrAvxFirstSparseRow(int M, int N, int K, const int *Ap,
                                 const int *Ai, const double *Ax,
                                 const double *Bx, double *Cx, int NThreads) {
#pragma omp parallel num_threads(NThreads)
  {
#pragma omp for
    for (int i = 0; i < M; ++i) {
      int row_wdth_A = Ap[i + 1] - Ap[i];
      int last_v_size = row_wdth_A % 4;
      int last_v_start = Ap[i + 1] - last_v_size;
      for (int k = 0; k < N; k++) {
        __m256d d_vec_C = _mm256_setzero_pd();
        for (int j = Ap[i]; j < last_v_start; j += 4) {
          __m256d d_vec_A = _mm256_loadu_pd(Ax + j);
          __m256d d_vec_B =
              _mm256_set_pd(Bx[Ai[j + 3] * N + k], Bx[Ai[j + 2] * N + k],
                            Bx[Ai[j + 1] * N + k], Bx[Ai[j] * N + k]);
          d_vec_C = _mm256_fmadd_pd(d_vec_A, d_vec_B, d_vec_C);
        }
        __m256d d_vec_Z = _mm256_setzero_pd();
        d_vec_C = _mm256_hadd_pd(d_vec_C, d_vec_Z);
        d_vec_C = _mm256_permute4x64_pd(d_vec_C, 0b01011000);
        __m256d result = _mm256_hadd_pd(d_vec_C, d_vec_Z);
        Cx[i * N + k] = result[0];
      }
      for (int j = last_v_start; j < Ap[i + 1]; j++) {
        int aij = Ai[j] * N;
        for (int k = 0; k < N; ++k) {
          assert(i * N + k < M * N);
          Cx[i * N + k] += Ax[j] * Bx[aij + k];
        }
      }
    }
  }
}

void spmmCsrAvxFirstDenseRow(int M, int N, int K, const int *Ap,
                                const int *Ai, const double *Ax,
                                const double *Bx, double *Cx, int NThreads) {
#pragma omp parallel num_threads(NThreads)
  {
#pragma omp for
    for (int i = 0; i < M; ++i) {
      int last_v_size = N % 4;
      int last_v_start = N - last_v_size;

      for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
        __m256d d_vec_A = _mm256_set1_pd(Ax[j]);
        int aij = Ai[j] * N;
        for (int k = 0; k < last_v_start; k += 4) {
          __m256d d_vec_C = _mm256_loadu_pd(Cx + i * N + k);
          __m256d d_vec_B = _mm256_loadu_pd(Bx + aij + k);
          d_vec_C = _mm256_fmadd_pd(d_vec_A, d_vec_B, d_vec_C);
          _mm256_storeu_pd(Cx + i * N + k, d_vec_C);
        }
      }
      for (int k = last_v_start; k < N; ++k) {
        for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
          int aij = Ai[j] * N;
          assert(i * N + k < M * N);
          Cx[i * N + k] += Ax[j] * Bx[aij + k];
        }
      }
    }
  }
}

void spmmCsrAvxFirstDenseRowSecondSparseRow(int M, int N, int K, const int *Ap,
                                   const int *Ai, const double *Ax,
                                   const double *Bx, double *Cx, int NThreads) {
#pragma omp parallel num_threads(NThreads)
  {
#pragma omp for
    for (int i = 0; i < M; ++i) {
      int last_B_v_size = N % 4;
      int last_B_v_start = N - last_B_v_size;

      for (int k = 0; k < last_B_v_start; k += 4) {
        __m256d d_vec_C = _mm256_setzero_pd();
        for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
          int aij = Ai[j] * N;
          __m256d d_vec_B = _mm256_loadu_pd(Bx + aij + k);
          __m256d d_vec_A = _mm256_set1_pd(Ax[j]);
          d_vec_C = _mm256_fmadd_pd(d_vec_A, d_vec_B, d_vec_C);
        }
        for (int j = 0; j < 4; j++) {
          Cx[i * N + k + j] = d_vec_C[j];
        }
      }

      int row_wdth_A = Ap[i + 1] - Ap[i];
      int last_A_v_size = row_wdth_A % 4;
      int last_A_v_start = Ap[i + 1] - last_A_v_size;
      for (int k = last_B_v_start; k < N; k++) {
        __m256d d_vec_C = _mm256_setzero_pd();
        for (int j = Ap[i]; j < last_A_v_start; j += 4) {
          __m256d d_vec_A = _mm256_loadu_pd(Ax + j);
          int aij1 = Ai[j] * N;
          int aij2 = Ai[j + 1] * N;
          int aij3 = Ai[j + 2] * N;
          int aij4 = Ai[j + 3] * N;
          __m256d d_vec_B = _mm256_set_pd(Bx[aij4 + k], Bx[aij3 + k],
                                          Bx[aij2 + k], Bx[aij1 + k]);
          d_vec_C = _mm256_fmadd_pd(d_vec_A, d_vec_B, d_vec_C);
        }
        __m256d d_vec_Z = _mm256_setzero_pd();
        d_vec_C = _mm256_hadd_pd(d_vec_C, d_vec_Z);
        d_vec_C = _mm256_permute4x64_pd(d_vec_C, 0b01011000);
        __m256d result = _mm256_hadd_pd(d_vec_C, d_vec_Z);
        Cx[i * N + k] += result[0];
      }
      for (int k = last_B_v_start; k < N; ++k) {
        for (int j = last_A_v_start; j < Ap[i + 1]; ++j) {
          int aij = Ai[j] * N;
          assert(i * N + k < M * N);
          Cx[i * N + k] += Ax[j] * Bx[aij + k];
        }
      }
    }
  }
}

void spmmCsrParallel(int M, int N, int K, const int *Ap, const int *Ai,
                     const double *Ax, const double *Bx, double *Cx,
                     int NThreads) {
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

void spmmCsrInnerProductParallel(int M, int N, int K, const int *Ap,
                                 const int *Ai, const double *Ax,
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
          cik += Ax[j] * Bx[aij + k]; // C[i][k] += A[i][j] * B[j][k];
        }
        Cx[i * N + k] = cik;
      }
    }
    pw_stop_instruments_loop(omp_get_thread_num());
  }
}

void spmmCsrInnerProductTiledCParallel(int M, int N, int K, const int *Ap,
                                       const int *Ai, const double *Ax,
                                       const double *Bx, double *Cx,
                                       int NThreads, int MTile, int NTile) {
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
            auto cik = 0; // cTile[ii * NTile + kk];
            for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; ++j) {
              int aij = Ai[j] * N;
              cik += Ax[j] * Bx[aij + k + kk]; // C[i][k] += A[i][j] * B[j][k];
            }
            Cx[(i + ii) * N + (k + kk)] += cik;
            // cTile[ii * NTile + kk] += cik;
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
          auto cik = 0; // Cx[(i+ii) * N + k];
          for (int j = Ap[i + ii]; j < Ap[i + ii + 1]; ++j) {
            int aij = Ai[j] * N;
            cik += Ax[j] * Bx[aij + k]; // C[i][k] += A[i][j] * B[j][k];
          }
          Cx[(i + ii) * N + k] += cik;
        }
      }
    }
    pw_stop_instruments_loop(omp_get_thread_num());
  }
  // tail iterations for i
#pragma omp parallel for num_threads(NThreads)
  for (int i = mTailBeg; i < M; i++) {
    for (int k = 0; k < N; ++k) {
      auto cik = Cx[i * N + k];
      for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
        int aij = Ai[j] * N;
        cik += Ax[j] * Bx[aij + k]; // C[i][k] += A[i][j] * B[j][k];
      }
      Cx[i * N + k] += cik;
    }
  }
}

/// D = B*A*C
void spmmCsrSpmmCsrFused(int M, int N, int K, int L, const int *Ap,
                         const int *Ai, const double *Ax, const int *Bp,
                         const int *Bi, const double *Bx, const double *Cx,
                         double *Dx, double *ACx, int LevelNo,
                         const int *LevelPtr, const int *ParPtr,
                         const int *Partition, const int *ParType,
                         int NThreads) {
  pw_init_instruments;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
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

// TODO: this is WIP, we want to tile kk to improve reuse
void spmmCsrSpmmCsrTiledFused(int M, int N, int K, int L, const int *Ap,
                              const int *Ai, const double *Ax, const int *Bp,
                              const int *Bi, const double *Bx, const double *Cx,
                              double *Dx, double *ACx, int LevelNo,
                              const int *LevelPtr, const int *ParPtr,
                              const int *Partition, const int *ParType,
                              int NThreads) {
  pw_init_instruments;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
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
void spmmCsrSpmmCsrInnerProductFused(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads) {
  pw_init_instruments;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
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

void spmmCsrSpmmCsrSeparatedFused(int M, int N, int K, int L, const int *Ap,
                                  const int *Ai, const double *Ax,
                                  const int *Bp, const int *Bi,
                                  const double *Bx, const double *Cx,
                                  double *Dx, double *ACx, int LevelNo,
                                  const int *LevelPtr, const int *ParPtr,
                                  const int *Partition, const int *ParType,
                                  const int *MixPtr, int NThreads) {
  int numKer = 2;
  pw_init_instruments;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
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
        for (int k1 = MixPtr[j1 * numKer]; k1 < MixPtr[j1 * numKer + 1]; ++k1) {
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
//
// Created by salehm32 on 21/02/24.
//
#ifdef PROF_WITH_PAPI
#include "papi_wrapper.h"
#else
#define pw_init_instruments
#define pw_start_instruments_loop(th)
#define pw_stop_instruments_loop(th)
#endif
#include <immintrin.h>
#include <omp.h>
namespace swiftware {
namespace sparse {
#ifdef __AVX512F__

inline void vectorCrossProduct8Avx512(double Ax, int Ai, const double *B,
                                      __m512d &Xv, int N, int I) {
  int bij = Ai * N;
  auto bxV = _mm512_set1_pd(Ax);
  auto acxV1 = _mm512_loadu_pd(B + bij);
  Xv = _mm512_fmadd_pd(bxV, acxV1, Xv);
}

inline void vectorCrossProduct4_8Avx512(const double* Ax, const int* Ai, const double *B,
                                        __m512d &Xv, int N, int I) {
  int bij0 = Ai[0] * N;
  int bij1 = Ai[1] * N;
  int bij2 = Ai[2] * N;
  int bij3 = Ai[3] * N;
  auto bxV0 = _mm512_set1_pd(Ax[0]);
  auto bxV1 = _mm512_set1_pd(Ax[1]);
  auto bxV2 = _mm512_set1_pd(Ax[2]);
  auto bxV3 = _mm512_set1_pd(Ax[3]);
  auto acxV0 = _mm512_loadu_pd(B + bij0);
  auto acxV1 = _mm512_loadu_pd(B + bij1);
  auto acxV2 = _mm512_loadu_pd(B + bij2);
  auto acxV3 = _mm512_loadu_pd(B + bij3);
  Xv = _mm512_fmadd_pd(bxV0, acxV0, Xv);
  Xv = _mm512_fmadd_pd(bxV1, acxV1, Xv);
  Xv = _mm512_fmadd_pd(bxV2, acxV2, Xv);
  Xv = _mm512_fmadd_pd(bxV3, acxV3, Xv);
}

inline void vectorCrossProduct64Avx512(double Ax, int Ai, const double *B,
                                       double *C, int N, int I) {
  int bij = Ai * N;
  auto bxV = _mm512_set1_pd(Ax);
  int offset = N * I;

  for (int kk = 0; kk < N; kk += 64) {
    auto acxV1 = _mm512_loadu_pd(B + bij + kk);
    auto dxV1 = _mm512_loadu_pd(C + offset + kk);
    auto acxV2 = _mm512_loadu_pd(B + bij + kk + 8);
    auto dxV2 = _mm512_loadu_pd(C + offset + kk + 8);
    auto acxV3 = _mm512_loadu_pd(B + bij + kk + 16);
    auto dxV3 = _mm512_loadu_pd(C + offset + kk + 16);
    auto acxV4 = _mm512_loadu_pd(B + bij + kk + 24);
    auto dxV4 = _mm512_loadu_pd(C + offset + kk + 24);
    auto acxV5 = _mm512_loadu_pd(B + bij + kk + 32);
    auto dxV5 = _mm512_loadu_pd(C + offset + kk + 32);
    auto acxV6 = _mm512_loadu_pd(B + bij + kk + 40);
    auto dxV6 = _mm512_loadu_pd(C + offset + kk + 40);
    auto acxV7 = _mm512_loadu_pd(B + bij + kk + 48);
    auto dxV7 = _mm512_loadu_pd(C + offset + kk + 48);
    auto acxV8 = _mm512_loadu_pd(B + bij + kk + 56);
    auto dxV8 = _mm512_loadu_pd(C + offset + kk + 56);
    dxV1 = _mm512_fmadd_pd(bxV, acxV1, dxV1);
    dxV2 = _mm512_fmadd_pd(bxV, acxV2, dxV2);
    dxV3 = _mm512_fmadd_pd(bxV, acxV3, dxV3);
    dxV4 = _mm512_fmadd_pd(bxV, acxV4, dxV4);
    dxV5 = _mm512_fmadd_pd(bxV, acxV5, dxV5);
    dxV6 = _mm512_fmadd_pd(bxV, acxV6, dxV6);
    dxV7 = _mm512_fmadd_pd(bxV, acxV7, dxV7);
    dxV8 = _mm512_fmadd_pd(bxV, acxV8, dxV8);
    _mm512_storeu_pd(C + offset + kk, dxV1);
    _mm512_storeu_pd(C + offset + kk + 8, dxV2);
    _mm512_storeu_pd(C + offset + kk + 16, dxV3);
    _mm512_storeu_pd(C + offset + kk + 24, dxV4);
    _mm512_storeu_pd(C + offset + kk + 32, dxV5);
    _mm512_storeu_pd(C + offset + kk + 40, dxV6);
    _mm512_storeu_pd(C + offset + kk + 48, dxV7);
    _mm512_storeu_pd(C + offset + kk + 56, dxV8);
  }
}

inline void vectorCrossProduct32Avx512(double Ax, int Ai,
                                       const double *B, double *C, int N,
                                       __m512d &dxV1,  __m512d &dxV2,
                                       __m512d &dxV3,  __m512d &dxV4) {
  int bij = Ai * N;
  auto bxV = _mm512_set1_pd(Ax);
  auto acxV1 = _mm512_loadu_pd(B + bij);
  auto acxV2 = _mm512_loadu_pd(B + bij + 8);
  auto acxV3 = _mm512_loadu_pd(B + bij + 16);
  auto acxV4 = _mm512_loadu_pd(B + bij + 24);
  dxV1 = _mm512_fmadd_pd(bxV, acxV1, dxV1);
  dxV2 = _mm512_fmadd_pd(bxV, acxV2, dxV2);
  dxV3 = _mm512_fmadd_pd(bxV, acxV3, dxV3);
  dxV4 = _mm512_fmadd_pd(bxV, acxV4, dxV4);
}

inline void vectorCrossProduct32Avx512SP(float Ax, int Ai,
                                       const float *B, float *C, int N,
                                       __m512 &dxV1,  __m512 &dxV2) {
  int bij = Ai * N;
  auto bxV = _mm512_set1_ps(Ax);
  auto acxV1 = _mm512_loadu_ps(B + bij);
  auto acxV2 = _mm512_loadu_ps(B + bij + 16);
  dxV1 = _mm512_fmadd_ps(bxV, acxV1, dxV1);
  dxV2 = _mm512_fmadd_ps(bxV, acxV2, dxV2);
}

inline void vectorCrossProduct2_32Avx512(const double* Ax, const int* Ai,
                                         const double *B,double *C, int N,
                                         __m512d &dxV1,  __m512d &dxV2,
                                         __m512d &dxV3,  __m512d &dxV4) {
  int bij0 = Ai[0] * N;
  int bij1 = Ai[1] * N;
  auto bxV0 = _mm512_set1_pd(Ax[0]);
  auto bxV1 = _mm512_set1_pd(Ax[1]);
  auto acxV11 = _mm512_loadu_pd(B + bij0);
  auto acxV12 = _mm512_loadu_pd(B + bij0 + 8);
  auto acxV13 = _mm512_loadu_pd(B + bij0 + 16);
  auto acxV14 = _mm512_loadu_pd(B + bij0 + 24);
  auto acxV21 = _mm512_loadu_pd(B + bij1);
  auto acxV22 = _mm512_loadu_pd(B + bij1 + 8);
  auto acxV23 = _mm512_loadu_pd(B + bij1 + 16);
  auto acxV24 = _mm512_loadu_pd(B + bij1 + 24);
  dxV1 = _mm512_fmadd_pd(bxV0, acxV11, dxV1);
  dxV1 = _mm512_fmadd_pd(bxV1, acxV21, dxV1);
  dxV2 = _mm512_fmadd_pd(bxV0, acxV12, dxV2);
  dxV2 = _mm512_fmadd_pd(bxV1, acxV22, dxV2);
  dxV3 = _mm512_fmadd_pd(bxV0, acxV13, dxV3);
  dxV3 = _mm512_fmadd_pd(bxV1, acxV23, dxV3);
  dxV4 = _mm512_fmadd_pd(bxV0, acxV14, dxV4);
  dxV4 = _mm512_fmadd_pd(bxV1, acxV24, dxV4);
}

inline void vectorCrossProduct2_32Avx512SP(const float* Ax, const int* Ai,
                                         const float *B,float *C, int N,
                                         __m512 &dxV1,  __m512 &dxV2) {
  int bij0 = Ai[0] * N;
  int bij1 = Ai[1] * N;
  auto bxV0 = _mm512_set1_ps(Ax[0]);
  auto bxV1 = _mm512_set1_ps(Ax[1]);
  auto acxV11 = _mm512_loadu_ps(B + bij0);
  auto acxV12 = _mm512_loadu_ps(B + bij0 + 16);
  auto acxV21 = _mm512_loadu_ps(B + bij1);
  auto acxV22 = _mm512_loadu_ps(B + bij1 + 16);
  dxV1 = _mm512_fmadd_ps(bxV0, acxV11, dxV1);
  dxV1 = _mm512_fmadd_ps(bxV1, acxV21, dxV1);
  dxV2 = _mm512_fmadd_ps(bxV0, acxV12, dxV2);
  dxV2 = _mm512_fmadd_ps(bxV1, acxV22, dxV2);
}

void spmmCsrSpmmCsrFusedVectorized2_32Avx512(
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
          int offset = N * i;
          if (t == 0) {
            for (int kk = 0; kk < N; kk += 32) {
              auto axV1 = _mm512_loadu_pd(ACx + offset + kk);
              auto axV2 = _mm512_loadu_pd(ACx + offset + kk + 8);
              auto axV3 = _mm512_loadu_pd(ACx + offset + kk + 16);
              auto axV4 = _mm512_loadu_pd(ACx + offset + kk + 24);
              int j = Ap[i];
              for (; j < Ap[i + 1] - 1; j += 2) {
                vectorCrossProduct2_32Avx512(Ax + j, Ai + j, Cx + kk, ACx + kk, N,
                                             axV1, axV2, axV3, axV4);
              }
              for (; j < Ap[i + 1]; j++) {
                vectorCrossProduct32Avx512(Ax[j], Ai[j], Cx + kk, ACx + kk, N,
                                           axV1, axV2, axV3, axV4);
              }
              _mm512_storeu_pd(ACx + offset + kk, axV1);
              _mm512_storeu_pd(ACx + offset + kk + 8, axV2);
              _mm512_storeu_pd(ACx + offset + kk + 16, axV3);
              _mm512_storeu_pd(ACx + offset + kk + 24, axV4);
            }
          } else {
            for (int kk = 0; kk < N; kk += 32) {
              auto dxV1 = _mm512_loadu_pd(Dx + offset + kk);
              auto dxV2 = _mm512_loadu_pd(Dx + offset + kk + 8);
              auto dxV3 = _mm512_loadu_pd(Dx + offset + kk + 16);
              auto dxV4 = _mm512_loadu_pd(Dx + offset + kk + 24);
              int k = Bp[i];
              for (; k < Bp[i + 1]-1; k+=2) {
                vectorCrossProduct2_32Avx512(Bx + k, Bi + k, ACx + kk, Dx + kk, N,
                                             dxV1, dxV2, dxV3, dxV4);
              }
              for (;k < Bp[i + 1]; k++){
                vectorCrossProduct32Avx512(Bx[k], Bi[k], ACx + kk, Dx + kk, N,
                                           dxV1, dxV2, dxV3, dxV4);
              }
              _mm512_storeu_pd(Dx + offset + kk, dxV1);
              _mm512_storeu_pd(Dx + offset + kk + 8, dxV2);
              _mm512_storeu_pd(Dx + offset + kk + 16, dxV3);
              _mm512_storeu_pd(Dx + offset + kk + 24, dxV4);
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}

void spmmCsrSpmmCsrFusedVectorized2_32Avx512SP(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const float *Ax,
    const int *Bp, const int *Bi, const float *Bx, const float *Cx,
    float *Dx, float *ACx, int LevelNo, const int *LevelPtr,
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
          int offset = N * i;
          if (t == 0) {
            for (int kk = 0; kk < N; kk += 32) {
              auto axV1 = _mm512_loadu_ps(ACx + offset + kk);
              auto axV2 = _mm512_loadu_ps(ACx + offset + kk + 16);
              int j = Ap[i];
              for (; j < Ap[i + 1] - 1; j += 2) {
                vectorCrossProduct2_32Avx512SP(Ax + j, Ai + j, Cx + kk, ACx + kk, N,
                                             axV1, axV2);
              }
              for (; j < Ap[i + 1]; j++) {
                vectorCrossProduct32Avx512SP(Ax[j], Ai[j], Cx + kk, ACx + kk, N,
                                           axV1, axV2);
              }
              _mm512_storeu_ps(ACx + offset + kk, axV1);
              _mm512_storeu_ps(ACx + offset + kk + 16, axV2);
            }
          } else {
            for (int kk = 0; kk < N; kk += 32) {
              auto dxV1 = _mm512_loadu_ps(Dx + offset + kk);
              auto dxV2 = _mm512_loadu_ps(Dx + offset + kk + 16);
              int k = Bp[i];
              for (; k < Bp[i + 1]-1; k+=2) {
                vectorCrossProduct2_32Avx512SP(Bx + k, Bi + k, ACx + kk, Dx + kk, N,
                                             dxV1, dxV2);
              }
              for (;k < Bp[i + 1]; k++){
                vectorCrossProduct32Avx512SP(Bx[k], Bi[k], ACx + kk, Dx + kk, N,
                                           dxV1, dxV2);
              }
              _mm512_storeu_ps(Dx + offset + kk, dxV1);
              _mm512_storeu_ps(Dx + offset + kk + 16, dxV2);
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}

void spmmCsrVectorized2_32Avx512(int M, int N, const int *Ap, const int *Ai,
                                 const double *Ax, const double *Cx,
                                 double *ACx, int TileSize, int NThreads) {
  pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
  {
    pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
    for (int ii = 0; ii < M; ii += TileSize) {
      for (int i = ii; i < ii + TileSize && i < M; i++) {
        int offset = N * i;
        for (int kk = 0; kk < N; kk += 32) {
          auto axV1 = _mm512_loadu_pd(ACx + offset + kk);
          auto axV2 = _mm512_loadu_pd(ACx + offset + kk + 8);
          auto axV3 = _mm512_loadu_pd(ACx + offset + kk + 16);
          auto axV4 = _mm512_loadu_pd(ACx + offset + kk + 24);
          int j = Ap[i];
          for (; j < Ap[i + 1] - 1; j += 2) {
            vectorCrossProduct2_32Avx512(Ax + j, Ai + j, Cx + kk, ACx + kk, N,
                                         axV1, axV2, axV3, axV4);
          }
          for (; j < Ap[i + 1]; j++) {
            vectorCrossProduct32Avx512(Ax[j], Ai[j], Cx + kk, ACx + kk, N,
                                       axV1, axV2, axV3, axV4);
          }
          _mm512_storeu_pd(ACx + offset + kk, axV1);
          _mm512_storeu_pd(ACx + offset + kk + 8, axV2);
          _mm512_storeu_pd(ACx + offset + kk + 16, axV3);
          _mm512_storeu_pd(ACx + offset + kk + 24, axV4);
        }
      }
    }
    pw_stop_instruments_loop(omp_get_thread_num());
  }
  pw_init_instruments;
}

void spmmCsrVectorized2_32Avx512SP(int M, int N, const int *Ap, const int *Ai,
                                   const float *Ax, const float *Cx,
                                   float *ACx, int TileSize, int NThreads){
  pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
  {
    pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
    for (int ii = 0; ii < M; ii += TileSize) {
      for (int i = ii; i < ii + TileSize && i < M; i++) {
        int offset = N * i;
        for (int kk = 0; kk < N; kk += 32) {
          auto axV1 = _mm512_loadu_ps(ACx + offset + kk);
          auto axV2 = _mm512_loadu_ps(ACx + offset + kk + 16);
          int j = Ap[i];
          for (; j < Ap[i + 1] - 1; j += 2) {
            vectorCrossProduct2_32Avx512SP(Ax + j, Ai + j, Cx + kk, ACx + kk, N,
                                           axV1, axV2);
          }
          for (; j < Ap[i + 1]; j++) {
            vectorCrossProduct32Avx512SP(Ax[j], Ai[j], Cx + kk, ACx + kk, N,
                                         axV1, axV2);
          }
          _mm512_storeu_ps(ACx + offset + kk, axV1);
          _mm512_storeu_ps(ACx + offset + kk + 16, axV2);
        }
      }
    }
    pw_stop_instruments_loop(omp_get_thread_num());
  }
  pw_init_instruments;
}

//void spmmCsrSpmmCsrFusedVectorized64Avx512(
//    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
//    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
//    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
//    const int *ParPtr, const int *Partition, const int *ParType, int NThreads) {
//  pw_init_instruments;
//  for (int i1 = 0; i1 < LevelNo; ++i1) {
//#pragma omp parallel num_threads(NThreads)
//    {
//      pw_start_instruments_loop(omp_get_thread_num());
//#pragma omp for
//      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
//        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
//          int i = Partition[k1];
//          int t = ParType[k1];
//          if (t == 0) {
//            int j = Ap[i];
//            for (; j < Ap[i + 1]-1; j+=2) {
//              vectorCrossProduct2_32Avx512(Ax + j, Ai + j, Cx, ACx, N, i);
//            }
//            for (;j < Ap[i + 1]; j++){
//              vectorCrossProduct64Avx512(Ax[j], Ai[j], Cx, ACx, N, i);
//            }
//          } else {
//            int k = Bp[i];
//            for (; k < Bp[i + 1]-1; k+=2) {
//              vectorCrossProduct2_32Avx512(Bx + k, Bi + k, ACx, Dx, N, i);
//            }
//            for (;k < Bp[i + 1]; k++){
//              vectorCrossProduct64Avx512(Bx[k], Bi[k], ACx, Dx, N, i);
//            }
//          }
//        }
//      }
//      pw_stop_instruments_loop(omp_get_thread_num());
//    }
//  }
//}

void spmmCsrSpmmCsrFusedVectorizedKTiled8Avx512(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads){
  for (int i1 = 0; i1 < LevelNo; ++i1) {
  pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k = 0; k < N; k+=8) {
          for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
            int i = Partition[k1];
            int t = ParType[k1];
            int offset = N * i;
            if (t == 0) {
              auto xv = _mm512_loadu_pd(ACx + offset + k);
              int j = Ap[i];
              for (; j < Ap[i + 1] - 3; j += 4) {
                vectorCrossProduct4_8Avx512(Ax + j, Ai + j, Cx + k, xv, N, i);
              }
              for (; j < Ap[i + 1]; j++) {
                vectorCrossProduct8Avx512(Ax[j], Ai[j], Cx + k, xv, N, i);
              }
              _mm512_storeu_pd(ACx + offset + k, xv);
            } else {
              auto xv = _mm512_loadu_pd(Dx + k + offset);
              int j = Bp[i];
              for (; j < Bp[i + 1] - 3; j += 4) {
                vectorCrossProduct4_8Avx512(Bx + j, Bi + j, ACx + k, xv, N, i);
              }
              for (; j < Bp[i + 1]; j++) {
                vectorCrossProduct8Avx512(Bx[j], Bi[j], ACx + k, xv, N, i);
              }
              _mm512_storeu_pd(Dx + k + offset, xv);
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}

void spmmCsrSpmmCsrFusedVectorized8Avx512(
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
          int offset = N * i;
          if (t == 0) {
            auto xv = _mm512_loadu_pd(ACx + offset);
            int j = Ap[i];
            for (; j < Ap[i + 1]-3; j+=4) {
              vectorCrossProduct4_8Avx512(Ax + j, Ai + j, Cx, xv, N, i);
            }
            for (; j < Ap[i+1];j++){
              vectorCrossProduct8Avx512(Ax[j], Ai[j], Cx, xv, N, i);
            }
            _mm512_storeu_pd(ACx + offset, xv);
          } else {
            auto xv = _mm512_loadu_pd(Dx + offset);
            int k = Bp[i];
            for (; k < Bp[i + 1]-3; k+=4) {
              vectorCrossProduct4_8Avx512(Bx + k, Bi + k, ACx, xv, N, i);
            }
            for(; k < Bp[i+1]; k++){
              vectorCrossProduct8Avx512(Bx[k], Bi[k], ACx, xv, N, i);
            }
            _mm512_storeu_pd(Dx + offset, xv);
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}

void spmmCsrSpmmCscFusedColoredAvx512(int M, int N, int K, int L, const int *Ap,
                                      const int *Ai, const double *Ax, const int *Bp,
                                      const int *Bi, const double *Bx,
                                      const double *Cx, double *Dx, double *ACx,
                                      int LevelNo, const int *LevelPtr, const int *Id,
                                      int TileSize, int NThreads) {
  int lastTileSize = M % TileSize;
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
        for (int ii = 0; ii < TileSize; ++ii) {
          auto ipii = i + ii;
          // first SpMM
          for (int k = 0; k < N; k += 64) {
            auto acxv0 = _mm512_setzero_pd();
            auto acxv1 = _mm512_setzero_pd();
            auto acxv2 = _mm512_setzero_pd();
            auto acxv3 = _mm512_setzero_pd();
            auto acxv4 = _mm512_setzero_pd();
            auto acxv5 = _mm512_setzero_pd();
            auto acxv6 = _mm512_setzero_pd();
            auto acxv7 = _mm512_setzero_pd();
            for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
              int aij = Ai[j] * N;
              auto axv0 = _mm512_set1_pd(Ax[j]);
              auto bxv0 = _mm512_loadu_pd(Cx + aij + k);
              auto bxv1 = _mm512_loadu_pd(Cx + aij + k + 8);
              auto bxv2 = _mm512_loadu_pd(Cx + aij + k + 16);
              auto bxv3 = _mm512_loadu_pd(Cx + aij + k + 24);
              auto bxv4 = _mm512_loadu_pd(Cx + aij + k + 32);
              auto bxv5 = _mm512_loadu_pd(Cx + aij + k + 40);
              auto bxv6 = _mm512_loadu_pd(Cx + aij + k + 48);
              auto bxv7 = _mm512_loadu_pd(Cx + aij + k + 56);
              acxv0 = _mm512_fmadd_pd(axv0, bxv0, acxv0);
              acxv1 = _mm512_fmadd_pd(axv0, bxv1, acxv1);
              acxv2 = _mm512_fmadd_pd(axv0, bxv2, acxv2);
              acxv3 = _mm512_fmadd_pd(axv0, bxv3, acxv3);
              acxv4 = _mm512_fmadd_pd(axv0, bxv4, acxv4);
              acxv5 = _mm512_fmadd_pd(axv0, bxv5, acxv5);
              acxv6 = _mm512_fmadd_pd(axv0, bxv6, acxv6);
              acxv7 = _mm512_fmadd_pd(axv0, bxv7, acxv7);
            }
            // second SpMM CSC
            for (int j = Bp[ipii]; j < Bp[ipii + 1];
                 j++) {
              auto bxv0 = _mm512_set1_pd(Bx[j]);
              auto dxv0 = _mm512_loadu_pd(Dx + Bi[j] * N + k);
              auto dxv1 = _mm512_loadu_pd(Dx + Bi[j] * N + k + 8);
              auto dxv2 = _mm512_loadu_pd(Dx + Bi[j] * N + k + 16);
              auto dxv3 = _mm512_loadu_pd(Dx + Bi[j] * N + k + 24);
              auto dxv4 = _mm512_loadu_pd(Dx + Bi[j] * N + k + 32);
              auto dxv5 = _mm512_loadu_pd(Dx + Bi[j] * N + k + 40);
              auto dxv6 = _mm512_loadu_pd(Dx + Bi[j] * N + k + 48);
              auto dxv7 = _mm512_loadu_pd(Dx + Bi[j] * N + k + 56);
              dxv0 = _mm512_fmadd_pd(bxv0, acxv0, dxv0);
              dxv1 = _mm512_fmadd_pd(bxv0, acxv1, dxv1);
              dxv2 = _mm512_fmadd_pd(bxv0, acxv2, dxv2);
              dxv3 = _mm512_fmadd_pd(bxv0, acxv3, dxv3);
              dxv4 = _mm512_fmadd_pd(bxv0, acxv4, dxv4);
              dxv5 = _mm512_fmadd_pd(bxv0, acxv5, dxv5);
              dxv6 = _mm512_fmadd_pd(bxv0, acxv6, dxv6);
              dxv7 = _mm512_fmadd_pd(bxv0, acxv7, dxv7);
              _mm512_storeu_pd(Dx + Bi[j] * N + k, dxv0);
              _mm512_storeu_pd(Dx + Bi[j] * N + k + 8, dxv1);
              _mm512_storeu_pd(Dx + Bi[j] * N + k + 16, dxv2);
              _mm512_storeu_pd(Dx + Bi[j] * N + k + 24, dxv3);
              _mm512_storeu_pd(Dx + Bi[j] * N + k + 32, dxv4);
              _mm512_storeu_pd(Dx + Bi[j] * N + k + 40, dxv5);
              _mm512_storeu_pd(Dx + Bi[j] * N + k + 48, dxv6);
              _mm512_storeu_pd(Dx + Bi[j] * N + k + 56, dxv7);
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
  int i = M - lastTileSize;
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
    for (int k = Bp[ipii]; k < Bp[ipii + 1]; k++) { // for each column of B
      for (int kk = 0; kk < N; ++kk) {
        int bij = Bi[k] * N;
        Dx[bij + kk] += Bx[k] * ACx[ipii * N + kk];
      }
    }
  }
}


#endif
#ifdef __AVX2__


void spmmCsrSpmmCsrFusedKTiled8Vectorized(int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
                                           const int *Bp, const int *Bi, const double *Bx, const double *Cx,
                                           double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
                                           const int *ParPtr, const int *Partition, const int *ParType, int NThreads) {

  for (int l1 = 0; l1 < LevelNo; ++l1) {
    pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
      for (int j1 = LevelPtr[l1]; j1 < LevelPtr[l1 + 1]; ++j1) {
          for (int k = 0; k < N; k+=8) {
            for (int i1 = ParPtr[j1]; i1 < ParPtr[j1 + 1]; ++i1) {
              int i = Partition[i1];
              int t = ParType[i1];
              if (t == 0) {
                int j = Ap[i];
                auto acxV1 = _mm256_loadu_pd(ACx + i * N + k );
                auto acxV2 = _mm256_loadu_pd(ACx + i * N + k  + 4);
                for (; j < Ap[i + 1] - 1; j += 2) {
                  int aij1 = Ai[j] * N;
                  int aij2 = Ai[j + 1] * N;
                  auto axV1 = _mm256_set1_pd(Ax[j]);
                  auto axV2 = _mm256_set1_pd(Ax[j + 1]);
                  auto cxV11 = _mm256_loadu_pd(Cx + aij1 + k );
                  auto cxV12 = _mm256_loadu_pd(Cx + aij1 + k  + 4);
                  auto cxV21 = _mm256_loadu_pd(Cx + aij2 + k );
                  auto cxV22 = _mm256_loadu_pd(Cx + aij2 + k  + 4);
                  acxV1 = _mm256_fmadd_pd(axV1, cxV11, acxV1);
                  acxV1 = _mm256_fmadd_pd(axV2, cxV21, acxV1);
                  acxV2 = _mm256_fmadd_pd(axV1, cxV12, acxV2);
                  acxV2 = _mm256_fmadd_pd(axV2, cxV22, acxV2);
                }
                for (; j < Ap[i + 1]; ++j) {
                  int aij = Ai[j] * N;
                  auto axv0 = _mm256_set1_pd(Ax[j]);
                  auto cxV11 = _mm256_loadu_pd(Cx + aij + k);
                  auto cxV12 = _mm256_loadu_pd(Cx + aij + k + 4);
                  acxV1 = _mm256_fmadd_pd(axv0, cxV11, acxV1);
                  acxV2 = _mm256_fmadd_pd(axv0, cxV12, acxV2);
                }
                _mm256_storeu_pd(ACx + i * N + k , acxV1);
                _mm256_storeu_pd(ACx + i * N + k  + 4, acxV2);
              } else {
                int j = Bp[i];
                auto dxV1 = _mm256_loadu_pd(Dx + i * N + k);
                auto dxV2 = _mm256_loadu_pd(Dx + i * N + k + 4);
                for (; j < Bp[i + 1] - 1; j += 2) {
                  int bij1 = Bi[j] * N;
                  int bij2 = Bi[j + 1] * N;
                  auto bxV1 = _mm256_set1_pd(Bx[j]);
                  auto bxV2 = _mm256_set1_pd(Bx[j + 1]);
                  auto acxV11 = _mm256_loadu_pd(ACx + bij1 + k);
                  auto acxV12 = _mm256_loadu_pd(ACx + bij1 + k + 4);
                  auto acxV21 = _mm256_loadu_pd(ACx + bij2 + k);
                  auto acxV22 = _mm256_loadu_pd(ACx + bij2 + k + 4);
                  dxV1 = _mm256_fmadd_pd(bxV1, acxV11, dxV1);
                  dxV1 = _mm256_fmadd_pd(bxV2, acxV21, dxV1);
                  dxV2 = _mm256_fmadd_pd(bxV1, acxV12, dxV2);
                  dxV2 = _mm256_fmadd_pd(bxV2, acxV22, dxV2);
                }
                for (; j < Bp[i + 1]; ++j) {
                  int bij = Bi[j] * N;
                  auto bxv0 = _mm256_set1_pd(Bx[j]);
                  auto cxV11 = _mm256_loadu_pd(ACx + bij + k);
                  auto cxV12 = _mm256_loadu_pd(ACx + bij + k + 4);
                  dxV1 = _mm256_fmadd_pd(bxv0, cxV11, dxV1);
                  dxV2 = _mm256_fmadd_pd(bxv0, cxV12, dxV2);
                }
                _mm256_storeu_pd(Dx + i * N + k, dxV1);
                _mm256_storeu_pd(Dx + i * N + k + 4, dxV2);
              }
            }
          }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}
void spmmCsrSpmmCsrFusedVectorized2_8(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads) {

  for (int i1 = 0; i1 < LevelNo; ++i1) {
    pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            int j = Ap[i];
            for (; j < Ap[i + 1]-1; j+=2) {
              int aij1 = Ai[j] * N;
              int aij2 = Ai[j+1] * N;
              auto axV1 = _mm256_set1_pd(Ax[j]);
              auto axV2 = _mm256_set1_pd(Ax[j+1]);
              for (int kk = 0; kk < N; kk += 8) {
                auto cxV11 = _mm256_loadu_pd(Cx + aij1 + kk);
                auto cxV12 = _mm256_loadu_pd(Cx + aij1 + kk + 4);
                auto cxV21 = _mm256_loadu_pd(Cx + aij2 + kk);
                auto cxV22 = _mm256_loadu_pd(Cx + aij2 + kk + 4);
                auto acxV1 = _mm256_loadu_pd(ACx + i * N + kk);
                auto acxV2 = _mm256_loadu_pd(ACx + i * N + kk + 4);
                acxV1 = _mm256_fmadd_pd(axV1, cxV11, acxV1);
                acxV1 = _mm256_fmadd_pd(axV2, cxV21, acxV1);
                _mm256_storeu_pd(ACx + i * N + kk, acxV1);
                acxV2 = _mm256_fmadd_pd(axV1, cxV12, acxV2);
                acxV2 = _mm256_fmadd_pd(axV2, cxV22, acxV2);
                _mm256_storeu_pd(ACx + i * N + kk + 4, acxV2);
              }
            }
            for (; j < Ap[i + 1]; ++j) {
              int aij = Ai[j] * N;
              auto axv0 = _mm256_set1_pd(Ax[j]);
              for (int kk = 0; kk < N; kk += 8) {
                auto cxV11 = _mm256_loadu_pd(Cx + aij + kk);
                auto cxV12 = _mm256_loadu_pd(Cx + aij + kk + 4);
                auto acxV = _mm256_loadu_pd(ACx + i * N + kk);
                auto acxV2 = _mm256_loadu_pd(ACx + i * N + kk + 4);
                acxV = _mm256_fmadd_pd(axv0, cxV11, acxV);
                acxV2 = _mm256_fmadd_pd(axv0, cxV12, acxV2);
                _mm256_storeu_pd(ACx + i * N + kk, acxV);
                _mm256_storeu_pd(ACx + i * N + kk + 4, acxV2);
              }
            }
          } else {
            int k = Bp[i];
            for (; k < Bp[i + 1]-1; k+=2) {
              int bij1 = Bi[k] * N;
              int bij2 = Bi[k+1] * N;
              auto bxV1 = _mm256_set1_pd(Bx[k]);
              auto bxV2 = _mm256_set1_pd(Bx[k+1]);
              for (int kk = 0; kk < N; kk += 8) {
                auto acxV11 = _mm256_loadu_pd(ACx + bij1 + kk);
                auto acxV12 = _mm256_loadu_pd(ACx + bij1 + kk + 4);
                auto acxV21 = _mm256_loadu_pd(ACx + bij2 + kk);
                auto acxV22 = _mm256_loadu_pd(ACx + bij2 + kk + 4);
                auto dxV1 = _mm256_loadu_pd(Dx + i * N + kk);
                auto dxV2 = _mm256_loadu_pd(Dx + i * N + kk + 4);
                dxV1 = _mm256_fmadd_pd(bxV1, acxV11, dxV1);
                dxV1 = _mm256_fmadd_pd(bxV2, acxV21, dxV1);
                _mm256_storeu_pd(Dx + i * N + kk, dxV1);
                dxV2 = _mm256_fmadd_pd(bxV1, acxV12, dxV2);
                dxV2 = _mm256_fmadd_pd(bxV2, acxV22, dxV2);
                _mm256_storeu_pd(Dx + i * N + kk + 4, dxV2);
              }
            }
            for (; k < Bp[i + 1]; ++k) {
              int bij = Bi[k] * N;
              auto bxv0 = _mm256_set1_pd(Bx[k]);
              for (int kk = 0; kk < N; kk += 8) {
                auto cxV11 = _mm256_loadu_pd(ACx + bij + kk);
                auto cxV12 = _mm256_loadu_pd(ACx + bij + kk + 4);
                auto dxV = _mm256_loadu_pd(Dx + i * N + kk);
                auto dxV2 = _mm256_loadu_pd(Dx + i * N + kk + 4);
                dxV = _mm256_fmadd_pd(bxv0, cxV11, dxV);
                dxV2 = _mm256_fmadd_pd(bxv0, cxV12, dxV2);
                _mm256_storeu_pd(Dx + i * N + kk, dxV);
                _mm256_storeu_pd(Dx + i * N + kk + 4, dxV2);
              }
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}

void spmmCsrSpmmCsrFusedVectorized2_16(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads) {
  for (int i1 = 0; i1 < LevelNo; ++i1) {
    pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            for (int kk = 0; kk < N; kk += 16) {
              auto acxV1 = _mm256_loadu_pd(ACx + i * N + kk);
              auto acxV2 = _mm256_loadu_pd(ACx + i * N + kk + 4);
              auto acxV3 = _mm256_loadu_pd(ACx + i * N + kk + 8);
              auto acxV4 = _mm256_loadu_pd(ACx + i * N + kk + 12);
              int j = Ap[i];
              for (; j < Ap[i + 1]-1; j+=2) {
                int aij1 = Ai[j] * N;
                int aij2 = Ai[j+1] * N;
                auto axV1 = _mm256_set1_pd(Ax[j]);
                auto axV2 = _mm256_set1_pd(Ax[j+1]);
                auto cxV11 = _mm256_loadu_pd(Cx + aij1 + kk);
                auto cxV12 = _mm256_loadu_pd(Cx + aij1 + kk + 4);
                auto cxV13 = _mm256_loadu_pd(Cx + aij1 + kk + 8);
                auto cxV14 = _mm256_loadu_pd(Cx + aij1 + kk + 12);
                auto cxV21 = _mm256_loadu_pd(Cx + aij2 + kk);
                auto cxV22 = _mm256_loadu_pd(Cx + aij2 + kk + 4);
                auto cxV23 = _mm256_loadu_pd(Cx + aij2 + kk + 8);
                auto cxV24 = _mm256_loadu_pd(Cx + aij2 + kk + 12);
                acxV1 = _mm256_fmadd_pd(axV1, cxV11, acxV1);
                acxV1 = _mm256_fmadd_pd(axV2, cxV21, acxV1);
                acxV2 = _mm256_fmadd_pd(axV1, cxV12, acxV2);
                acxV2 = _mm256_fmadd_pd(axV2, cxV22, acxV2);
                acxV3 = _mm256_fmadd_pd(axV1, cxV13, acxV3);
                acxV3 = _mm256_fmadd_pd(axV2, cxV23, acxV3);
                acxV4 = _mm256_fmadd_pd(axV1, cxV14, acxV4);
                acxV4 = _mm256_fmadd_pd(axV2, cxV24, acxV4);
              }
              for (; j < Ap[i + 1]; ++j) {
                int aij = Ai[j] * N;
                auto axv0 = _mm256_set1_pd(Ax[j]);
                auto cxV11 = _mm256_loadu_pd(Cx + aij + kk);
                auto cxV12 = _mm256_loadu_pd(Cx + aij + kk + 4);
                auto cxV13 = _mm256_loadu_pd(Cx + aij + kk + 8);
                auto cxV14 = _mm256_loadu_pd(Cx + aij + kk + 12);
                acxV1 = _mm256_fmadd_pd(axv0, cxV11, acxV1);
                acxV2 = _mm256_fmadd_pd(axv0, cxV12, acxV2);
                acxV3 = _mm256_fmadd_pd(axv0, cxV13, acxV3);
                acxV4 = _mm256_fmadd_pd(axv0, cxV14, acxV4);
              }
              _mm256_storeu_pd(ACx + i * N + kk, acxV1);
              _mm256_storeu_pd(ACx + i * N + kk + 4, acxV2);
              _mm256_storeu_pd(ACx + i * N + kk + 8, acxV3);
              _mm256_storeu_pd(ACx + i * N + kk + 12, acxV4);
            }
          } else {
            for (int kk = 0; kk < N; kk += 16) {
              auto dxV1 = _mm256_loadu_pd(Dx + i * N + kk);
              auto dxV2 = _mm256_loadu_pd(Dx + i * N + kk + 4);
              auto dxV3 = _mm256_loadu_pd(Dx + i * N + kk + 8);
              auto dxV4 = _mm256_loadu_pd(Dx + i * N + kk + 12);
              int k = Bp[i];
              for (; k < Bp[i + 1]-1; k+=2) {
                int bij1 = Bi[k] * N;
                int bij2 = Bi[k+1] * N;
                auto bxV1 = _mm256_set1_pd(Bx[k]);
                auto bxV2 = _mm256_set1_pd(Bx[k+1]);
                auto acxV11 = _mm256_loadu_pd(ACx + bij1 + kk);
                auto acxV12 = _mm256_loadu_pd(ACx + bij1 + kk + 4);
                auto acxV13 = _mm256_loadu_pd(ACx + bij1 + kk + 8);
                auto acxV14 = _mm256_loadu_pd(ACx + bij1 + kk + 12);
                auto acxV21 = _mm256_loadu_pd(ACx + bij2 + kk);
                auto acxV22 = _mm256_loadu_pd(ACx + bij2 + kk + 4);
                auto acxV23 = _mm256_loadu_pd(ACx + bij2 + kk + 8);
                auto acxV24 = _mm256_loadu_pd(ACx + bij2 + kk + 12);
                dxV1 = _mm256_fmadd_pd(bxV1, acxV11, dxV1);
                dxV1 = _mm256_fmadd_pd(bxV2, acxV21, dxV1);
                dxV2 = _mm256_fmadd_pd(bxV1, acxV12, dxV2);
                dxV2 = _mm256_fmadd_pd(bxV2, acxV22, dxV2);
                dxV3 = _mm256_fmadd_pd(bxV1, acxV13, dxV3);
                dxV3 = _mm256_fmadd_pd(bxV2, acxV23, dxV3);
                dxV4 = _mm256_fmadd_pd(bxV1, acxV14, dxV4);
                dxV4 = _mm256_fmadd_pd(bxV2, acxV24, dxV4);
              }
              for (; k < Bp[i + 1]; ++k) {
                int bij = Bi[k] * N;
                auto bxv0 = _mm256_set1_pd(Bx[k]);
                auto cxV11 = _mm256_loadu_pd(ACx + bij + kk);
                auto cxV12 = _mm256_loadu_pd(ACx + bij + kk + 4);
                auto cxV13 = _mm256_loadu_pd(ACx + bij + kk + 8);
                auto cxV14 = _mm256_loadu_pd(ACx + bij + kk + 12);
                dxV1 = _mm256_fmadd_pd(bxv0, cxV11, dxV1);
                dxV2 = _mm256_fmadd_pd(bxv0, cxV12, dxV2);
                dxV3 = _mm256_fmadd_pd(bxv0, cxV13, dxV3);
                dxV4 = _mm256_fmadd_pd(bxv0, cxV14, dxV4);
              }
              _mm256_storeu_pd(Dx + i * N + kk, dxV1);
              _mm256_storeu_pd(Dx + i * N + kk + 4, dxV2);
              _mm256_storeu_pd(Dx + i * N + kk + 8, dxV3);
              _mm256_storeu_pd(Dx + i * N + kk + 12, dxV4);
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}

void spmmCsrSpmmCsrFusedVectorized2_32SP(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const float *Ax,
    const int *Bp, const int *Bi, const float *Bx, const float *Cx,
    float *Dx, float *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads) {
  for (int i1 = 0; i1 < LevelNo; ++i1) {
    pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            for (int kk = 0; kk < N; kk += 32) {
              auto acxV1 = _mm256_loadu_ps(ACx + i * N + kk);
              auto acxV2 = _mm256_loadu_ps(ACx + i * N + kk + 8);
              auto acxV3 = _mm256_loadu_ps(ACx + i * N + kk + 16);
              auto acxV4 = _mm256_loadu_ps(ACx + i * N + kk + 24);
              int j = Ap[i];
              for (; j < Ap[i + 1]-1; j+=2) {
                int aij1 = Ai[j] * N;
                int aij2 = Ai[j+1] * N;
                auto axV1 = _mm256_set1_ps(Ax[j]);
                auto axV2 = _mm256_set1_ps(Ax[j+1]);
                auto cxV11 = _mm256_loadu_ps(Cx + aij1 + kk);
                auto cxV12 = _mm256_loadu_ps(Cx + aij1 + kk + 8);
                auto cxV13 = _mm256_loadu_ps(Cx + aij1 + kk + 16);
                auto cxV14 = _mm256_loadu_ps(Cx + aij1 + kk + 24);
                auto cxV21 = _mm256_loadu_ps(Cx + aij2 + kk);
                auto cxV22 = _mm256_loadu_ps(Cx + aij2 + kk + 8);
                auto cxV23 = _mm256_loadu_ps(Cx + aij2 + kk + 16);
                auto cxV24 = _mm256_loadu_ps(Cx + aij2 + kk + 24);
                acxV1 = _mm256_fmadd_ps(axV1, cxV11, acxV1);
                acxV1 = _mm256_fmadd_ps(axV2, cxV21, acxV1);
                acxV2 = _mm256_fmadd_ps(axV1, cxV12, acxV2);
                acxV2 = _mm256_fmadd_ps(axV2, cxV22, acxV2);
                acxV3 = _mm256_fmadd_ps(axV1, cxV13, acxV3);
                acxV3 = _mm256_fmadd_ps(axV2, cxV23, acxV3);
                acxV4 = _mm256_fmadd_ps(axV1, cxV14, acxV4);
                acxV4 = _mm256_fmadd_ps(axV2, cxV24, acxV4);
              }
              for (; j < Ap[i + 1]; ++j) {
                int aij = Ai[j] * N;
                auto axv0 = _mm256_set1_ps(Ax[j]);
                auto cxV11 = _mm256_loadu_ps(Cx + aij + kk);
                auto cxV12 = _mm256_loadu_ps(Cx + aij + kk + 8);
                auto cxV13 = _mm256_loadu_ps(Cx + aij + kk + 16);
                auto cxV14 = _mm256_loadu_ps(Cx + aij + kk + 24);
                acxV1 = _mm256_fmadd_ps(axv0, cxV11, acxV1);
                acxV2 = _mm256_fmadd_ps(axv0, cxV12, acxV2);
                acxV3 = _mm256_fmadd_ps(axv0, cxV13, acxV3);
                acxV4 = _mm256_fmadd_ps(axv0, cxV14, acxV4);
              }
              _mm256_storeu_ps(ACx + i * N + kk, acxV1);
              _mm256_storeu_ps(ACx + i * N + kk + 8, acxV2);
              _mm256_storeu_ps(ACx + i * N + kk + 16, acxV3);
              _mm256_storeu_ps(ACx + i * N + kk + 24, acxV4);
            }
          } else {
            for (int kk = 0; kk < N; kk += 32) {
              auto dxV1 = _mm256_loadu_ps(Dx + i * N + kk);
              auto dxV2 = _mm256_loadu_ps(Dx + i * N + kk + 8);
              auto dxV3 = _mm256_loadu_ps(Dx + i * N + kk + 16);
              auto dxV4 = _mm256_loadu_ps(Dx + i * N + kk + 24);
              int k = Bp[i];
              for (; k < Bp[i + 1]-1; k+=2) {
                int bij1 = Bi[k] * N;
                int bij2 = Bi[k+1] * N;
                auto bxV1 = _mm256_set1_ps(Bx[k]);
                auto bxV2 = _mm256_set1_ps(Bx[k+1]);
                auto acxV11 = _mm256_loadu_ps(ACx + bij1 + kk);
                auto acxV12 = _mm256_loadu_ps(ACx + bij1 + kk + 8);
                auto acxV13 = _mm256_loadu_ps(ACx + bij1 + kk + 16);
                auto acxV14 = _mm256_loadu_ps(ACx + bij1 + kk + 24);
                auto acxV21 = _mm256_loadu_ps(ACx + bij2 + kk);
                auto acxV22 = _mm256_loadu_ps(ACx + bij2 + kk + 8);
                auto acxV23 = _mm256_loadu_ps(ACx + bij2 + kk + 16);
                auto acxV24 = _mm256_loadu_ps(ACx + bij2 + kk + 24);
                dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
                dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
                dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
                dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
                dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
                dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
                dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
                dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
              }
              for (; k < Bp[i + 1]; ++k) {
                int bij = Bi[k] * N;
                auto bxv0 = _mm256_set1_ps(Bx[k]);
                auto cxV11 = _mm256_loadu_ps(ACx + bij + kk);
                auto cxV12 = _mm256_loadu_ps(ACx + bij + kk + 8);
                auto cxV13 = _mm256_loadu_ps(ACx + bij + kk + 16);
                auto cxV14 = _mm256_loadu_ps(ACx + bij + kk + 24);
                dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
                dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
                dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
                dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
              }
              _mm256_storeu_ps(Dx + i * N + kk, dxV1);
              _mm256_storeu_ps(Dx + i * N + kk + 8, dxV2);
              _mm256_storeu_ps(Dx + i * N + kk + 16, dxV3);
              _mm256_storeu_ps(Dx + i * N + kk + 24, dxV4);
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}


//only for two kernels right now.
void spmmCsrSpmmCsrOneSparseMatrixFusedVectorized2_32SP(
    int M, int N, int K, int L,  const int *__restrict__ Ap, const int *__restrict__ Ai, const float *__restrict__ Ax,
    const float *Cx, float *Dx, float *__restrict__ ACx,
    int LevelNo, const int *LevelPtr,const int *ParPtr,
    const int* Partition, const int *MixPtr, int NThreads) {
  int numKernels = 2;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
    pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
          int kBeginL1 = ParPtr[j1];
          int kEndL1 = MixPtr[j1 * numKernels];
          int kEndL2 = MixPtr[j1 * numKernels + 1];
          int tileSize = kEndL1 - kBeginL1;
          int iL1 = Partition[kBeginL1];
          for (int i = iL1; i < iL1 + tileSize; i++){
            for (int kk = 0; kk < N; kk += 32) {
              auto acxV1 = _mm256_loadu_ps(ACx + i * N + kk);
              auto acxV2 = _mm256_loadu_ps(ACx + i * N + kk + 8);
              auto acxV3 = _mm256_loadu_ps(ACx + i * N + kk + 16);
              auto acxV4 = _mm256_loadu_ps(ACx + i * N + kk + 24);
              int j = Ap[i];
              for (; j < Ap[i + 1]-1; j+=2) {
                int aij1 = Ai[j] * N;
                int aij2 = Ai[j+1] * N;
                auto axV1 = _mm256_set1_ps(Ax[j]);
                auto axV2 = _mm256_set1_ps(Ax[j+1]);
                auto cxV11 = _mm256_loadu_ps(Cx + aij1 + kk);
                auto cxV12 = _mm256_loadu_ps(Cx + aij1 + kk + 8);
                auto cxV13 = _mm256_loadu_ps(Cx + aij1 + kk + 16);
                auto cxV14 = _mm256_loadu_ps(Cx + aij1 + kk + 24);
                auto cxV21 = _mm256_loadu_ps(Cx + aij2 + kk);
                auto cxV22 = _mm256_loadu_ps(Cx + aij2 + kk + 8);
                auto cxV23 = _mm256_loadu_ps(Cx + aij2 + kk + 16);
                auto cxV24 = _mm256_loadu_ps(Cx + aij2 + kk + 24);
                acxV1 = _mm256_fmadd_ps(axV1, cxV11, acxV1);
                acxV1 = _mm256_fmadd_ps(axV2, cxV21, acxV1);
                acxV2 = _mm256_fmadd_ps(axV1, cxV12, acxV2);
                acxV2 = _mm256_fmadd_ps(axV2, cxV22, acxV2);
                acxV3 = _mm256_fmadd_ps(axV1, cxV13, acxV3);
                acxV3 = _mm256_fmadd_ps(axV2, cxV23, acxV3);
                acxV4 = _mm256_fmadd_ps(axV1, cxV14, acxV4);
                acxV4 = _mm256_fmadd_ps(axV2, cxV24, acxV4);
              }
              for (; j < Ap[i + 1]; ++j) {
                int aij = Ai[j] * N;
                auto axv0 = _mm256_set1_ps(Ax[j]);
                auto cxV11 = _mm256_loadu_ps(Cx + aij + kk);
                auto cxV12 = _mm256_loadu_ps(Cx + aij + kk + 8);
                auto cxV13 = _mm256_loadu_ps(Cx + aij + kk + 16);
                auto cxV14 = _mm256_loadu_ps(Cx + aij + kk + 24);
                acxV1 = _mm256_fmadd_ps(axv0, cxV11, acxV1);
                acxV2 = _mm256_fmadd_ps(axv0, cxV12, acxV2);
                acxV3 = _mm256_fmadd_ps(axv0, cxV13, acxV3);
                acxV4 = _mm256_fmadd_ps(axv0, cxV14, acxV4);
              }
              _mm256_storeu_ps(ACx + i * N + kk, acxV1);
              _mm256_storeu_ps(ACx + i * N + kk + 8, acxV2);
              _mm256_storeu_ps(ACx + i * N + kk + 16, acxV3);
              _mm256_storeu_ps(ACx + i * N + kk + 24, acxV4);
            }
          }
          for (int k1 = kEndL1; k1 < kEndL2; k1++){
            int i = Partition[k1];
            for (int kk = 0; kk < N; kk += 32) {
              auto dxV1 = _mm256_loadu_ps(Dx + i * N + kk);
              auto dxV2 = _mm256_loadu_ps(Dx + i * N + kk + 8);
              auto dxV3 = _mm256_loadu_ps(Dx + i * N + kk + 16);
              auto dxV4 = _mm256_loadu_ps(Dx + i * N + kk + 24);
              int k = Ap[i];
              for (; k < Ap[i + 1]-1; k+=2) {
                int bij1 = Ai[k] * N;
                int bij2 = Ai[k+1] * N;
                auto bxV1 = _mm256_set1_ps(Ax[k]);
                auto bxV2 = _mm256_set1_ps(Ax[k+1]);
                auto acxV11 = _mm256_loadu_ps(ACx + bij1 + kk);
                auto acxV12 = _mm256_loadu_ps(ACx + bij1 + kk + 8);
                auto acxV13 = _mm256_loadu_ps(ACx + bij1 + kk + 16);
                auto acxV14 = _mm256_loadu_ps(ACx + bij1 + kk + 24);
                auto acxV21 = _mm256_loadu_ps(ACx + bij2 + kk);
                auto acxV22 = _mm256_loadu_ps(ACx + bij2 + kk + 8);
                auto acxV23 = _mm256_loadu_ps(ACx + bij2 + kk + 16);
                auto acxV24 = _mm256_loadu_ps(ACx + bij2 + kk + 24);
                dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
                dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
                dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
                dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
                dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
                dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
                dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
                dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
              }
              for (; k < Ap[i + 1]; ++k) {
                int bij = Ai[k] * N;
                auto bxv0 = _mm256_set1_ps(Ax[k]);
                auto cxV11 = _mm256_loadu_ps(ACx + bij + kk);
                auto cxV12 = _mm256_loadu_ps(ACx + bij + kk + 8);
                auto cxV13 = _mm256_loadu_ps(ACx + bij + kk + 16);
                auto cxV14 = _mm256_loadu_ps(ACx + bij + kk + 24);
                dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
                dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
                dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
                dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
              }
              _mm256_storeu_ps(Dx + i * N + kk, dxV1);
              _mm256_storeu_ps(Dx + i * N + kk + 8, dxV2);
              _mm256_storeu_ps(Dx + i * N + kk + 16, dxV3);
              _mm256_storeu_ps(Dx + i * N + kk + 24, dxV4);
            }
          }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}


//only for odd kernels right now.
void spmmCsrSpmmCsrOneSparseMatrixFusedVectorizedReorderedUnfused_32SP(
    int M, int N, int K, int L,  const int *__restrict__ Ap, const int *__restrict__ Ai, const float *__restrict__ Ax,
    const int *__restrict__ UFAp, const int *__restrict__ UFAi, const float *__restrict__ UFAx,
    const float *Cx, float *Dx, float *__restrict__ ACx,
    int LevelNo, const int *LevelPtr,const int *ParPtr,
    const int* Partition, const int *MixPtr, int NThreads) {
  int numKernels = 2;
  for (int i1 = 0; i1 < LevelNo; i1+=2) {
    pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
          int kBeginL1 = ParPtr[j1];
          int kEndL1 = MixPtr[j1 * numKernels];
          int kEndL2 = MixPtr[j1 * numKernels + 1];
          int tileSize = kEndL1 - kBeginL1;
          int iL1 = Partition[kBeginL1];
          for (int i = iL1; i < iL1 + tileSize; i++){
            for (int kk = 0; kk < N; kk += 32) {
              auto acxV1 = _mm256_loadu_ps(ACx + i * N + kk);
              auto acxV2 = _mm256_loadu_ps(ACx + i * N + kk + 8);
              auto acxV3 = _mm256_loadu_ps(ACx + i * N + kk + 16);
              auto acxV4 = _mm256_loadu_ps(ACx + i * N + kk + 24);
              int j = Ap[i];
              for (; j < Ap[i + 1]-1; j+=2) {
                int aij1 = Ai[j] * N;
                int aij2 = Ai[j+1] * N;
                auto axV1 = _mm256_set1_ps(Ax[j]);
                auto axV2 = _mm256_set1_ps(Ax[j+1]);
                auto cxV11 = _mm256_loadu_ps(Cx + aij1 + kk);
                auto cxV12 = _mm256_loadu_ps(Cx + aij1 + kk + 8);
                auto cxV13 = _mm256_loadu_ps(Cx + aij1 + kk + 16);
                auto cxV14 = _mm256_loadu_ps(Cx + aij1 + kk + 24);
                auto cxV21 = _mm256_loadu_ps(Cx + aij2 + kk);
                auto cxV22 = _mm256_loadu_ps(Cx + aij2 + kk + 8);
                auto cxV23 = _mm256_loadu_ps(Cx + aij2 + kk + 16);
                auto cxV24 = _mm256_loadu_ps(Cx + aij2 + kk + 24);
                acxV1 = _mm256_fmadd_ps(axV1, cxV11, acxV1);
                acxV1 = _mm256_fmadd_ps(axV2, cxV21, acxV1);
                acxV2 = _mm256_fmadd_ps(axV1, cxV12, acxV2);
                acxV2 = _mm256_fmadd_ps(axV2, cxV22, acxV2);
                acxV3 = _mm256_fmadd_ps(axV1, cxV13, acxV3);
                acxV3 = _mm256_fmadd_ps(axV2, cxV23, acxV3);
                acxV4 = _mm256_fmadd_ps(axV1, cxV14, acxV4);
                acxV4 = _mm256_fmadd_ps(axV2, cxV24, acxV4);
              }
              for (; j < Ap[i + 1]; ++j) {
                int aij = Ai[j] * N;
                auto axv0 = _mm256_set1_ps(Ax[j]);
                auto cxV11 = _mm256_loadu_ps(Cx + aij + kk);
                auto cxV12 = _mm256_loadu_ps(Cx + aij + kk + 8);
                auto cxV13 = _mm256_loadu_ps(Cx + aij + kk + 16);
                auto cxV14 = _mm256_loadu_ps(Cx + aij + kk + 24);
                acxV1 = _mm256_fmadd_ps(axv0, cxV11, acxV1);
                acxV2 = _mm256_fmadd_ps(axv0, cxV12, acxV2);
                acxV3 = _mm256_fmadd_ps(axv0, cxV13, acxV3);
                acxV4 = _mm256_fmadd_ps(axv0, cxV14, acxV4);
              }
              _mm256_storeu_ps(ACx + i * N + kk, acxV1);
              _mm256_storeu_ps(ACx + i * N + kk + 8, acxV2);
              _mm256_storeu_ps(ACx + i * N + kk + 16, acxV3);
              _mm256_storeu_ps(ACx + i * N + kk + 24, acxV4);
            }
          }
          for (int k1 = kEndL1; k1 < kEndL2; k1++){
            int i = Partition[k1];
            for (int kk = 0; kk < N; kk += 32) {
              auto dxV1 = _mm256_loadu_ps(Dx + i * N + kk);
              auto dxV2 = _mm256_loadu_ps(Dx + i * N + kk + 8);
              auto dxV3 = _mm256_loadu_ps(Dx + i * N + kk + 16);
              auto dxV4 = _mm256_loadu_ps(Dx + i * N + kk + 24);
              int k = Ap[i];
              for (; k < Ap[i + 1]-1; k+=2) {
                int bij1 = Ai[k] * N;
                int bij2 = Ai[k+1] * N;
                auto bxV1 = _mm256_set1_ps(Ax[k]);
                auto bxV2 = _mm256_set1_ps(Ax[k+1]);
                auto acxV11 = _mm256_loadu_ps(ACx + bij1 + kk);
                auto acxV12 = _mm256_loadu_ps(ACx + bij1 + kk + 8);
                auto acxV13 = _mm256_loadu_ps(ACx + bij1 + kk + 16);
                auto acxV14 = _mm256_loadu_ps(ACx + bij1 + kk + 24);
                auto acxV21 = _mm256_loadu_ps(ACx + bij2 + kk);
                auto acxV22 = _mm256_loadu_ps(ACx + bij2 + kk + 8);
                auto acxV23 = _mm256_loadu_ps(ACx + bij2 + kk + 16);
                auto acxV24 = _mm256_loadu_ps(ACx + bij2 + kk + 24);
                dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
                dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
                dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
                dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
                dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
                dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
                dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
                dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
              }
              for (; k < Ap[i + 1]; ++k) {
                int bij = Ai[k] * N;
                auto bxv0 = _mm256_set1_ps(Ax[k]);
                auto cxV11 = _mm256_loadu_ps(ACx + bij + kk);
                auto cxV12 = _mm256_loadu_ps(ACx + bij + kk + 8);
                auto cxV13 = _mm256_loadu_ps(ACx + bij + kk + 16);
                auto cxV14 = _mm256_loadu_ps(ACx + bij + kk + 24);
                dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
                dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
                dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
                dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
              }
              _mm256_storeu_ps(Dx + i * N + kk, dxV1);
              _mm256_storeu_ps(Dx + i * N + kk + 8, dxV2);
              _mm256_storeu_ps(Dx + i * N + kk + 16, dxV3);
              _mm256_storeu_ps(Dx + i * N + kk + 24, dxV4);
            }
          }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }

    int unfusedStart = MixPtr[LevelPtr[1] * numKernels];
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
      for (int j1 = LevelPtr[i1+1]; j1 < LevelPtr[i1 + 2]; ++j1) {
          int kEndL1 = MixPtr[j1 * numKernels];
          int kEndL2 = MixPtr[j1 * numKernels + 1];
          for (int k1 = kEndL1; k1 < kEndL2; k1++){
            int sRow = k1 - unfusedStart;
            int i = Partition[k1];
            for (int kk = 0; kk < N; kk += 32) {
              auto dxV1 = _mm256_loadu_ps(Dx + i * N + kk);
              auto dxV2 = _mm256_loadu_ps(Dx + i * N + kk + 8);
              auto dxV3 = _mm256_loadu_ps(Dx + i * N + kk + 16);
              auto dxV4 = _mm256_loadu_ps(Dx + i * N + kk + 24);
              int k = UFAp[sRow];
              for (; k < UFAp[sRow + 1]-1; k+=2) {
                int bij1 = UFAi[k] * N;
                int bij2 = UFAi[k+1] * N;
                auto bxV1 = _mm256_set1_ps(UFAx[k]);
                auto bxV2 = _mm256_set1_ps(UFAx[k+1]);
                auto acxV11 = _mm256_loadu_ps(ACx + bij1 + kk);
                auto acxV12 = _mm256_loadu_ps(ACx + bij1 + kk + 8);
                auto acxV13 = _mm256_loadu_ps(ACx + bij1 + kk + 16);
                auto acxV14 = _mm256_loadu_ps(ACx + bij1 + kk + 24);
                auto acxV21 = _mm256_loadu_ps(ACx + bij2 + kk);
                auto acxV22 = _mm256_loadu_ps(ACx + bij2 + kk + 8);
                auto acxV23 = _mm256_loadu_ps(ACx + bij2 + kk + 16);
                auto acxV24 = _mm256_loadu_ps(ACx + bij2 + kk + 24);
                dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
                dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
                dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
                dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
                dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
                dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
                dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
                dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
              }
              for (; k < UFAp[sRow + 1]; ++k) {
                int bij = UFAi[k] * N;
                auto bxv0 = _mm256_set1_ps(UFAx[k]);
                auto cxV11 = _mm256_loadu_ps(ACx + bij + kk);
                auto cxV12 = _mm256_loadu_ps(ACx + bij + kk + 8);
                auto cxV13 = _mm256_loadu_ps(ACx + bij + kk + 16);
                auto cxV14 = _mm256_loadu_ps(ACx + bij + kk + 24);
                dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
                dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
                dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
                dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
              }
              _mm256_storeu_ps(Dx + i * N + kk, dxV1);
              _mm256_storeu_ps(Dx + i * N + kk + 8, dxV2);
              _mm256_storeu_ps(Dx + i * N + kk + 16, dxV3);
              _mm256_storeu_ps(Dx + i * N + kk + 24, dxV4);
            }
          }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
}

#ifdef __AVX2__
#define BUSY_WAIT_CONSTRUCT(task) {\
  int n = NParents[task]; \
  int *c = Parents[task]; \
  for (int pCntr = 0; pCntr < n; ++pCntr) \
    while (!TaskFinished[c[pCntr]]) _mm_pause(); \
}
#else
#define BUSY_WAIT_CONSTRUCT(task) {\
  int n = NParents[task]; \
  int *c = Parents[task]; \
  for (int i = 0; i < n; ++i) \
    while (!TaskFinished[c[i]]); \
}
#endif

void spmmCsrSpmmCsrOneSparseMatrixFusedVectorized2P2PThreading_32SP(
    int M, int N, int K, int L,  const int *__restrict__ Ap, const int *__restrict__ Ai, const float *__restrict__ Ax,
    const float *Cx, float *Dx, float *__restrict__ ACx,
    int LevelNo, const int *LevelPtr,const int *ParPtr,
    const int* Partition, const int *MixPtr, int NThreads,
    int *NParents, int **Parents, bool *TaskFinished) {

  int numKernels = 2;
    pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
    {
      pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
      for (int j1 = LevelPtr[0]; j1 < LevelPtr[numKernels]; ++j1) {
        BUSY_WAIT_CONSTRUCT(j1);
        int kBeginL1 = ParPtr[j1];
        int kEndL1 = MixPtr[j1 * numKernels];
        int kEndL2 = MixPtr[j1 * numKernels + 1];
        int tileSize = kEndL1 - kBeginL1;
        int iL1 = Partition[kBeginL1];
        for (int i = iL1; i < iL1 + tileSize; i++){
          for (int kk = 0; kk < N; kk += 32) {
            auto acxV1 = _mm256_loadu_ps(ACx + i * N + kk);
            auto acxV2 = _mm256_loadu_ps(ACx + i * N + kk + 8);
            auto acxV3 = _mm256_loadu_ps(ACx + i * N + kk + 16);
            auto acxV4 = _mm256_loadu_ps(ACx + i * N + kk + 24);
            int j = Ap[i];
            for (; j < Ap[i + 1]-1; j+=2) {
              int aij1 = Ai[j] * N;
              int aij2 = Ai[j+1] * N;
              auto axV1 = _mm256_set1_ps(Ax[j]);
              auto axV2 = _mm256_set1_ps(Ax[j+1]);
              auto cxV11 = _mm256_loadu_ps(Cx + aij1 + kk);
              auto cxV12 = _mm256_loadu_ps(Cx + aij1 + kk + 8);
              auto cxV13 = _mm256_loadu_ps(Cx + aij1 + kk + 16);
              auto cxV14 = _mm256_loadu_ps(Cx + aij1 + kk + 24);
              auto cxV21 = _mm256_loadu_ps(Cx + aij2 + kk);
              auto cxV22 = _mm256_loadu_ps(Cx + aij2 + kk + 8);
              auto cxV23 = _mm256_loadu_ps(Cx + aij2 + kk + 16);
              auto cxV24 = _mm256_loadu_ps(Cx + aij2 + kk + 24);
              acxV1 = _mm256_fmadd_ps(axV1, cxV11, acxV1);
              acxV1 = _mm256_fmadd_ps(axV2, cxV21, acxV1);
              acxV2 = _mm256_fmadd_ps(axV1, cxV12, acxV2);
              acxV2 = _mm256_fmadd_ps(axV2, cxV22, acxV2);
              acxV3 = _mm256_fmadd_ps(axV1, cxV13, acxV3);
              acxV3 = _mm256_fmadd_ps(axV2, cxV23, acxV3);
              acxV4 = _mm256_fmadd_ps(axV1, cxV14, acxV4);
              acxV4 = _mm256_fmadd_ps(axV2, cxV24, acxV4);
            }
            for (; j < Ap[i + 1]; ++j) {
              int aij = Ai[j] * N;
              auto axv0 = _mm256_set1_ps(Ax[j]);
              auto cxV11 = _mm256_loadu_ps(Cx + aij + kk);
              auto cxV12 = _mm256_loadu_ps(Cx + aij + kk + 8);
              auto cxV13 = _mm256_loadu_ps(Cx + aij + kk + 16);
              auto cxV14 = _mm256_loadu_ps(Cx + aij + kk + 24);
              acxV1 = _mm256_fmadd_ps(axv0, cxV11, acxV1);
              acxV2 = _mm256_fmadd_ps(axv0, cxV12, acxV2);
              acxV3 = _mm256_fmadd_ps(axv0, cxV13, acxV3);
              acxV4 = _mm256_fmadd_ps(axv0, cxV14, acxV4);
            }
            _mm256_storeu_ps(ACx + i * N + kk, acxV1);
            _mm256_storeu_ps(ACx + i * N + kk + 8, acxV2);
            _mm256_storeu_ps(ACx + i * N + kk + 16, acxV3);
            _mm256_storeu_ps(ACx + i * N + kk + 24, acxV4);
          }
        }
        for (int k1 = kEndL1; k1 < kEndL2; k1++){
          int i = Partition[k1];
          for (int kk = 0; kk < N; kk += 32) {
            auto dxV1 = _mm256_loadu_ps(Dx + i * N + kk);
            auto dxV2 = _mm256_loadu_ps(Dx + i * N + kk + 8);
            auto dxV3 = _mm256_loadu_ps(Dx + i * N + kk + 16);
            auto dxV4 = _mm256_loadu_ps(Dx + i * N + kk + 24);
            int k = Ap[i];
            for (; k < Ap[i + 1]-1; k+=2) {
              int bij1 = Ai[k] * N;
              int bij2 = Ai[k+1] * N;
              auto bxV1 = _mm256_set1_ps(Ax[k]);
              auto bxV2 = _mm256_set1_ps(Ax[k+1]);
              auto acxV11 = _mm256_loadu_ps(ACx + bij1 + kk);
              auto acxV12 = _mm256_loadu_ps(ACx + bij1 + kk + 8);
              auto acxV13 = _mm256_loadu_ps(ACx + bij1 + kk + 16);
              auto acxV14 = _mm256_loadu_ps(ACx + bij1 + kk + 24);
              auto acxV21 = _mm256_loadu_ps(ACx + bij2 + kk);
              auto acxV22 = _mm256_loadu_ps(ACx + bij2 + kk + 8);
              auto acxV23 = _mm256_loadu_ps(ACx + bij2 + kk + 16);
              auto acxV24 = _mm256_loadu_ps(ACx + bij2 + kk + 24);
              dxV1 = _mm256_fmadd_ps(bxV1, acxV11, dxV1);
              dxV1 = _mm256_fmadd_ps(bxV2, acxV21, dxV1);
              dxV2 = _mm256_fmadd_ps(bxV1, acxV12, dxV2);
              dxV2 = _mm256_fmadd_ps(bxV2, acxV22, dxV2);
              dxV3 = _mm256_fmadd_ps(bxV1, acxV13, dxV3);
              dxV3 = _mm256_fmadd_ps(bxV2, acxV23, dxV3);
              dxV4 = _mm256_fmadd_ps(bxV1, acxV14, dxV4);
              dxV4 = _mm256_fmadd_ps(bxV2, acxV24, dxV4);
            }
            for (; k < Ap[i + 1]; ++k) {
              int bij = Ai[k] * N;
              auto bxv0 = _mm256_set1_ps(Ax[k]);
              auto cxV11 = _mm256_loadu_ps(ACx + bij + kk);
              auto cxV12 = _mm256_loadu_ps(ACx + bij + kk + 8);
              auto cxV13 = _mm256_loadu_ps(ACx + bij + kk + 16);
              auto cxV14 = _mm256_loadu_ps(ACx + bij + kk + 24);
              dxV1 = _mm256_fmadd_ps(bxv0, cxV11, dxV1);
              dxV2 = _mm256_fmadd_ps(bxv0, cxV12, dxV2);
              dxV3 = _mm256_fmadd_ps(bxv0, cxV13, dxV3);
              dxV4 = _mm256_fmadd_ps(bxv0, cxV14, dxV4);
            }
            _mm256_storeu_ps(Dx + i * N + kk, dxV1);
            _mm256_storeu_ps(Dx + i * N + kk + 8, dxV2);
            _mm256_storeu_ps(Dx + i * N + kk + 16, dxV3);
            _mm256_storeu_ps(Dx + i * N + kk + 24, dxV4);
          }
        }
        TaskFinished[j1] = true;
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
}

void spmm8CsrVectorizedUnrollJ4(int M, int N, const int *Ap, const int *Ai,
                                const double *Ax, const double *Cx, double *ACx,
                                int TileSize, int NThreads) {
  pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
  {
    pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
    for (int ii = 0; ii < M; ii += TileSize) {
      for (int i = ii; i < ii + TileSize && i < M; i++) {
        int j = Ap[i];
        for (; j < Ap[i + 1] - 3; j += 4) {
          auto a = _mm256_castsi256_pd(
              _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(Ax + j)));
          int aij = Ai[j] * N;
          int aij2 = Ai[j + 1] * N;
          int aij3 = Ai[j + 2] * N;
          int aij4 = Ai[j + 3] * N;
          auto axv0 = _mm256_permute4x64_pd(a, 0b00000000);
          auto axv1 = _mm256_permute4x64_pd(a, 0b01010101);
          auto axv2 = _mm256_permute4x64_pd(a, 0b10101010);
          auto axv3 = _mm256_permute4x64_pd(a, 0b11111111);
          for (int kk = 0; kk < N; kk += 8) {
            auto cxV11 = _mm256_loadu_pd(Cx + aij + kk);
            auto cxV12 = _mm256_loadu_pd(Cx + aij + kk + 4);
            auto cxV21 = _mm256_loadu_pd(Cx + aij2 + kk);
            auto cxV22 = _mm256_loadu_pd(Cx + aij2 + kk + 4);
            auto cxV31 = _mm256_loadu_pd(Cx + aij3 + kk);
            auto cxV32 = _mm256_loadu_pd(Cx + aij3 + kk + 4);
            auto cxV41 = _mm256_loadu_pd(Cx + aij4 + kk);
            auto cxV42 = _mm256_loadu_pd(Cx + aij4 + kk + 4);
            auto acxV = _mm256_loadu_pd(ACx + i * N + kk);
            auto acxV2 = _mm256_loadu_pd(ACx + i * N + kk + 4);
            acxV = _mm256_fmadd_pd(axv0, cxV11, acxV);
            acxV = _mm256_fmadd_pd(axv1, cxV21, acxV);
            acxV = _mm256_fmadd_pd(axv2, cxV31, acxV);
            acxV = _mm256_fmadd_pd(axv3, cxV41, acxV);
            acxV2 = _mm256_fmadd_pd(axv0, cxV12, acxV2);
            acxV2 = _mm256_fmadd_pd(axv1, cxV22, acxV2);
            acxV2 = _mm256_fmadd_pd(axv2, cxV32, acxV2);
            acxV2 = _mm256_fmadd_pd(axv3, cxV42, acxV2);
            _mm256_storeu_pd(ACx + i * N + kk, acxV);
            _mm256_storeu_pd(ACx + i * N + kk + 4, acxV2);
          }
        }
        for (; j < Ap[i + 1]; ++j) {
          int aij = Ai[j] * N;
          auto axv0 = _mm256_set1_pd(Ax[j]);
          for (int kk = 0; kk < N; kk += 8) {
            auto cxV11 = _mm256_loadu_pd(Cx + aij + kk);
            auto cxV12 = _mm256_loadu_pd(Cx + aij + kk + 4);
            auto acxV = _mm256_loadu_pd(ACx + i * N + kk);
            auto acxV2 = _mm256_loadu_pd(ACx + i * N + kk + 4);
            acxV = _mm256_fmadd_pd(axv0, cxV11, acxV);
            acxV2 = _mm256_fmadd_pd(axv0, cxV12, acxV2);
            _mm256_storeu_pd(ACx + i * N + kk, acxV);
            _mm256_storeu_pd(ACx + i * N + kk + 4, acxV2);
          }
        }
      }
    }
    pw_stop_instruments_loop(omp_get_thread_num());
  }
  pw_init_instruments;
}

void spmmCsrVectorized2_16(int M, int N, const int *Ap, const int *Ai,
                                 const double *Ax, const double *Cx,
                                 double *ACx, int TileSize, int NThreads) {
  pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
  {
    pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
    for (int ii = 0; ii < M; ii += TileSize) {
      for (int i = ii; i < ii + TileSize && i < M; i++) {
        for (int kk = 0; kk < N; kk += 16) {
        int j = Ap[i];
        auto acxV = _mm256_loadu_pd(ACx + i * N + kk);
        auto acxV2 = _mm256_loadu_pd(ACx + i * N + kk + 4);
        auto acxV3 = _mm256_loadu_pd(ACx + i * N + kk + 8);
        auto acxV4 = _mm256_loadu_pd(ACx + i * N + kk + 12);
        for (; j < Ap[i + 1] - 1; j += 2) {
          int aij = Ai[j] * N;
          int aij2 = Ai[j + 1] * N;
          auto axv0 = _mm256_set1_pd(Ax[j]);
          auto axv1 = _mm256_set1_pd(Ax[j + 1]);
            auto cxV11 = _mm256_loadu_pd(Cx + aij + kk);
            auto cxV12 = _mm256_loadu_pd(Cx + aij + kk + 4);
            auto cxV13 = _mm256_loadu_pd(Cx + aij + kk + 8);
            auto cxV14 = _mm256_loadu_pd(Cx + aij + kk + 12);
            auto cxV21 = _mm256_loadu_pd(Cx + aij2 + kk);
            auto cxV22 = _mm256_loadu_pd(Cx + aij2 + kk + 4);
            auto cxV23 = _mm256_loadu_pd(Cx + aij2 + kk + 8);
            auto cxV24 = _mm256_loadu_pd(Cx + aij2 + kk + 12);
            acxV = _mm256_fmadd_pd(axv0, cxV11, acxV);
            acxV = _mm256_fmadd_pd(axv1, cxV21, acxV);
            acxV2 = _mm256_fmadd_pd(axv0, cxV12, acxV2);
            acxV2 = _mm256_fmadd_pd(axv1, cxV22, acxV2);
            acxV3 = _mm256_fmadd_pd(axv0, cxV13, acxV3);
            acxV3 = _mm256_fmadd_pd(axv1, cxV23, acxV3);
            acxV4 = _mm256_fmadd_pd(axv0, cxV14, acxV4);
            acxV4 = _mm256_fmadd_pd(axv1, cxV24, acxV4);
          }
        for (; j < Ap[i + 1]; ++j) {
            int aij = Ai[j] * N;
            auto axv0 = _mm256_set1_pd(Ax[j]);
              auto cxV11 = _mm256_loadu_pd(Cx + aij + kk);
              auto cxV12 = _mm256_loadu_pd(Cx + aij + kk + 4);
              auto cxV13 = _mm256_loadu_pd(Cx + aij + kk + 8);
              auto cxV14 = _mm256_loadu_pd(Cx + aij + kk + 12);
              acxV = _mm256_fmadd_pd(axv0, cxV11, acxV);
              acxV2 = _mm256_fmadd_pd(axv0, cxV12, acxV2);
              acxV3 = _mm256_fmadd_pd(axv0, cxV13, acxV3);
              acxV4 = _mm256_fmadd_pd(axv0, cxV14, acxV4);
          }
          _mm256_storeu_pd(ACx + i * N + kk, acxV);
          _mm256_storeu_pd(ACx + i * N + kk + 4, acxV2);
          _mm256_storeu_pd(ACx + i * N + kk + 8, acxV3);
          _mm256_storeu_pd(ACx + i * N + kk + 12, acxV3);
        }
      }
    }
    pw_stop_instruments_loop(omp_get_thread_num());
  }
  pw_init_instruments;
}

void spmmCsrVectorized2_32SP(int M, int N, const int *Ap, const int *Ai,
                           const float *Ax, const float *Cx,
                           float *ACx, int TileSize, int NThreads) {
  pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
  {
    pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
    for (int ii = 0; ii < M; ii += TileSize) {
      for (int i = ii; i < ii + TileSize && i < M; i++) {
        for (int kk = 0; kk < N; kk += 32) {
          int j = Ap[i];
          auto acxV = _mm256_loadu_ps(ACx + i * N + kk);
          auto acxV2 = _mm256_loadu_ps(ACx + i * N + kk + 8);
          auto acxV3 = _mm256_loadu_ps(ACx + i * N + kk + 16);
          auto acxV4 = _mm256_loadu_ps(ACx + i * N + kk + 24);
          for (; j < Ap[i + 1] - 1; j += 2) {
              int aij = Ai[j] * N;
              int aij2 = Ai[j + 1] * N;
              auto axv0 = _mm256_set1_ps(Ax[j]);
              auto axv1 = _mm256_set1_ps(Ax[j + 1]);
              auto cxV11 = _mm256_loadu_ps(Cx + aij + kk);
              auto cxV12 = _mm256_loadu_ps(Cx + aij + kk + 8);
              auto cxV13 = _mm256_loadu_ps(Cx + aij + kk + 16);
              auto cxV14 = _mm256_loadu_ps(Cx + aij + kk + 24);
              auto cxV21 = _mm256_loadu_ps(Cx + aij2 + kk);
              auto cxV22 = _mm256_loadu_ps(Cx + aij2 + kk + 8);
              auto cxV23 = _mm256_loadu_ps(Cx + aij2 + kk + 16);
              auto cxV24 = _mm256_loadu_ps(Cx + aij2 + kk + 24);
              acxV = _mm256_fmadd_ps(axv0, cxV11, acxV);
              acxV = _mm256_fmadd_ps(axv1, cxV21, acxV);
              acxV2 = _mm256_fmadd_ps(axv0, cxV12, acxV2);
              acxV2 = _mm256_fmadd_ps(axv1, cxV22, acxV2);
              acxV3 = _mm256_fmadd_ps(axv0, cxV13, acxV3);
              acxV3 = _mm256_fmadd_ps(axv1, cxV23, acxV3);
              acxV4 = _mm256_fmadd_ps(axv0, cxV14, acxV4);
              acxV4 = _mm256_fmadd_ps(axv1, cxV24, acxV4);
          }
          for (; j < Ap[i + 1]; ++j) {
              int aij = Ai[j] * N;
              auto axv0 = _mm256_set1_ps(Ax[j]);
              auto cxV11 = _mm256_loadu_ps(Cx + aij + kk);
              auto cxV12 = _mm256_loadu_ps(Cx + aij + kk + 8);
              auto cxV13 = _mm256_loadu_ps(Cx + aij + kk + 16);
              auto cxV14 = _mm256_loadu_ps(Cx + aij + kk + 24);
              acxV = _mm256_fmadd_ps(axv0, cxV11, acxV);
              acxV2 = _mm256_fmadd_ps(axv0, cxV12, acxV2);
              acxV3 = _mm256_fmadd_ps(axv0, cxV13, acxV3);
              acxV4 = _mm256_fmadd_ps(axv0, cxV14, acxV4);
          }
          _mm256_storeu_ps(ACx + i * N + kk, acxV);
          _mm256_storeu_ps(ACx + i * N + kk + 8, acxV2);
          _mm256_storeu_ps(ACx + i * N + kk + 16, acxV3);
          _mm256_storeu_ps(ACx + i * N + kk + 24, acxV3);
        }
      }
    }
    pw_stop_instruments_loop(omp_get_thread_num());
  }
  pw_init_instruments;
}

void spmm16CsrVectorized(int M, int N, const int *Ap, const int *Ai,
                         const double *Ax, const double *Cx, double *ACx,
                         int TileSize, int NThreads) {
  pw_init_instruments;
#pragma omp parallel num_threads(NThreads)
  {
    pw_start_instruments_loop(omp_get_thread_num());
#pragma omp for
    for (int ii = 0; ii < M; ii += TileSize) {
      for (int i = ii; i < ii + TileSize && i < M; i++) {
        int j = Ap[i];
        for (; j < Ap[i + 1]; j += 1) {
          int aij = Ai[j] * N;
          auto axv0 = _mm256_set1_pd(Ax[j]);
          for (int kk = 0; kk < N; kk += 16) {
            auto cxV11 = _mm256_loadu_pd(Cx + aij + kk);
            auto cxV12 = _mm256_loadu_pd(Cx + aij + kk + 4);
            auto cxV13 = _mm256_loadu_pd(Cx + aij + kk + 8);
            auto cxV14 = _mm256_loadu_pd(Cx + aij + kk + 12);
            auto acxV = _mm256_loadu_pd(ACx + i * N + kk);
            auto acxV2 = _mm256_loadu_pd(ACx + i * N + kk + 4);
            auto acxV3 = _mm256_loadu_pd(ACx + i * N + kk + 8);
            auto acxV4 = _mm256_loadu_pd(ACx + i * N + kk + 12);
            acxV = _mm256_fmadd_pd(axv0, cxV11, acxV);
            acxV2 = _mm256_fmadd_pd(axv0, cxV12, acxV2);
            acxV3 = _mm256_fmadd_pd(axv0, cxV13, acxV3);
            acxV4 = _mm256_fmadd_pd(axv0, cxV14, acxV4);
            _mm256_storeu_pd(ACx + i * N + kk, acxV);
            _mm256_storeu_pd(ACx + i * N + kk + 4, acxV2);
            _mm256_storeu_pd(ACx + i * N + kk + 8, acxV3);
            _mm256_storeu_pd(ACx + i * N + kk + 12, acxV3);
          }
        }
      }
    }
    pw_stop_instruments_loop(omp_get_thread_num());
  }
  pw_init_instruments;
}

void spmmCsrSpmmCscFusedColoredAvx256(int M, int N, int K, int L, const int *Ap,
                                const int *Ai, const double *Ax, const int *Bp,
                                const int *Bi, const double *Bx,
                                const double *Cx, double *Dx, double *ACx,
                                int LevelNo, const int *LevelPtr, const int *Id,
                                int TileSize, int NThreads) {
  int lastTileSize = M % TileSize;
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
        for (int ii = 0; ii < TileSize; ++ii) {
          auto ipii = i + ii;
          // first SpMM
          for (int k = 0; k < N; k += 16) {
            auto acxv0 = _mm256_setzero_pd();
            auto acxv1 = _mm256_setzero_pd();
            auto acxv2 = _mm256_setzero_pd();
            auto acxv3 = _mm256_setzero_pd();
            for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
              int aij = Ai[j] * N;
//              for (int kk = 0; kk < N; ++kk) {
//                tAcxi[kk] += Ax[j] * Cx[aij + kk];
//              }
              auto axv0 = _mm256_set1_pd(Ax[j]);
              auto bxv0 = _mm256_loadu_pd(Cx + aij + k);
              auto bxv1 = _mm256_loadu_pd(Cx + aij + k + 4);
              auto bxv2 = _mm256_loadu_pd(Cx + aij + k + 8);
              auto bxv3 = _mm256_loadu_pd(Cx + aij + k + 12);
              acxv0 = _mm256_fmadd_pd(axv0, bxv0, acxv0);
              acxv1 = _mm256_fmadd_pd(axv0, bxv1, acxv1);
              acxv2 = _mm256_fmadd_pd(axv0, bxv2, acxv2);
              acxv3 = _mm256_fmadd_pd(axv0, bxv3, acxv3);
            }
            // second SpMM CSC
            for (int j = Bp[ipii]; j < Bp[ipii + 1];
                 j++) { // for each column of B
//              for (int kk = 0; kk < N; ++kk) {
//                int bij = Bi[k] * N;
//                Dx[bij + kk] += Bx[k] * tAcxi[kk];
//              }
              auto bxv0 = _mm256_set1_pd(Bx[j]);
              auto dxv0 = _mm256_loadu_pd(Dx + Bi[j] * N + k);
              auto dxv1 = _mm256_loadu_pd(Dx + Bi[j] * N + k + 4);
              auto dxv2 = _mm256_loadu_pd(Dx + Bi[j] * N + k + 8);
              auto dxv3 = _mm256_loadu_pd(Dx + Bi[j] * N + k + 12);
              dxv0 = _mm256_fmadd_pd(bxv0, acxv0, dxv0);
              dxv1 = _mm256_fmadd_pd(bxv0, acxv1, dxv1);
              dxv2 = _mm256_fmadd_pd(bxv0, acxv2, dxv2);
              dxv3 = _mm256_fmadd_pd(bxv0, acxv3, dxv3);
              _mm256_storeu_pd(Dx + Bi[j] * N + k, dxv0);
              _mm256_storeu_pd(Dx + Bi[j] * N + k + 4, dxv1);
              _mm256_storeu_pd(Dx + Bi[j] * N + k + 8, dxv2);
              _mm256_storeu_pd(Dx + Bi[j] * N + k + 12, dxv3);
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
  int i = M - lastTileSize;
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
    for (int k = Bp[ipii]; k < Bp[ipii + 1]; k++) { // for each column of B
      for (int kk = 0; kk < N; ++kk) {
        int bij = Bi[k] * N;
        Dx[bij + kk] += Bx[k] * ACx[ipii * N + kk];
      }
    }
  }
}

void spmmCsrSpmmCscFusedColoredAvx256Packed(int M, int N, int K, int L, const int *Ap,
                                      const int *Ai, const double *Ax, const int *Bp,
                                      const int *Bi, const double *Bx,
                                      const double *Cx, double *Dx, double *ACx,
                                      int LevelNo, const int *LevelPtr, const int *Id,
                                      int TileSize, int NThreads) {
  int lastTileSize = M % TileSize;
  int packedN = 16;
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
        for (int k = 0; k < N; k += packedN) {
          int pckId = k/packedN;
          double* curPack = Dx + pckId * M * packedN;
          for (int ii = 0; ii < TileSize; ++ii) {
            auto ipii = i + ii;
            // first SpMM
            auto acxv0 = _mm256_setzero_pd();
            auto acxv1 = _mm256_setzero_pd();
            auto acxv2 = _mm256_setzero_pd();
            auto acxv3 = _mm256_setzero_pd();
            for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
              int aij = Ai[j] * N;
              //              for (int kk = 0; kk < N; ++kk) {
              //                tAcxi[kk] += Ax[j] * Cx[aij + kk];
              //              }
              auto axv0 = _mm256_set1_pd(Ax[j]);
              auto bxv0 = _mm256_loadu_pd(Cx + aij + k);
              auto bxv1 = _mm256_loadu_pd(Cx + aij + k + 4);
              auto bxv2 = _mm256_loadu_pd(Cx + aij + k + 8);
              auto bxv3 = _mm256_loadu_pd(Cx + aij + k + 12);
              acxv0 = _mm256_fmadd_pd(axv0, bxv0, acxv0);
              acxv1 = _mm256_fmadd_pd(axv0, bxv1, acxv1);
              acxv2 = _mm256_fmadd_pd(axv0, bxv2, acxv2);
              acxv3 = _mm256_fmadd_pd(axv0, bxv3, acxv3);
            }
            // second SpMM CSC
            for (int j = Bp[ipii]; j < Bp[ipii + 1];
                 j++) {
              int dij = Bi[j] * packedN;
              auto bxv0 = _mm256_set1_pd(Bx[j]);
              auto dxv0 = _mm256_loadu_pd(curPack + dij);
              auto dxv1 = _mm256_loadu_pd(curPack + dij + 4);
              auto dxv2 = _mm256_loadu_pd(curPack + dij + 8);
              auto dxv3 = _mm256_loadu_pd(curPack + dij + 12);
              dxv0 = _mm256_fmadd_pd(bxv0, acxv0, dxv0);
              dxv1 = _mm256_fmadd_pd(bxv0, acxv1, dxv1);
              dxv2 = _mm256_fmadd_pd(bxv0, acxv2, dxv2);
              dxv3 = _mm256_fmadd_pd(bxv0, acxv3, dxv3);
              _mm256_storeu_pd(curPack + dij , dxv0);
              _mm256_storeu_pd(curPack + dij + 4, dxv1);
              _mm256_storeu_pd(curPack + dij + 8, dxv2);
              _mm256_storeu_pd(curPack + dij + 12, dxv3);
            }
          }
        }
      }
      pw_stop_instruments_loop(omp_get_thread_num());
    }
  }
//#pragma omp parallel for num_threads(NThreads)
//  for (int ii = 0; ii < M; ++ii) {
//  for (int k = 0; k < N; k += packedN){
//      for (int j = 0; j < packedN; ++j) {
//        Dx[ii * N + k + j] = PackedDx[ii * packedN + j];
//      }
//    }
//  }
  int i = M - lastTileSize;
  for (int k = 0; k < N; k +=packedN) {
    int pckId = k /packedN;
    double* curPack = Dx + pckId * M * packedN;
    for (int ii = 0; ii < lastTileSize; ++ii) {
      auto ipii = i + ii;
      // first SpMM
        for (int j = Ap[ipii]; j < Ap[ipii + 1]; j++) {
          int aij = Ai[j] * N;
          for (int kk = 0; kk < packedN; kk++) {
            ACx[ipii * N + k + kk] += Ax[j] * Cx[aij + k + kk];
          }
        }
      // second SpMM CSC
      for (int j = Bp[ipii]; j < Bp[ipii + 1]; j++) {
          for (int kk = 0; kk < packedN; kk++){
            int bij = Bi[j] * packedN;
            curPack[bij + kk] += Bx[j] * ACx[ipii * N + k + kk];
          }// for each column of B
        }
      }
  }
}

#endif

} // namespace sparse
} // namespace swiftware
//
// Created by salehm32 on 21/02/24.
//

#ifndef SPARSE_FUSION_SPMM_SPMM_VECTORIZED_H
#define SPARSE_FUSION_SPMM_SPMM_VECTORIZED_H
#include <immintrin.h>
namespace swiftware {
namespace sparse {

#ifdef __AVX512F__
void spmmCsrSpmmCsrFusedVectorizedKTiled8Avx512(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads);

void spmmCsrSpmmCsrFusedVectorized8Avx512(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads);

//void spmmCsrSpmmCsrFusedVectorized64Avx512(
//    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
//    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
//    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
//    const int *ParPtr, const int *Partition, const int *ParType, int NThreads);

inline void vectorCrossProduct32Avx512SP(float Ax, int Ai,
                                         const float *B, float *C, int N,
                                         __m512 &dxV1,  __m512 &dxV2);

inline void vectorCrossProduct32Avx512(double Ax, int Ai, const double *B,
                                       double *C, int N, int I);

inline void vectorCrossProduct2_32Avx512SP(const float* Ax, const int* Ai,
                                           const float *B,float *C, int N,
                                           __m512 &dxV1,  __m512 &dxV2);

void spmmCsrSpmmCsrFusedVectorized2_32Avx512(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads);

void spmmCsrSpmmCsrFusedVectorized2_32Avx512SP(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const float *Ax,
    const int *Bp, const int *Bi, const float *Bx, const float *Cx,
    float *Dx, float *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads);

void spmmCsrVectorized2_32Avx512(int M, int N, const int *Ap, const int *Ai,
                                 const double *Ax, const double *Cx,
                                 double *ACx, int TileSize, int NThreads);

void spmmCsrVectorized2_32Avx512SP(int M, int N, const int *Ap, const int *Ai,
                                 const float *Ax, const float *Cx,
                                 float *ACx, int TileSize, int NThreads);

void spmmCsrSpmmCscFusedColoredAvx512(int M, int N, int K, int L, const int *Ap,
                                      const int *Ai, const double *Ax,
                                      const int *Bp, const int *Bi,
                                      const double *Bx, const double *Cx,
                                      double *Dx, double *ACx, int LevelNo,
                                      const int *LevelPtr, const int *Id,
                                      int TileSize, int NThreads);

inline void vectorCrossProduct8Avx512(double Ax, int Ai, const double *B,
                                      __m512d &Xv, int N, int I);

inline void vectorCrossProduct4_8Avx512(const double *Ax, const int *Ai,
                                        const double *B, __m512d &Xv, int N,
                                        int I);

inline void vectorCrossProduct64Avx512(double Ax, int Ai, const double *B,
                                       double *C, int N, int I);

inline void vectorCrossProduct32Avx512(double Ax, int Ai,
                                       const double *B, double *C, int N,
                                       __m512d &dxV1,  __m512d &dxV2,
                                       __m512d &dxV3,  __m512d &dxV4);

inline void vectorCrossProduct2_32Avx512(const double* Ax, const int* Ai,
                                         const double *B,double *C, int N,
                                         __m512d &dxV1,  __m512d &dxV2,
                                         __m512d &dxV3,  __m512d &dxV4);

inline void vectorCrossProduct128Avx512(double Ax, int Ai, const double *B,
                                        double *C, int N, int I);
#endif
#if defined(__AVX2__)

void spmmCsrSpmmCsrFusedKTiled8Vectorized(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads);

void spmmCsrSpmmCsrFusedVectorized2_16(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads);

void spmmCsrSpmmCsrFusedVectorized2_32SP(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const float *Ax,
    const int *Bp, const int *Bi, const float *Bx, const float *Cx,
    float *Dx, float *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads);

void spmmCsrSpmmCsrOneSparseMatrixFusedVectorized2_32SP(
    int M, int N, int K, int L,  const int *__restrict__ Ap, const int *__restrict__ Ai, const float *__restrict__ Ax,
    const float *Cx, float *Dx, float *__restrict__ ACx,
    int LevelNo, const int *LevelPtr,const int *ParPtr,
    const int* Partition, const int *MixPtr, int NThreads);

void spmmCsrSpmmCsrFusedVectorized2_8(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr,
    const int *ParPtr, const int *Partition, const int *ParType, int NThreads);

void spmm8CsrVectorizedUnrollJ4(int M, int N, const int *Ap, const int *Ai,
                                const double *Ax, const double *Cx, double *ACx,
                                int TileSize, int NThreads);

void spmmCsrVectorized2_16(int M, int N, const int *Ap, const int *Ai,
                                 const double *Ax, const double *Cx,
                                 double *ACx, int TileSize, int NThreads);

void spmmCsrVectorized2_32SP(int M, int N, const int *Ap, const int *Ai,
                             const float *Ax, const float *Cx,
                             float *ACx, int TileSize, int NThreads);

void spmm16CsrVectorized(int M, int N, const int *Ap, const int *Ai,
                         const double *Ax, const double *Cx, double *ACx,
                         int TileSize, int NThreads);

void spmmCsrSpmmCscFusedColoredAvx256(int M, int N, int K, int L, const int *Ap,
                                      const int *Ai, const double *Ax,
                                      const int *Bp, const int *Bi,
                                      const double *Bx, const double *Cx,
                                      double *Dx, double *ACx, int LevelNo,
                                      const int *LevelPtr, const int *Id,
                                      int TileSize, int NThreads);
void spmmCsrSpmmCscFusedColoredAvx256Packed(
    int M, int N, int K, int L, const int *Ap, const int *Ai, const double *Ax,
    const int *Bp, const int *Bi, const double *Bx, const double *Cx,
    double *Dx, double *ACx, int LevelNo, const int *LevelPtr, const int *Id,
    int TileSize, int NThreads);

#endif

} // namespace sparse
} // namespace swiftware
#endif // SPARSE_FUSION_SPMM_SPMM_VECTORIZED_H

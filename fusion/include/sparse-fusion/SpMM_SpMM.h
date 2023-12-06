//
// Created by kazem on 08/05/23.
//

#ifndef SPARSE_FUSION_SPMM_SPMM_H
#define SPARSE_FUSION_SPMM_SPMM_H

namespace swiftware {
namespace sparse {

/// C = A*B, where A is sparse CSR MxK and B (K x N) and C (MxN) are Dense
/// \param M
/// \param N
/// \param K
/// \param Ap : row pointer
/// \param Ai : column index
/// \param Ax : values
/// \param Bx : values
/// \param Cx : values
void spmmCsrSequential(int M, int N, int K,
                       const int *Ap, const int *Ai, const double *Ax,
                       const double *Bx, double *Cx);
void spmmCsrParallel(int M, int N, int K,
                     const int *Ap, const int *Ai, const double *Ax,
                     const double *Bx, double *Cx, int NThreads);
void spmmCsrInnerProductParallel(int M, int N, int K,
                                 const int *Ap, const int *Ai, const double *Ax,
                                 const double *Bx, double *Cx, int NThreads);
void spmmCsrInnerProductTiledCParallel(int M, int N, int K,
                                       const int *Ap, const int *Ai, const double *Ax,
                                       const double *Bx, double *Cx, int NThreads
                                       ,int MTile, int NTile);
void spmmCsrParallelTiled(int M, int N, int K,
                          const int *Ap, const int *Ai, const double *Ax,
                          const double *Bx, double *Cx, int NThreads,
                          int MTile, int NTile);


    /// D = B*A*C where A (MxK) and B (LxM) are sparse and C (KxN) and D (LxN) are dense
/// \param M
/// \param N
/// \param K
/// \param L
/// \param Ap
/// \param Ai
/// \param Ax
/// \param Bx
/// \param Cx
/// \param Dx
/// \param ACx temporary values of AxC
/// \param LevelNo
/// \param LevelPtr
/// \param ParPtr
/// \param Partition
/// \param ParType
void spmmCsrSpmmCsrFused(int M, int N, int K,
                         int L,
                         const int *Ap, const int *Ai, const double *Ax,
                         const int *Bp, const int *Bi,const double *Bx,
                         const double *Cx,
                         double *Dx,
                         double *ACx,
                         int LevelNo, const int *LevelPtr, const int *ParPtr,
                         const int *Partition, const int *ParType,
                         int NThreads);
void spmmCsrSpmmCsrInnerProductFused(int M, int N, int K,
                         int L,
                         const int *Ap, const int *Ai, const double *Ax,
                         const int *Bp, const int *Bi,const double *Bx,
                         const double *Cx,
                         double *Dx,
                         double *ACx,
                         int LevelNo, const int *LevelPtr, const int *ParPtr,
                         const int *Partition, const int *ParType,
                         int NThreads);
void spmmCsrSpmmCsrMixedScheduleFused(int M, int N, int K,
                                     int L,
                                     const int *Ap, const int *Ai, const double *Ax,
                                     const int *Bp, const int *Bi,const double *Bx,
                                     const double *Cx,
                                     double *Dx,
                                     double *ACx,
                                     int LevelNo, const int *LevelPtr, const int *ParPtr,
                                     const int *Partition, const int *ParType,
                                     int NThreads);
void spmmCsrSpmmCsrSeparatedFused(int M, int N, int K, int L,
                                  const int *Ap, const int *Ai, const double *Ax,
                                  const int *Bp, const int *Bi,const double *Bx,
                                  const double *Cx,
                                  double *Dx,
                                  double *ACx,
                                  int LevelNo, const int *LevelPtr, const int *ParPtr,
                                  const int *Partition, const int *ParType,
                                  const int *MixPtr,
                                  int NThreads) ;
void spmmCsrSpmmCsrTiledFused(int M, int N, int K, int L,
                              const int *Ap, const int *Ai, const double *Ax,
                              const int *Bp, const int *Bi,const double *Bx,
                              const double *Cx,
                              double *Dx,
                              double *ACx,
                              int LevelNo, const int *LevelPtr, const int *ParPtr,
                              const int *Partition, const int *ParType,
                              int NThreads, int MTile, int NTile, double *Ws);
void spmmCsrSpmmCsrTiledFusedRedundantBanded(int M, int N, int K, int L,
                                       const int *Ap, const int *Ai, const double *Ax,
                                       const int *Bp, const int *Bi,const double *Bx,
                                       const double *Cx,
                                       double *Dx,
                                       double *ACx,
                                       int LevelNo, const int *LevelPtr, const int *ParPtr,
                                       const int *Partition, const int *ParType, const int*MixPtr,
                                       int NThreads, int MTile, int NTile, double *Ws);
void spmmCsrSpmmCsrTiledFusedRedundantGeneral(int M, int N, int K, int L,
                                              const int *Ap, const int *Ai, const double *Ax,
                                              const int *Bp, const int *Bi,const double *Bx,
                                              const double *Cx,
                                              double *Dx,
                                              double *ACx,
                                              int LevelNo, const int *LevelPtr, const int *ParPtr,
                                              const int *Partition, const int *ParType, const int*MixPtr,
                                              int NThreads, int MTile, int NTile, double *Ws);


void spmmCsrSpmmCscFused(int M, int N, int K, int L,
                         const int *Ap, const int *Ai, const double *Ax,
                         const int *Bp, const int *Bi,const double *Bx,
                         const double *Cx,
                         double *Dx,
                         double *ACx,
                         int LevelNo, const int *LevelPtr, const int *ParPtr,
                         const int *Partition, const int *ParType,
                         int NThreads);


void spmmCsrSpmmCscFusedAffine(int M, int N, int K, int L,
                               const int *Ap, const int *Ai, const double *Ax,
                               const int *Bp, const int *Bi,const double *Bx,
                               const double *Cx,
                               double *Dx,
                               double *ACx,
                               int NThreads);

void spmmCsrSpmmCscFusedColored(int M, int N, int K, int L,
                                const int *Ap, const int *Ai, const double *Ax,
                                const int *Bp, const int *Bi,const double *Bx,
                                const double *Cx,
                                double *Dx,
                                double *ACx,
                                int LevelNo, const int *LevelPtr,
                                const int *Id, const int *TileSizes, int MaxTileSize,
                                int NThreads);

void spmmCsrSpmmCscFusedColoredWithScheduledKTiles(int M, int N, int K, int L,
                                const int *Ap, const int *Ai, const double *Ax,
                                const int *Bp, const int *Bi,const double *Bx,
                                const double *Cx,
                                double *Dx,
                                double *ACx,
                                int LevelNo, const int *LevelPtr,
                                const int *Id, const int *TileSizes, int MaxTileSize,
                                int KTileSize, int NThreads);

void spmmCsrSpmmCscFusedColoredWithReplicatedKTiles(int M, int N, int K, int L,
                                                   const int *Ap, const int *Ai, const double *Ax,
                                                   const int *Bp, const int *Bi,const double *Bx,
                                                   const double *Cx,
                                                   double *Dx,
                                                   double *ACx,
                                                   int LevelNo, const int *LevelPtr,
                                                   const int *Id, const int *TileSizes, int MaxTileSize,
                                                   int KTileSize, int NThreads);

} // namespace sparse
} // namespace swiftware
#endif // SPARSE_FUSION_SPMM_SPMM_H

//
// Created by salehm32 on 30/01/25.
//

#ifndef SPARSE_FUSION_GEMM_H
#define SPARSE_FUSION_GEMM_H
namespace swiftware {
namespace dense {

/// C = A*B, where A is sparse CSR MxK and B (K x N) and C (MxN) are Dense
/// \param M
/// \param N
/// \param K
/// \param Ap : row pointer
/// \param Ai : column index
/// \param Ax : values
/// \param Bx : values
/// \param Cx : values
template <class T>
void geMMParallel(int M, int N, int K, const T *Ax, const T *Bx, T *Cx, int NumThreads);
} // namespace dense
} // namespace swiftware
#endif // SPARSE_FUSION_GEMM_H

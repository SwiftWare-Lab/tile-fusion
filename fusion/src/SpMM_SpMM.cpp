//
// Created by kazem on 02/05/23.
//

namespace sym_lib {

 void spmmCSRSpmmCSRFused(int N, int M, const int *Ap,
                                       const int *Ai,
                                       const double *Ax, double *Bx,
                                       double *ABx,
                                       double *Cx,
                                       int LevelNo,
                                       const int *LevelPtr,
                                       const int *ParPtr,
                                       const int *Partition,
                                       const int *ParType,
                                       double *Tmp) {
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel
   {
#pragma omp  for schedule(auto) nowait
    for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
     for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
      int i = Partition[k1];
      int t = ParType[k1];
      if (t == 0) {
       for (int j = Ap[i]; j < Ap[i + 1]; j++) {
        int aij = Ai[j] * M;
        for (int kk = 0; kk < M; ++kk) {
         ABx[i * M + kk] += Ax[j] * Bx[aij + kk];
        }
       }
      } else {
       for (int k = Ap[i]; k < Ap[i + 1]; k++) {
        int aij = Ai[k] * M;
        for (int kk = 0; kk < M; ++kk) {
         Cx[i * M + kk] += Ax[k] * ABx[aij + kk];
        }
       }
      }
     }
    }
   }
  }
 }

} // namespace sym_lib
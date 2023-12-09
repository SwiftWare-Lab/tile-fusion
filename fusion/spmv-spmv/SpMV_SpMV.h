//
// Created by salehm32 on 08/12/23.
//

#ifndef SPARSE_FUSION_SPMV_SPMV_UTILS_H
#define SPARSE_FUSION_SPMV_SPMV_UTILS_H
void spMVCsrSequential(int M, int K,
                       const int *Ap, const int *Ai, const double *Ax,
                       const double *Bx, double *Cx);
void spMVCsrParallel(int M, int K,
                     const int *Ap, const int *Ai, const double *Ax,
                     const double *Bx, double *Cx, int NumThreads);


void spMVCsrSequential(int M, int K,
              const int *Ap, const int *Ai, const double *Ax,
              const double *Bx, double *Cx){
  for(int i = 0; i < M; i++){
    for(int j = Ap[i]; j < Ap[i+1]; j++){
      Cx[i] += Ax[j] * Bx[Ai[j]];
    }
  }
}

void spMVCsrParallel(int M, int K,
            const int *Ap, const int *Ai, const double *Ax,
            const double *Bx, double *Cx, int NumThreads){
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

#endif // SPARSE_FUSION_SPMV_SPMV_UTILS_H

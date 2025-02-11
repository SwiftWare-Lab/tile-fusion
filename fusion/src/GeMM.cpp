//
// Created by salehm32 on 30/01/25.
//

#ifdef PROF_WITH_PAPI
#include "papi_wrapper.h"
#else
#define pw_init_instruments
#define pw_start_instruments_loop(th)
#define pw_stop_instruments_loop(th)
#endif
namespace swiftware {
namespace dense {
template <class T>
void geMMParallel(int M, int N, int K, const T *Ax, const T *Bx, T *Cx, int NumThreads){
#pragma omp parallel for num_threads(NumThreads)
  for (int i = 0; i < M; i++){
    for (int k = 0; k < K; k++){
      for (int j = 0; j < N; j++){
        Cx[i*N + j] += Ax[i*K + k] * Bx[k*N + j];
      }
    }
  }
}

template void geMMParallel<float>(int M, int N, int K, const float *Ax, const float *Bx, float *Cx, int NumThreads);
} // namespace dense
} // namespace swiftware
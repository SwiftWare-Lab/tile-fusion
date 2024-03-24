//
// Created by kazem on 21/03/24.
//

#ifndef SPARSE_FUSION_GEMM_SPMM_CODEGEN_H
#define SPARSE_FUSION_GEMM_SPMM_CODEGEN_H

#include "GCN_Def.h"
#include "omp.h"

namespace swiftware{

// A = B x C x D
// A is Dense IxL
// B is sparse IxJ
// C is dense JxK
// D is dense KxL
template <class mytype>
int compute_LNR(int Am, int An, mytype *A,
            int *Bp, int *Bi, mytype* Bx,
            int Cm, int Cn, mytype *C,
            int Dm, int Dn, mytype *D,int NumThreads) {
  int A1_dimension = (int)(Am);
  int A2_dimension = (int)(An);
  mytype* A_vals = A;
  int B1_dimension = (int)(Am);
  int* B2_pos = (int*)(Bp);
  int*  B2_crd = (int*)(Bi);
  mytype* B_vals = (Bx);
  int C1_dimension = Cm;
  int C2_dimension = Cn;
  mytype* C_vals = C;
  int D1_dimension = Dm;
  int D2_dimension = Dn;
  mytype*  D_vals = D;

#pragma omp parallel for schedule(static) num_threads(NumThreads)
  for (int32_t pA = 0; pA < (A1_dimension * A2_dimension); pA++) {
    A_vals[pA] = 0.0;
  }

  mytype* tA_all = 0;
  tA_all = (mytype*)malloc(sizeof(float) * (D1_dimension *
                                              NumThreads));

#pragma omp parallel for schedule(runtime) num_threads(NumThreads)
  for (int32_t i = 0; i < B1_dimension; i++) {
    mytype* tA = tA_all + D1_dimension * omp_get_thread_num();
    for (int32_t ptA = 0; ptA < D1_dimension; ptA++) {
      tA[ptA] = 0.0;
    }
    for (int32_t jB = B2_pos[i]; jB < B2_pos[(i + 1)]; jB++) {
      int32_t j = B2_crd[jB];
      for (int32_t k = 0; k < D1_dimension; k++) {
        int32_t kC = j * C2_dimension + k;
        tA[k] = tA[k] + B_vals[jB] * C_vals[kC];
      }
    }
    for (int32_t k = 0; k < D1_dimension; k++) {
      for (int32_t l = 0; l < D2_dimension; l++) {
        int32_t lA = i * A2_dimension + l;
        int32_t lD = k * D2_dimension + l;
        A_vals[lA] = A_vals[lA] + tA[k] * D_vals[lD];
      }
    }
  }

  free(tA_all);
  return 0;
}


template <class mytype>
int compute_TACO(int Am, int An, mytype *A,
                 int *Bp, int *Bi, mytype* Bx,
                int Cm, int Cn, mytype *C,
                int Dm, int Dn, mytype *D, int NumThreads) {
  int A1_dimension = (int)(Am);
  int A2_dimension = (int)(An);
  mytype* A_vals = A;
  int B1_dimension = (int)(Am);
  int* B2_pos = (int*)(Bp);
  int*  B2_crd = (int*)(Bi);
  mytype* B_vals = (Bx);
  int C1_dimension = Cm;
  int C2_dimension = Cn;
  mytype* C_vals = C;
  int D1_dimension = Dm;
  int D2_dimension = Dn;
  mytype*  D_vals = D;

#pragma omp parallel for schedule(static) num_threads(NumThreads)
  for (int32_t pA = 0; pA < (A1_dimension * A2_dimension); pA++) {
    A_vals[pA] = 0.0;
  }

#pragma omp parallel for schedule(runtime) num_threads(NumThreads)
  for (int32_t i = 0; i < B1_dimension; i++) {
    for (int32_t jB = B2_pos[i]; jB < B2_pos[(i + 1)]; jB++) {
      int32_t j = B2_crd[jB];
      for (int32_t k = 0; k < D1_dimension; k++) {
        int32_t kC = j * C2_dimension + k;
        for (int32_t l = 0; l < D2_dimension; l++) {
          int32_t lA = i * A2_dimension + l;
          int32_t lD = k * D2_dimension + l;
          A_vals[lA] = A_vals[lA] + (B_vals[jB] * C_vals[kC]) * D_vals[lD];
        }
      }
    }
  }
  return 0;
}



}


#endif // SPARSE_FUSION_GEMM_SPMM_CODEGEN_H

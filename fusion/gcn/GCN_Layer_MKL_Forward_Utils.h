//
// Created by salehm32 on 29/09/23.
//

#include "SWTensorBench.h"
#include <cstring>
#include <math.h>
#include <omp.h>

#ifdef MKL
#include <mkl.h>
#endif

#ifdef BLAS_VEN
#include "cblas.h"
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifndef SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H
#define SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H

using namespace swiftware::benchmark;

#ifdef MKL
void forwardForOneLayerWithMKLGeMMAndMKLSpMM(int NumOfNodes,
                                             sparse_matrix_t AdjMatrix,
                                             double *Features, int FeatDim,
                                             double *Weight, int OutDim,
                                             double *Output, double* IntermediateResult) {
  matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, NumOfNodes, OutDim,
              FeatDim, 1., Features, FeatDim, Weight, FeatDim, 0.,
              IntermediateResult,
              OutDim);
  mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, AdjMatrix, d,
                  SPARSE_LAYOUT_ROW_MAJOR, IntermediateResult, OutDim, OutDim, 0, Output,
                  OutDim);
}


void forwardForOneLayerWithMKLGeMMAndMKLSpMMSP(int NumOfNodes,
                                               sparse_matrix_t AdjMatrix,
                                               float *Features, int FeatDim,
                                               float *Weight, int OutDim,
                                               float *Output, float *IntermediateResult) {
  matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_GENERAL;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, NumOfNodes, OutDim,
              FeatDim, 1., Features, FeatDim, Weight, FeatDim, 0.,
              IntermediateResult,
              OutDim);
  mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, AdjMatrix, d,
                  SPARSE_LAYOUT_ROW_MAJOR, IntermediateResult, OutDim, OutDim, 0, Output,
                  OutDim);
}
#endif



#endif // SPARSE_FUSION_GCN_LAYER_MKL_DEMO_H

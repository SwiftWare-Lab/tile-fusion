//
// Created by mehdi on 6/16/24.
//
#include "aggregation/def.h"
#include "Timer.h"
#include <cusparse.h>
#include <iostream>

#define WARMUP_NUM_CUDA 20
#define EXE_NUM_CUDA 200


void cusparseCSRSpMM(const sym_lib::CSR *mat1, const float *B,const int DenseRows, const int DenseCols, float* C, int algid, float& time) {
  swiftware::benchmark::Timer timer;
  const int n_rows_mat1 = mat1->m;
  const int n_cols_mat1 = mat1->n;
  const int n_rows_mat2 = DenseRows;
  const int n_cols_mat2 = DenseCols;
  const int m = n_rows_mat1;
  const int k = n_cols_mat1;
  const int n = n_cols_mat2;
  using scalar_t = float;
  scalar_t alpha = 1.0;
  scalar_t beta = 0.0;
  int nnzA = mat1->nnz;
  int *rowindA_csr = mat1->p;
  int *colindA = mat1->i;
  float* valuesA = new float[nnzA]; // TODO: Delete this allocated Arr
  for (int i=0; i < nnzA; i++){
    valuesA[i] = (float)mat1->x[i];
  }
  // cuda handle
  cusparseHandle_t cusparse_handle = 0;
  cusparseCreate(&cusparse_handle);
#if CUDART_VERSION < 11000
  int ldb = n;
  int ldc = m;
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  // kernel
  for (int i = 0; i < WARMUP_NUM_CUDA; i++) {
    cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_TRANSPOSE, m, n, k,
                    nnzA, &alpha, descr, valuesA, rowindA_csr, colindA,
                    B, ldb, &beta, C, ldc);
  }
  timer.Start();
  for (int i = 0; i < EXE_NUM_CUDA; i++) {
    cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_TRANSPOSE, m, n, k,
                    nnzA, &alpha, descr, valuesA, rowindA_csr, colindA,
                    B, ldb, &beta, C, ldc);
  }
  timer.Stop();
  time =  (float)timer.Elapsed()/EXE_NUM_CUDA;
  std::cout << "cuSPARSE 101 time = " << time << " ms" << std::endl;
#else
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  cusparseCreateCsr(&matA,
                    m, k, nnzA,
                    rowindA_csr,
                    colindA,
                    valuesA,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateDnMat(&matB,
                      k, n, n, (void *)B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
  cusparseCreateDnMat(&matC,
                      m, n, n,
                      C, CUDA_R_32F, CUSPARSE_ORDER_ROW);

  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;
  cusparseSpMMAlg_t alg = CUSPARSE_SPMM_CSR_ALG2;
  if(algid == -1){
    alg = CUSPARSE_SPMM_ALG_DEFAULT;
  } else if(algid == 2){
    alg = CUSPARSE_SPMM_CSR_ALG2;
  } else if (algid == 3) {
    alg = CUSPARSE_SPMM_CSR_ALG3;
  }
  for (int i = 0; i < WARMUP_NUM_CUDA; i++) {
    cusparseSpMM_bufferSize(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, alg,
        &workspace_size);
  }
  timer.startGPU();
  for (int i = 0; i < EXE_NUM_CUDA; i++) {
    cusparseSpMM_bufferSize(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, alg,
        &workspace_size);
  }
  timer.stopGPU("CuSparse CSR Buffer Time");
  std::cout << "cusparse csr buffer time: " <<  timer.ElapsedSeconds.count() / EXE_NUM_CUDA << " ms " << std::endl;
  void* workspace=NULL;
  cudaMalloc(&workspace, workspace_size);
  for (int i = 0; i < WARMUP_NUM_CUDA; i++) {
    cusparseSpMM(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F,alg,
        workspace);
  }
  swiftware::benchmark::Timer exeTime;
  timer.startGPU();
  for (int i = 0; i < EXE_NUM_CUDA; i++) {
    cusparseSpMM(
        cusparse_handle, transA, transB,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, alg,
        workspace);
  }
  timer.stopGPU("CuSparse CSR Exe Time");
  time = timer.ElapsedSeconds.count() / EXE_NUM_CUDA;
  std::cout << "cusparse csr exe time: " <<  time << " ms " << std::endl;
  cudaFree(workspace);
  cusparseDestroySpMat(matA);
  cusparseDestroyDnMat(matB);
  cusparseDestroyDnMat(matC);
#endif
}

int main () {
  int N = 1 << 20;
  float *x, *y, *d_x, *d_y;
  x = (float*) malloc(N*sizeof(float));
  y = (float*) malloc(N*sizeof(float));
  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++){
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = std::max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
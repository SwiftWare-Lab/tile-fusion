//
// Created by salehm32 on 08/12/23.
//

#ifndef SPARSE_FUSION_SPMV_SPMV_MKL_DEMO_UTILS_H
#define SPARSE_FUSION_SPMV_SPMV_MKL_DEMO_UTILS_H

template<typename T>
struct TensorInputs : public Inputs<T>{
  int M, N, K;
  sym_lib::CSC *A, *B;
  sym_lib::CSR *ACsr, *BCsr;

  T *Cx;
  T *CorrectMul;
  bool IsSolProvided;

  TensorInputs(int M1, int N1, int K1, int L1,
               sym_lib::CSC *A1, sym_lib::CSC *B1,
               int NumThreads1, int NumTrial1, std::string ExpN):Inputs<T>(NumTrial1, NumThreads1, ExpN){
    M = M1;
    N = N1;
    K = K1;
    L = L1;
    A = sym_lib::copy_sparse(A1);
    B = sym_lib::copy_sparse(B1);
    ACsr = sym_lib::csc_to_csr(A);
    BCsr = sym_lib::csc_to_csr(B);
    Cx = new double[K * N]();
    // randomly initialize the input
    for(int i=0; i<K*N; ++i){
      Cx[i] = 1.0; //(double)rand()/RAND_MAX;
    }
    CorrectMul = new double[L * N](); // the correct solution
    IsSolProvided = false;
    Inputs<T>::Threshold = 1e-6;
  }

  ~TensorInputs(){
    delete[] Cx;
    delete[] CorrectMul;
    delete A;
    delete B;
    delete ACsr;
    delete BCsr;
  }
};

#endif // SPARSE_FUSION_SPMV_SPMV_MKL_DEMO_UTILS_H

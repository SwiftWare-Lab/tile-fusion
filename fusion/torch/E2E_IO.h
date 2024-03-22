//
// Created by salehm32 on 10/01/24.
//

#ifndef SPARSE_FUSION_E2E_IO_H
#define SPARSE_FUSION_E2E_IO_H
#include "E2D_Def.h"
#include "aggregation/exceptions.h"
#include "aggregation/sparse_io.h"
#include "aggregation/test_utils.h"
#include "sparse-fusion/Fusion_Defs.h"
#include <fstream>
#include <random>
#include <torch/torch.h>

FloatDense *generateRandomFloatDenseMatrix(int M, int N);

FloatDense *readFloatDenseMatrixFromParameter(const sym_lib::TestParameters *Tp,
                                              int Rows, int Cols,
                                              std::string MtxPath);

FloatDense *readMtxArrayFloatDense(std::ifstream &in_file);

void normalizeAdjacencyMatrix(sym_lib::CSR *aCSC);

torch::Tensor convertCSRToTorchTensor(sym_lib::CSR &matrix);

torch::Tensor convertCSCToTorchTensor(sym_lib::CSC &matrix);

long *readMtxArrayInteger(std::ifstream &in_file, int m, int n);

void readMtxCscRealWithAddingSelfLoops(std::ifstream &in_file, sym_lib::CSC *&A, bool add_self_loops);

void compressTripletsToCscWithAddingSelfLoops(std::vector<sym_lib::triplet>& triplet_vec, sym_lib::CSC *A, bool add_self_loops = true);

long *getTargetsFromParameter(const sym_lib::TestParameters *Tp, int Rows, int Cols, std::string MtxPath);

FloatDense *generateRandomFloatDenseMatrix(int M, int N) {
  FloatDense *denseMatrix = new FloatDense(M, N, N);
  std::random_device randDev;
  std::mt19937 generator(randDev());
  std::uniform_real_distribution<float> distr(-1., 1.);
  float *a = denseMatrix->a;
  for (int i = 0; i < M * N; i++) {
    a[i] = distr(generator);
  }
  return denseMatrix;
}

FloatDense *readFloatDenseMatrixFromParameter(const sym_lib::TestParameters *Tp,
                                              int Rows, int Cols,
                                              std::string MtxPath) {
  FloatDense *featureMatrix = NULLPNTR;
  if (Tp->_gnn_parameters_mode == "Random") {
    featureMatrix = generateRandomFloatDenseMatrix(Rows, Cols);
  } else {
    //  std::string fileExt = Tp->_feature_matrix_path.substr(
    //      Tp->_matrix_path.find_last_of(".") + 1);
    //  if (fileExt == "mtx") {
    std::ifstream fin(MtxPath);
    featureMatrix = readMtxArrayFloatDense(fin);
    //  } else{
    //   std::cout << "File extension is not supported" << std::endl;
    //   exit(1);
    //  }
  }
  return featureMatrix;
}

FloatDense *readMtxArrayFloatDense(std::ifstream &in_file) {
  int shape, arith, mtx_format;
  int m, n;
  size_t nnz;
  sym_lib::read_header(in_file, m, n, nnz, arith, shape, mtx_format);
  FloatDense *matrix = new FloatDense(m, n, n);
  if (arith != sym_lib::REAL)
    throw sym_lib::mtx_arith_error("REAL", sym_lib::type_str(arith));
  else if (mtx_format != sym_lib::ARRAY)
    throw sym_lib::mtx_format_error("ARRAY", sym_lib::format_str(mtx_format));
  for (int i = 0; i < m * n; i++) { // writing from file row by row
    in_file >> matrix->a[i];
  }
  return matrix;
  // print_dense(A->row, A->col, A->lda, A->a);
}

void normalizeAdjacencyMatrix(sym_lib::CSR *aCSR) {
  float *degrees = new float[aCSR->m]{};
  std::memset(degrees, 0, aCSR->m * sizeof(float));
  for (int i = 0; i < aCSR->m; i++) {
    degrees[i] = aCSR->p[i + 1] - aCSR->p[i];
  }
  for (int i = 0; i < aCSR->m; i++) {
    for (int j = aCSR->p[i]; j < aCSR->p[i + 1]; j++) {
      aCSR->x[j] = aCSR->x[j] / sqrt(degrees[i] * degrees[aCSR->i[j]]);
    }
  }
}

torch::Tensor convertCSCToTorchTensor(sym_lib::CSC &matrix) {
  float *values = new float[matrix.nnz];
  for (int i = 0; i < matrix.nnz; i++) {
    values[i] = (float)matrix.x[i];
  }
  return torch::sparse_csc_tensor(
      torch::from_blob(matrix.p, {long(matrix.n)}, torch::kInt32),
      torch::from_blob(matrix.i, {long(matrix.nnz)}, torch::kInt32),
      torch::from_blob(values, {long(matrix.nnz)}, torch::kFloat32),
      {long(matrix.m), long(matrix.n)}, torch::kFloat32);
}
torch::Tensor convertCSRToTorchTensor(sym_lib::CSR &matrix) {
  float *values = new float[matrix.nnz];
  for (int i = 0; i < matrix.nnz; i++) {
    values[i] = (float)matrix.x[i];
  }
  return torch::sparse_csr_tensor(
      torch::from_blob(matrix.p, {long(matrix.n)}, torch::kInt32),
      torch::from_blob(matrix.i, {long(matrix.nnz)}, torch::kInt32),
      torch::from_blob(values, {long(matrix.nnz)}, torch::kFloat32),
      {long(matrix.m), long(matrix.n)}, torch::kFloat32);
}

long *readMtxArrayInteger(std::ifstream &in_file, int m, int n) {
  int shape, arith, mtx_format;
  size_t nnz;
  sym_lib::read_header(in_file, m, n, nnz, arith, shape, mtx_format);
  if (arith != sym_lib::INT && arith != sym_lib::REAL)
    throw sym_lib::mtx_arith_error("INT", sym_lib::type_str(arith));
  if (mtx_format != sym_lib::ARRAY)
    throw sym_lib::mtx_format_error("ARRAY", sym_lib::format_str(mtx_format));
  long *A = new long[m * n];
  for (int i = 0; i < m * n; i++) { // writing from file row by row
    in_file >> A[i];
  }
  std::ofstream file;
  return A;
}

sym_lib::CSC *get_matrix_from_parameter_with_adding_self_loops(const sym_lib::TestParameters *TP, bool AddSelfLoops) {
  sym_lib::CSC *aCSC = NULLPNTR;
  if (TP->_mode == "Random") {
    aCSC = sym_lib::random_square_sparse(TP->_dim1, TP->_density);
  } else {
    std::string fileExt =
        TP->_matrix_path.substr(TP->_matrix_path.find_last_of(".") + 1);
    if (fileExt == "mtx") {
      std::ifstream fin(TP->_matrix_path);
      readMtxCscRealWithAddingSelfLoops(fin, aCSC, AddSelfLoops);
    } else {
      std::cout << "File extension is not supported" << std::endl;
      exit(1);
    }
  }
  return aCSC;
}

void compressTripletsToCscWithAddingSelfLoops(std::vector<sym_lib::triplet>& triplet_vec, sym_lib::CSC *A, bool add_self_loops){
  assert(A->nnz == triplet_vec.size());
  auto *count = new int[A->n]();
  auto *has_diag = new bool[A->n]();
  for (auto i = 0; i < A->nnz; ++i) {
    count[triplet_vec[i].col]++;
    if(triplet_vec[i].col == triplet_vec[i].row)
      has_diag[triplet_vec[i].col] = true;
  }
  if(add_self_loops){
    int newNNZcount = 0;
    for (int i = 0; i < A->n; ++i) {
      if (!has_diag[i]){
        triplet_vec.push_back({i, i, 1});
        newNNZcount++;
        count[i]++;
      }
    }
    newNNZcount += A->nnz;
    int m = A->m;
    int n = A->n;
    delete A;
    A = new sym_lib::CSC(m, n, newNNZcount);
  }
  std::sort(triplet_vec.begin(), triplet_vec.end(),
            [](const sym_lib::triplet& a, const sym_lib::triplet& b){return (a.col<b.col) || (a.col==b.col && a.row<b.row);});
  A->p[0] = 0;
  for (auto j = 0; j < A->n; ++j) {
      A->p[j+1] = A->p[j] + count[j];
  }
  delete []count;
  for (auto k = 0; k < A->nnz; ++k) {
    A->i[k] = triplet_vec[k].row;
    A->x[k] = triplet_vec[k].val;
  }
}

void readMtxCscRealWithAddingSelfLoops(std::ifstream &in_file, sym_lib::CSC *&A, bool add_self_loops) {
  int n, m;
  int shape, arith, mtx_format;
  size_t nnz;
  std::vector<sym_lib::triplet> triplet_vec;

  sym_lib::read_header(in_file, m, n, nnz, arith, shape, mtx_format);
  if(arith != sym_lib::REAL && arith != sym_lib::INT && arith != sym_lib::PATTERN)
    throw sym_lib::mtx_arith_error("REAL", sym_lib::type_str(arith));
  if (mtx_format != sym_lib::COORDINATE)
    throw sym_lib::mtx_format_error("COORDINATE", sym_lib::format_str(mtx_format));
  bool read_val = true;
  if(arith == sym_lib::PATTERN)
    read_val = false;
  A = new sym_lib::CSC(m,n,nnz,false, sym_lib::shape2int(shape));
  sym_lib::read_triplets_real(in_file, nnz, triplet_vec, read_val, false);
  compressTripletsToCscWithAddingSelfLoops(triplet_vec, A, add_self_loops);
  A->nnz = A->p[n]; // if insert diag is true, it will be different.
                    //print_csc(A->n, A->p, A->i, A->x);
}

long *getTargetsFromParameter(const sym_lib::TestParameters *Tp, int Rows, int Cols, std::string MtxPath){
  std::string fileExt = Tp->_matrix_path.substr(
      Tp->_matrix_path.find_last_of(".") + 1);
  long *targets = NULLPNTR;
  if (fileExt == "mtx") {
    std::ifstream fin(MtxPath);
    targets = readMtxArrayInteger(fin, Rows, Cols);
  } else{
    std::cout << "File extension is not supported" << std::endl;
    exit(1);
  }
  return targets;
}

#endif // SPARSE_FUSION_E2E_IO_H

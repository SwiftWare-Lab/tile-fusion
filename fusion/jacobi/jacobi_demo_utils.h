//
// Created by kazem on 1/19/24.
//

#ifndef SPARSE_FUSION_JACOBI_DEMO_UTILS_H
#define SPARSE_FUSION_JACOBI_DEMO_UTILS_H

#include "jacobi.h"
#include "SWTensorBench.h"
#include "jacobi.h"
#include "sparse-fusion/Fusion_Defs.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include "sparse-fusion/SparseFusion.h"

using namespace swiftware::benchmark;

// print a dense matrix with dimension MxN
template <typename T> void printDense(int M, int N, T *X) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << X[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

inline void getSumRowCSR(int m, int *Ap, int *Ai, double *Ax, double *SumRow,
                         double &MaxVal) {
  double maxSum = 0; MaxVal = 0;
  for (int i = 0; i < m; ++i) {
    SumRow[i] = 0;
    for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
      SumRow[i] += Ax[j];
      if (std::abs(Ax[j]) > MaxVal) {
        MaxVal = std::abs(Ax[j]);
      }
    }
    if (std::abs(SumRow[i]) > maxSum) {
      maxSum = std::abs(SumRow[i]);
    }
  }
//  for (int i = 0; i < m; ++i) {
//    SumRow[i] /= maxSum;
//  }
}

template <typename T> struct TensorInputs : public Inputs<T> {
  int M, K; // AXx = Bx; where A is MxM and Bx and Xx is Mxk
  sym_lib::CSC *A;
  sym_lib::CSR *ACsr;

  T *Bx;
  T *CorrectSol;
  bool IsSolProvided;
  double MaxVal;

  TensorInputs(int M1, int K1, sym_lib::CSC *A1, int NumThreads1, int NumTrial1,
               std::string ExpN)
      : Inputs<T>(NumTrial1, NumThreads1, ExpN) {
    M = M1;
    K = K1;
    A = sym_lib::copy_sparse(A1);
    ACsr = sym_lib::csc_to_csr(A);
    Bx = new double[M * K]();
    double *tmpSumRow = new double[M]();
    getSumRowCSR(M, ACsr->p, ACsr->i, ACsr->x, tmpSumRow, MaxVal);
    // initialize the RHS
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        Bx[i * K + j] = tmpSumRow[i];
      }
    }
    //    for (int i = 0; i < K * M; ++i) {
    //      Bx[i] = 1.0; //(double)rand()/RAND_MAX;
    //    }
    CorrectSol = new double[M * K](); // the correct solution
    IsSolProvided = false;
    Inputs<T>::Threshold = 1e-6;
  }

  ~TensorInputs() {
    delete[] Bx;
    delete[] CorrectSol;
    delete A;
    delete ACsr;
  }
};

template <typename T> struct TensorOutputs : public Outputs<T> {
  int M, K;
  T *Xx1, *Xx2;

  TensorOutputs(int m, int k) : M(m), K(k) {
    Xx1 = new T[M * K]();
    Xx2 = new T[M * K]();
  }

  ~TensorOutputs() {
    delete[] Xx2;
    delete[] Xx1;
  }

  void printDx() {
    std::cout << "\n Xx1:\n";
    printDense<T>(M, K, Xx1);
    std::cout << "\n Xx2\n";
    printDense<T>(M, K, Xx2);
    std::cout << "\n";
  }

  void reset() {
    std::fill_n(Xx2, M * K, 0.0);
    std::fill_n(Xx1, M * K, 0.0);
  }
};

class JacobiCSRUnfused : public SWTensorBench<double> {
protected:
  TensorInputs<double> *InTensor;
  double Threshold = 1e-4;
  int MaxIters = 10000;
  double *WS;
  int RetValue = 0, WSSize = 0;

  void setup() override {
    this->St->OtherStats["NTile"] = {4};
    Threshold *= InTensor->MaxVal; //normalize
  }

  void preExecute() override {}

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    std::fill_n(WS, WSSize, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    RetValue =
        sym_lib::jacobiCSR(InTensor->M, InTensor->K, InTensor->ACsr->p,
                           InTensor->ACsr->i, InTensor->ACsr->x, OutTensor->Xx2,
                           InTensor->Bx, InTensor->K, Threshold, MaxIters, WS);
    t.stop();
    std::cout << "Return value: " << RetValue << std::endl;
    return t;
  }

  bool verify(double &Error) override {
    bool retValue = true;
    if (!InTensor->IsSolProvided) {
      Error = 0;
      return true;
    }
    double infNorm = 0;
    for (int i = 0; i < InTensor->M * InTensor->K; ++i) {
      if (std::abs(OutTensor->Xx2[i] - InTensor->CorrectSol[i]) > infNorm) {
        infNorm = std::abs(OutTensor->Xx2[i] - InTensor->CorrectSol[i]);
      }
    }
    Error = (double)infNorm;
    if (infNorm > InTensor->Threshold) {
      retValue = false;
    }
    return retValue;
  }

public:
  TensorOutputs<double> *OutTensor;
  JacobiCSRUnfused(TensorInputs<double> *In1, Stats *Stat1)
      : SWTensorBench<double>(In1, Stat1) {
    OutTensor = new TensorOutputs<double>(In1->M, In1->K);
    InTensor = In1;
    WSSize = In1->M + In1->ACsr->nnz + In1->M * In1->K;
    WS = new double[WSSize];
  }

  ~JacobiCSRUnfused() {
    delete OutTensor;
    delete[] WS;
  }
};

class JacobiCSRFused : public JacobiCSRUnfused {
protected:
  sym_lib::MultiDimensionalSet *FusedCompSet;
  sym_lib::ScheduleParameters Sp;
  Timer analysis() override {
    Timer t;
    t.start();
    // sym_lib::ScheduleParameters sp;
    // sp._num_threads = InTensor->NumThreads;
    //  create the fused set

    Sp._num_w_partition = std::max<int>(InTensor->ACsr->m / Sp.IterPerPartition,
                                        2 * Sp._num_threads);
    auto *sf01 = new sym_lib::SparseFusion(&Sp, 2);
    auto *mvDAG = sym_lib::diagonal(InTensor->ACsr->m, 1.0);
    auto *tmpCSCCSR = new sym_lib::CSC(InTensor->ACsr->m, InTensor->ACsr->n,
                                       InTensor->ACsr->nnz, InTensor->ACsr->p,
                                       InTensor->ACsr->i, InTensor->ACsr->x);
    auto *Di = InTensor->ACsr;
    // sym_lib::print_csc(1, "Di", 6, Di->p, Di->i, Di->x);
    sf01->fuse(0, mvDAG, tmpCSCCSR);

    // sf01->print_final_list();
    sf01->fuse(1, mvDAG, tmpCSCCSR);
    // sf01->print_final_list();
    auto pt = St->OtherStats["PackingType"];
    FusedCompSet = sf01->getFusedCompressed((int)pt[0]);
    int fusedNodesNum = FusedCompSet->getNumberOfFusedNodes();
    int fusedNnzNum = FusedCompSet->getFusedNnzNum(InTensor->ACsr);
    this->St->OtherStats["Number of Fused Nodes"] = {(double)fusedNodesNum};
    this->St->OtherStats["Number of Fused nnz"] = {(double)fusedNnzNum};
    // FusedCompSet->print_3d();
    delete sf01;
    delete mvDAG;
    delete tmpCSCCSR;

    t.stop();
    return t;
  }
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx2, InTensor->L * InTensor->N, 0.0);
    std::fill_n(WS, WSSize, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    RetValue = sym_lib::jacobiBiIterationFusedCSR(
        InTensor->M, InTensor->K, InTensor->ACsr->p, InTensor->ACsr->i,
        InTensor->ACsr->x, OutTensor->Xx1, OutTensor->Xx2, InTensor->Bx,
        InTensor->K, Threshold, MaxIters, WS, FusedCompSet->n1_,
        FusedCompSet->ptr1_, FusedCompSet->ptr2_, FusedCompSet->id_,
        FusedCompSet->type_, InTensor->NumThreads);
    t.stop();
     std::cout << "Return value: " << RetValue << std::endl;
    return t;
  }

public:
  TensorOutputs<double> *OutTensor;
  JacobiCSRFused(TensorInputs<double> *In1, Stats *Stat1,
                 sym_lib::ScheduleParameters Sp1)
      : JacobiCSRUnfused(In1, Stat1), Sp(Sp1) {
    OutTensor = new TensorOutputs<double>(In1->M, In1->K);
    InTensor = In1;
    WSSize = In1->M + In1->ACsr->nnz + In1->M * In1->K;
    WS = new double[WSSize];
  }

  ~JacobiCSRFused() { delete FusedCompSet; }
};

#endif // SPARSE_FUSION_JACOBI_DEMO_UTILS_H

//
// Created by salehm32 on 05/03/24.
//

#ifndef SPARSE_FUSION_SPMM_SPMM_SP_DEMO_H
#define SPARSE_FUSION_SPMM_SPMM_SP_DEMO_H

#include "SpMM_SpMM_Demo_Utils.h"

class SpMMSpMMUnFusedSP : public SWTensorBench<float> {
protected:
  TensorInputs<float> *InTensor;
  float* AValues;

  void setup() override {
    this->St->OtherStats["NTile"] = {4};
    this->St->OtherStats["Number of Fused Nodes"] = {0.};
    this->St->OtherStats["Number of Fused nnz"] = {0.};
    this->St->OtherStats["Tile Size Mean"] = {0.};
    this->St->OtherStats["Tile Size STD"] = {0.};
    for (int i = 0; i < InTensor->ACsr->nnz; ++i) {
      AValues[i] = (float)InTensor->ACsr->x[i];
    }
  }

  void preExecute() override {}

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSequential<float>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->ACsr->p,
        InTensor->ACsr->i, AValues, InTensor->Bx, OutTensor->ACx);
    swiftware::sparse::spmmCsrSequential<float>(
        InTensor->L, InTensor->N, InTensor->M, InTensor->BCsr->p,
        InTensor->BCsr->i, AValues, OutTensor->ACx, OutTensor->Xx);
    t.stop();
    return t;
  }

  bool verify(double &Error) override {
    bool retValue = true;
    if (!InTensor->IsSolProvided) {
      Error = 0;
      return true;
    }
    float infNorm = 0;
    for (int i = 0; i < InTensor->L * InTensor->N; ++i) {
      if (std::abs(OutTensor->Xx[i] - InTensor->CorrectSol[i]) > infNorm) {
        infNorm = std::abs(OutTensor->Xx[i] - InTensor->CorrectSol[i]);
      }
    }
    Error = (float)infNorm;
    if (infNorm > InTensor->Threshold) {
      retValue = false;
    }
    return retValue;
  }

public:
  TensorOutputs<float> *OutTensor;
  SpMMSpMMUnFusedSP(TensorInputs<float> *In1, Stats *Stat1)
      : SWTensorBench<float>(In1, Stat1) {
    OutTensor = new TensorOutputs<float>(In1->M, In1->N, In1->L);
    InTensor = In1;
    AValues = new float[In1->ACsr->nnz];
  }

  ~SpMMSpMMUnFusedSP() {
    delete OutTensor;
    delete AValues;
  }
};

class SpMMSpMMUnFusedParallelSP : public SpMMSpMMUnFusedSP {
protected:
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrParallel<float>(InTensor->M, InTensor->N, InTensor->K,
                                       InTensor->ACsr->p, InTensor->ACsr->i,
                                       AValues, InTensor->Bx,
                                       OutTensor->ACx, InTensor->NumThreads);
    swiftware::sparse::spmmCsrParallel<float>(InTensor->L, InTensor->N, InTensor->M,
                                       InTensor->BCsr->p, InTensor->BCsr->i,
                                       AValues, OutTensor->ACx,
                                       OutTensor->Xx, InTensor->NumThreads);
    t.stop();
    return t;
  }

public:
  SpMMSpMMUnFusedParallelSP(TensorInputs<float> *In1, Stats *Stat1)
      : SpMMSpMMUnFusedSP(In1, Stat1) {}
};

#ifdef MKL
class SpMMSpMMMKL_SP : public SpMMSpMMUnFusedSP {
protected:
  sparse_matrix_t A;
  sparse_matrix_t B;
  MKL_INT *LLI_A;
  MKL_INT *LLI_B;
  matrix_descr d;
  Timer execute() override {
    Timer t;
    OutTensor->reset();
    t.start();
    mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, this->A, this->d,
                    SPARSE_LAYOUT_ROW_MAJOR, this->InTensor->Bx,
                    this->InTensor->N, this->InTensor->N, 0,
                    this->OutTensor->ACx, this->InTensor->N);
    mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, this->B, this->d,
                    SPARSE_LAYOUT_ROW_MAJOR, this->OutTensor->ACx,
                    this->InTensor->N, this->InTensor->N, 0,
                    this->OutTensor->Xx, this->InTensor->N);
    t.stop();
    return t;
  }

public:
  SpMMSpMMMKL_SP(TensorInputs<float> *In1, Stats *Stat1)
      : SpMMSpMMUnFusedSP(In1, Stat1) {
    d.type = SPARSE_MATRIX_TYPE_GENERAL;

    LLI_A = new MKL_INT[this->InTensor->M + 1]();
    for (int l = 0; l < this->InTensor->M + 1; ++l) {
      LLI_A[l] = this->InTensor->ACsr->p[l];
    }

    LLI_B = new MKL_INT[this->InTensor->L + 1]();
    for (int l = 0; l < this->InTensor->L + 1; ++l) {
      LLI_B[l] = this->InTensor->BCsr->p[l];
    }

    mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, this->InTensor->M,
                            this->InTensor->K, LLI_A, LLI_A + 1,
                            this->InTensor->ACsr->i, this->AValues);
    mkl_sparse_s_create_csr(&B, SPARSE_INDEX_BASE_ZERO, this->InTensor->L,
                            this->InTensor->M, LLI_B, LLI_B + 1,
                            this->InTensor->BCsr->i, this->AValues);
    mkl_set_num_threads(this->InTensor->NumThreads);
  }

  ~SpMMSpMMMKL_SP() {
    mkl_free(A);
    mkl_free(B);
    delete[] LLI_A;
    delete[] LLI_B;
  }
};
#endif
class SpMMSpMMFusedInterLayerSP : public SpMMSpMMUnFusedSP {
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
    auto *tmpCSCCSR = new sym_lib::CSC(InTensor->BCsr->m, InTensor->BCsr->n,
                                       InTensor->BCsr->nnz, InTensor->BCsr->p,
                                       InTensor->BCsr->i, InTensor->BCsr->x);
    auto *Di = InTensor->BCsr;
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
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrFused<float>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, AValues, InTensor->BCsr->p,
        InTensor->BCsr->i, AValues, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

public:
  SpMMSpMMFusedInterLayerSP(TensorInputs<float> *In1, Stats *Stat1,
                          sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMUnFusedSP(In1, Stat1), Sp(SpIn) {}

  ~SpMMSpMMFusedInterLayerSP() { delete FusedCompSet; }
};

class SpMMSpMMFusedVariableTileSizeSP: public SpMMSpMMFusedInterLayerSP{
protected:
  InspectorForTileFusedCSRVariableTileSize *Inspector;

  Timer analysis() override{
    auto tm = St->OtherStats["TilingMethod"];
    if(tm[0] == sym_lib::Fixed){
      return SpMMSpMMFusedInterLayerSP::analysis();
    }
    else {
      Timer t1;
      t1.start();
      FusedCompSet = Inspector->generateVariableTileSizeSchedule(InTensor->ACsr,InTensor->N,sizeof(float));
      //    FusedCompSet->print_3d();
      t1.stop();
      return t1;
    }
  }

  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrFused<float>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, AValues, InTensor->BCsr->p,
        InTensor->BCsr->i, AValues, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

  SpMMSpMMFusedVariableTileSizeSP(TensorInputs<float> *In1, Stats *Stat1,
                                sym_lib::ScheduleParameters SpIn,
                                InspectorForTileFusedCSRVariableTileSize *Inspector1)
      : SpMMSpMMFusedInterLayerSP(In1, Stat1, SpIn){
    Inspector = Inspector1;
  }
public:
  SpMMSpMMFusedVariableTileSizeSP(TensorInputs<float> *In1, Stats *Stat1,
                                sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedInterLayerSP(In1, Stat1, SpIn){
    Inspector = new InspectorForTileFusedCSRVariableTileSize(SpIn, Stat1);
  }

  ~SpMMSpMMFusedVariableTileSizeSP(){
    delete Inspector;
  }
};
#ifdef __AVX2__
class SpMMSpMMFusedInterLayerVectorizedAvx256SP: public SpMMSpMMFusedVariableTileSizeSP{
protected:
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrFusedVectorized2_32SP(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, AValues, InTensor->BCsr->p,
        InTensor->BCsr->i, AValues, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

  SpMMSpMMFusedInterLayerVectorizedAvx256SP(TensorInputs<float> *In1, Stats *Stat1,
                                  sym_lib::ScheduleParameters SpIn,
                                  InspectorForTileFusedCSRVariableTileSize *Inspector1)
      : SpMMSpMMFusedVariableTileSizeSP(In1, Stat1, SpIn, Inspector1){
  }
public:
  SpMMSpMMFusedInterLayerVectorizedAvx256SP(TensorInputs<float> *In1, Stats *Stat1,
                                  sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedVariableTileSizeSP(In1, Stat1, SpIn){
  }
};
#endif

#ifdef __AVX512F__
class SpMMSpMMFusedInterLayerVectorizedAvx512SP: public SpMMSpMMFusedVariableTileSizeSP{
protected:
  Timer execute() override {
    //    std::fill_n(OutTensor->Xx, InTensor->L * InTensor->N, 0.0);
    //    std::fill_n(OutTensor->ACx, InTensor->M * InTensor->N, 0.0);
    OutTensor->reset();
    Timer t;
    t.start();
    swiftware::sparse::spmmCsrSpmmCsrFusedVectorized2_32Avx512SP(
        InTensor->M, InTensor->N, InTensor->K, InTensor->L, InTensor->ACsr->p,
        InTensor->ACsr->i, AValues, InTensor->BCsr->p,
        InTensor->BCsr->i, AValues, InTensor->Bx, OutTensor->Xx,
        OutTensor->ACx, FusedCompSet->n1_, FusedCompSet->ptr1_,
        FusedCompSet->ptr2_, FusedCompSet->id_, FusedCompSet->type_,
        InTensor->NumThreads);

    t.stop();
    return t;
  }

  SpMMSpMMFusedInterLayerVectorizedAvx512SP(TensorInputs<float> *In1, Stats *Stat1,
                                            sym_lib::ScheduleParameters SpIn,
                                            InspectorForTileFusedCSRVariableTileSize *Inspector1)
      : SpMMSpMMFusedVariableTileSizeSP(In1, Stat1, SpIn, Inspector1){
  }
public:
  SpMMSpMMFusedInterLayerVectorizedAvx512SP(TensorInputs<float> *In1, Stats *Stat1,
                                            sym_lib::ScheduleParameters SpIn)
      : SpMMSpMMFusedVariableTileSizeSP(In1, Stat1, SpIn){
  }
};
#endif

#endif // SPARSE_FUSION_SPMM_SPMM_SP_DEMO_H

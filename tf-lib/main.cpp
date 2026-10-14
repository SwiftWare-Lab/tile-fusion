//
// Created by salehm32 on 2026-05-08.
//

#include "Functions.h"
#include "def.h"
#include <stdexcept>

void bind_torch_functions(py::module_& m);
std::tuple<
    py::array_t<int>,
    py::array_t<int>,
    py::array_t<int>
    > inspect_vt(swiftware::def::CSRMatrix A, int64_t InDim, int64_t OutDim, int64_t CacheSize, int64_t NumThreads){

  int** rawSchedule = swiftware::fusion::generateVariableTileSizeScheduleGeMMSpMM(A.rows,
                                                               A.indptr_ptr(),
                                                               A.indices_ptr(),
                                                               InDim, OutDim, CacheSize, NumThreads, sizeof(float));

  int numKernels = 2;
  int* levelPtr = rawSchedule[0];
  int* mixPtr = rawSchedule[1];
  int* id = rawSchedule[2];
  int mixPtrSize = levelPtr[2] * 2 + 2;
  auto levelPtrPyArr = py::array_t<int>(
      {3},
      {sizeof(int)},
      levelPtr
  );
  auto mixPtrPyArr = py::array_t<int>(
      {mixPtrSize},
      {sizeof(int)},
      mixPtr
  );
  auto l2PtrPyArr = py::array_t<int>(
      {A.rows},
      {sizeof(int)},
      id
  );
  return {levelPtrPyArr, mixPtrPyArr, l2PtrPyArr};
}

std::tuple<
    py::array_t<int>,
    py::array_t<int>,
    py::array_t<int>
    > inspect(swiftware::def::CSRMatrix Adj, int64_t MTileSize) {
  std::vector<int *> schedule = swiftware::fusion::createSchedule(Adj.indptr_ptr(),
                                               Adj.indices_ptr(),
                                               Adj.rows,
                                               MTileSize);
  int* levelPtr = schedule[0];
  int* mixPtr = schedule[1];
  int* id = schedule[2];
  int M = Adj.rows;
  int mixPtrSize = levelPtr[2] * 2 + 1;
  auto levelPtrPyArr = py::array_t<int>(
      {3},
      {sizeof(int)},
      levelPtr
  );
  auto mixPtrPyArr = py::array_t<int>(
      {mixPtrSize},
      {sizeof(int)},
      mixPtr
  );
  auto l2PtrPyArr = py::array_t<int>(
      {M*2},
      {sizeof(int)},
      id
  );
  return {levelPtrPyArr, mixPtrPyArr, l2PtrPyArr};
}

std::tuple<
    py::array_t<int>,
    py::array_t<int>,
    py::array_t<int>,
    py::array_t<int>
    > inspect_spmm_spmm_vt(swiftware::def::CSRMatrix A, int64_t FeatureDim,
                           int64_t CacheSize, int64_t NumThreads) {
  int** rawSchedule = swiftware::fusion::generateVariableTileSizeScheduleSpMMSpMM(
      A.rows, A.indptr_ptr(), A.indices_ptr(), FeatureDim, CacheSize,
      NumThreads, sizeof(float));
  int* levelPtr = rawSchedule[0];
  int* parPtr = rawSchedule[1];
  int* partition = rawSchedule[2];
  int* parType = rawSchedule[3];
  int parPtrSize = levelPtr[2] + 1;
  int partitionSize = A.rows * 2;

  auto levelPtrPyArr = py::array_t<int>({3}, {sizeof(int)}, levelPtr);
  auto parPtrPyArr = py::array_t<int>({parPtrSize}, {sizeof(int)}, parPtr);
  auto partitionPyArr = py::array_t<int>({partitionSize}, {sizeof(int)}, partition);
  auto parTypePyArr = py::array_t<int>({partitionSize}, {sizeof(int)}, parType);
  return {levelPtrPyArr, parPtrPyArr, partitionPyArr, parTypePyArr};
}

std::tuple<
    py::array_t<int>,
    py::array_t<int>,
    py::array_t<int>,
    py::array_t<int>
    > inspect_spmm_spmm_fixed(swiftware::def::CSRMatrix A, int64_t MTileSize) {
  int** rawSchedule = swiftware::fusion::createSpMMSpMMFixedSchedule(A.rows, MTileSize);
  int* levelPtr = rawSchedule[0];
  int* parPtr = rawSchedule[1];
  int* partition = rawSchedule[2];
  int* parType = rawSchedule[3];
  int parPtrSize = levelPtr[2] + 1;
  int partitionSize = A.rows * 2;

  auto levelPtrPyArr = py::array_t<int>({3}, {sizeof(int)}, levelPtr);
  auto parPtrPyArr = py::array_t<int>({parPtrSize}, {sizeof(int)}, parPtr);
  auto partitionPyArr = py::array_t<int>({partitionSize}, {sizeof(int)}, partition);
  auto parTypePyArr = py::array_t<int>({partitionSize}, {sizeof(int)}, parType);
  return {levelPtrPyArr, parPtrPyArr, partitionPyArr, parTypePyArr};
}


py::array_t<float> executeFusedGeMMSpMM(swiftware::def::CSRMatrix Adj, py::array_t<float> Weight, py::array_t<float> Feature,
                                                swiftware::def::Op opC,
                                                py::array_t<int> LevelPtr, py::array_t<int> MixPtr, py::array_t<int> Partition,
                                                int64_t NumThreads) {
  if (opC == swiftware::def::Op::NonTransposed) {
    float *out = new float[Adj.rows * Weight.shape(1)]{};
    swiftware::fusion::fusedMKLGeMMSpMM(
        Adj.rows, Adj.indptr_ptr(), Adj.indices_ptr(), Adj.data_ptr(),
        Feature.shape(1), Weight.shape(1), Feature.mutable_data(),
        Weight.mutable_data(), out, NumThreads, 2, LevelPtr.mutable_data(),
        MixPtr.mutable_data(), Partition.mutable_data());
    //    std::cout << elapsed.count() << '\n';
    auto outArr = py::array_t<float>(
        {
            static_cast<py::ssize_t>(Adj.rows),
            static_cast<py::ssize_t>(Weight.shape(1))
        },
        {
            static_cast<py::ssize_t>(Weight.shape(1) * sizeof(float)),
            static_cast<py::ssize_t>(sizeof(float))
        },
        out
    );
    return outArr;
  }
  else{
    float *out = new float[Adj.rows * Weight.shape(0)]{};
    mkl_set_num_threads(1);
    swiftware::fusion::fusedMKLGeMMSpMMTransposedWeight(Adj.rows,
                                     Adj.indptr_ptr(),
                                     Adj.indices_ptr(),
                                     Adj.data_ptr(),
                                     Feature.shape(1),
                                     Weight.shape(0), Feature.mutable_data(),
                                     Weight.mutable_data(), out, NumThreads, 2, LevelPtr.mutable_data(),
                                     MixPtr.mutable_data(),
                                     Partition.mutable_data());
    auto outArr = py::array_t<float>(
        {
            static_cast<py::ssize_t>(Adj.rows),
            static_cast<py::ssize_t>(Weight.shape(0))
        },
        {
            static_cast<py::ssize_t>(Weight.shape(0) * sizeof(float)),
            static_cast<py::ssize_t>(sizeof(float))
        },
        out
    );
    return outArr;
  }
}

py::array_t<float> executeFusedGeMMSpMMVT(swiftware::def::CSRMatrix Adj, py::array_t<float> Weight, py::array_t<float> Feature,
                                                swiftware::def::Op opC,
                                                py::array_t<int> LevelPtr, py::array_t<int> MixPtr, py::array_t<int> Partition,
                                                int64_t NumThreads) {
  if (opC == swiftware::def::Op::NonTransposed) {
    float *out = new float[Adj.rows * Weight.shape(1)]{};
    swiftware::fusion::fusedMKLGeMMSpMMVT(
        Adj.rows, Adj.indptr_ptr(), Adj.indices_ptr(), Adj.data_ptr(),
        Feature.shape(1), Weight.shape(1), Feature.mutable_data(),
        Weight.mutable_data(), out, NumThreads, 2, LevelPtr.mutable_data(),
        MixPtr.mutable_data(), Partition.mutable_data());
    //    std::cout << elapsed.count() << '\n';
    auto outArr = py::array_t<float>(
        {
            static_cast<py::ssize_t>(Adj.rows),
            static_cast<py::ssize_t>(Weight.shape(1))
        },
        {
            static_cast<py::ssize_t>(Weight.shape(1) * sizeof(float)),
            static_cast<py::ssize_t>(sizeof(float))
        },
        out
    );
    return outArr;
  }
  else{
    float *out = new float[Adj.rows * Weight.shape(0)]{};
    mkl_set_num_threads(1);
    swiftware::fusion::fusedMKLGeMMSpMMTransposedWeightVT(Adj.rows,
                                     Adj.indptr_ptr(),
                                     Adj.indices_ptr(),
                                     Adj.data_ptr(),
                                     Feature.shape(1),
                                     Weight.shape(0), Feature.mutable_data(),
                                     Weight.mutable_data(), out, NumThreads, 2, LevelPtr.mutable_data(),
                                     MixPtr.mutable_data(),
                                     Partition.mutable_data());
    auto outArr = py::array_t<float>(
        {
            static_cast<py::ssize_t>(Adj.rows),
            static_cast<py::ssize_t>(Weight.shape(0))
        },
        {
            static_cast<py::ssize_t>(Weight.shape(0) * sizeof(float)),
            static_cast<py::ssize_t>(sizeof(float))
        },
        out
    );
    return outArr;
  }
}

py::array_t<float> executeFusedSpMMSpMMAvx256(
    swiftware::def::CSRMatrix A, swiftware::def::CSRMatrix B,
    py::array_t<float> Feature, py::array_t<int> LevelPtr,
    py::array_t<int> ParPtr, py::array_t<int> Partition,
    py::array_t<int> ParType, int64_t NumThreads) {
  if (Feature.ndim() != 2) {
    throw std::invalid_argument("Feature must be a 2D array");
  }
  if (A.rows != B.rows || A.rows != B.cols || A.cols != B.rows) {
    throw std::invalid_argument("A and B must have compatible square dimensions");
  }
  if (Feature.shape(0) != A.cols) {
    throw std::invalid_argument("Feature rows must match A.cols");
  }
  int featureDim = Feature.shape(1);
  if (featureDim % 32 != 0) {
    throw std::invalid_argument("SpMM-SpMM AVX256 kernels require feature dimension to be a multiple of 32");
  }

  float* out = new float[B.rows * featureDim]{};
  float* intermediate = new float[A.rows * featureDim]{};
  swiftware::fusion::fusedSpMMSpMMVectorizedAvx256SP(
      A.rows, featureDim, A.indptr_ptr(), A.indices_ptr(), A.data_ptr(),
      B.indptr_ptr(), B.indices_ptr(), B.data_ptr(), Feature.mutable_data(),
      out, intermediate, 2, LevelPtr.mutable_data(), ParPtr.mutable_data(),
      Partition.mutable_data(), ParType.mutable_data(), NumThreads);
  delete[] intermediate;

  auto outArr = py::array_t<float>(
      {
          static_cast<py::ssize_t>(B.rows),
          static_cast<py::ssize_t>(featureDim)
      },
      {
          static_cast<py::ssize_t>(featureDim * sizeof(float)),
          static_cast<py::ssize_t>(sizeof(float))
      },
      out
  );
  return outArr;
}

PYBIND11_MODULE(tf_lib, m) {
  py::enum_<swiftware::def::Op>(m, "Op")
      .value("NonTransposed", swiftware::def::Op::NonTransposed)
      .value("Transposed", swiftware::def::Op::Transposed);

  py::class_<swiftware::def::CSRMatrix>(m, "CSRMatrix")
      .def(py::init<
           py::array_t<float>,
           py::array_t<int>,
           py::array_t<int>,
           int,
           int
           >())
      .def_readonly("data", &swiftware::def::CSRMatrix::data)
      .def_readonly("indices", &swiftware::def::CSRMatrix::indices)
      .def_readonly("indptr", &swiftware::def::CSRMatrix::indptr)
      .def_readonly("rows", &swiftware::def::CSRMatrix::rows)
      .def_readonly("cols", &swiftware::def::CSRMatrix::cols);

  m.def("inspect_vt", &inspect_vt);
  m.def("inspect", &inspect);
  m.def("inspect_spmm_spmm_vt", &inspect_spmm_spmm_vt);
  m.def("inspect_spmm_spmm_fixed", &inspect_spmm_spmm_fixed);
  m.def("fusedGeMMSpMM", &executeFusedGeMMSpMM);
  m.def("fusedGeMMSpMMVT", &executeFusedGeMMSpMMVT);
  m.def("fusedSpMMSpMMAvx256", &executeFusedSpMMSpMMAvx256);
  m.def("fusedSpMMSpMMVT", &executeFusedSpMMSpMMAvx256);
  m.def("SpMMSpMMFusedInterLayerVectorizedAvx256SP", &executeFusedSpMMSpMMAvx256);
  m.def("FusedSpMMSpMM_VT", &executeFusedSpMMSpMMAvx256);
  bind_torch_functions(m);
}

//
// Created by salehm32 on 2026-05-08.
//

#ifndef TILE_FUSION_LIB_DEF_H
#define TILE_FUSION_LIB_DEF_H
#include<pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace swiftware{
  namespace def{
    class CSRMatrix {
    public:
     py::array_t<float> data;
     py::array_t<int> indices;
     py::array_t<int> indptr;
     int rows;
     int cols;

     CSRMatrix(py::array_t<float> data_,
               py::array_t<int> indices_,
               py::array_t<int> indptr_,
               int rows_,
               int cols_)
         : data(data_), indices(indices_), indptr(indptr_),
           rows(rows_), cols(cols_) {}

     float* data_ptr() { return data.mutable_data(); }
     int* indices_ptr() { return indices.mutable_data(); }
     int* indptr_ptr() { return indptr.mutable_data(); }
    };
    enum class Op{
      NonTransposed,
      Transposed,
    };
  }
}


#endif // TILE_FUSION_LIB_DEF_H

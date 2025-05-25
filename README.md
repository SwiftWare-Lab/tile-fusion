# Tile-fusion

This repository is a research artifact for "Loop Fusion in Matrix Multiplications with Sparse Dependence", Mohammad Mehdi Salehi, Kazem Cheshmi, ICS25.

## Build
You will need a C++ compiler (tested for GCC) and CMake to build this repository. MKL library is also needed for comparing the performance.

```
git clone --recursive https://github.com/SwiftWare-Lab/tile-fusion.git
cd tile-fusion/fusion/
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/mkl/latest/lib/intel64/;/path/to/mkl/latest/include/" ..
make -j 8
```

replace "/path/to/mkl/" with mkl directory path.

## Fused Operation API

These functionalities are defined in "tile-fusion.h" file. For now spmm-spmm and gemm-spmm executer codes are available.
inspector function:

```C++
swiftware::def::FusedSchedule swiftware::fusion::inspector(swiftware::def::CSRMatrix A, int NumThreads)
```

executor function for spmm-spmm and gemm-spmm when D = A * B * C:
```C++
swiftware::fusion::spMMSpMMExecutor(swiftware::def::CSRMatrix A, swiftware::def::CSRMatrix B, swiftware::def::Matrix C, swiftware::def::Matrix D, swiftware::def::FusedSchedule schedule, int NumThreads);
swiftware::fusion::geMMSpMMExecutor(swiftware::def::CSRMatrix A, swiftware::def::Matrix B, swiftware::def::Matrix C, swiftware::def::Matrix D, swiftware::def::FusedSchedule schedule, int NumThreads);
```
## Running experiments

### Download dataset

You also need to download the data where you like. The default location
is set to data in the current directory.

```bash
python fusion/scripts/dl_matrix.py
```

you can run the experiments using:

```bash
bash run.sh -t 1 -m ./data/SPD -c 128 -e spmm_spmm_sp
```
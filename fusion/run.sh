#!/bin/bash

BASELINE="SpMM_SpMM_Demo_UnFusedParallel"
UFDB=./data
BCOL=4
TEST=0
THRD=40
while getopts ":b:lt:d:m:" arg; do
  case "${arg}" in
    b)
      BASELINE=$OPTARG
      ;;
    l)
      TEST=1
      ;;
    d)
      BCOL=$OPTARG
      ;;
    t)
      THRD=$OPTARG
      ;;
    m)
      UFDB=$OPTARG
      ;;
    *) echo "Usage:
    -b BASELINE=SpMM_SpMM_Demo_UnFusedParallel        Choose a baseline to compare with Fused SpMM SpMM(Current base lines: SpMM_SpMM_Demo_UnFusedParallel,SpMM_SpMM_MKL)
    -l TEST=FALSE                                     Set if you want to run the script for one b_col
    -d BCOL=4                                         num of the columns of the dense matrix
    -t THRD=40                                        num of threads
    -m UFDB=./data                                    path of matrices data"
      exit 0
  esac
done
BINFILE="spmm_spmm_fusion"
if [ $BASELINE = "SpMM_SpMM_MKL" ]; then
  BINFILE="fused_vs_mkl"
fi



export MKL_DIR=$MKLROOT

which cmake
which gcc
which g++
which gdb
which make

#### Build
mkdir build
# shellcheck disable=SC2164
cd build
#make clean
#rm -rf *.txt
cmake -DCMAKE_PREFIX_PATH="$MKL_DIR/lib/intel64;$MKL_DIR/include;$MKL_DIR/../compiler/lib/intel64;_deps/openblas-build/lib/;/home/m/mmehride/kazem/programs/metis-5.1.0/libmetis;/home/m/mmehride/kazem/programs/metis-5.1.0/include/;"  -DCMAKE_BUILD_TYPE=Release ..
make -j 40


cd ..
BINPATH=./build/example/
LOGS=./build/logs/
SCRIPTPATH=./scripts/
MATLIST=$UFDB/mat_list.txt

mkdir $LOGS

MODE=2
# performing the experiments

if [ $TEST -eq 1 ]; then
  NUM_THREAD=$THRD
  export OMP_NUM_THREADS=$NUM_THREAD
  MKL_NUM_THREADS=$NUM_THREAD; export MKL_NUM_THREADS
  OMP_NUM_THREADS=$NUM_THREAD; export OMP_NUM_THREADS
  export MKL_DYNAMIC=FALSE;
  export OMP_DYNAMIC=FALSE;
  #export MKL_VERBOSE=1

#  python3 $SCRIPTPATH/dl_matrix.py $UFDB $MATLIST
  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST $BCOL > $LOGS/spmv_spmv_$BCOL.csv
  # plotting
#  python3 $SCRIPTPATH/plot.py $LOGS $BASELINE
else
  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 4 > $LOGS/spmv_spmv_4.csv
  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 32 > $LOGS/spmv_spmv_32.csv
  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 64 > $LOGS/spmv_spmv_64.csv
  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 128 > $LOGS/spmv_spmv_128.csv
  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 256 > $LOGS/spmv_spmv_256.csv
fi



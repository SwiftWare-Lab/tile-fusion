#!/bin/bash

UFDB="./data/ss-graphs/"
BCOL=4
EXP="spmm_spmm"
THRD=20
DOWNLOAD=0
BINPATH="./build/example"
USE_PAPI=0
MATLIST_FOLDER=""
OUTPUT_FOLDER=""
INTERNAL_JOB_ID=0
while getopts ":t:dc:m:i:e:l:j:z:" arg; do

  case "${arg}" in
    c)
      BCOL=$OPTARG
      ;;
    t)
      THRD=$OPTARG
      ;;
    m)
      UFDB=$OPTARG
      ;;
    d)
      DOWNLOAD=1
      ;;
    i)
      ID=$OPTARG
      ;;
    e)
      EXP=$OPTARG
      ;;
    l)
      MATLIST_FOLDER=$OPTARG
      ;;
    j)
      JOB_ID=$OPTARG
      ;;
    z)
      INTERNAL_JOB_ID=$OPTARG
      ;;
    *) echo "Usage:
    -c BCOL=4                                         num of the columns of the dense matrix
    -t THRD=40                                        num of threads
    -m UFDB=./data                                    path of matrices data
    -d DOWNLOAD=TRUE                                  Set if you want to download matrices under fusion folder"

      exit 0
  esac
done
MODE=3
if [ $EXP == "spmm_spmm" ]; then
  BINFILE="spmm_spmm_fusion"
  BINPATH="./build/example/"
elif [ $EXP == "spmm_spmm_sp" ]; then
  BINFILE="spmm_spmm_fusion_sp"
  BINPATH="./build/example/"
elif [ $EXP == "gemm_spmm" ]; then
  BINFILE="gcn_layer_demo"
  BINPATH="./build/gcn/"
elif [ $EXP == "gemm_spmm_sp" ]; then
  BINFILE="gcn_layer_sp_demo"
  BINPATH="./build/gcn/"
elif [ $EXP == "spmv_spmv" ]; then
  BINPATH="./build/spmv-spmv/"
  BINFILE="spmv_spmv_demo"
elif [ $EXP == "jacobi" ]; then
  BINPATH="./build/jacobi/"
  BINFILE="jacobi_demo"
  MODE=4
elif [ $EXP == "inspector" ]; then
  BINPATH="./build/example/"
  BINFILE="fusion_profiler"
elif [ $EXP == "profiling" ]; then
  BINPATH="./build/example/"
  BINFILE="spmm_spmm_papi_profiler"
  MODE=5
  USE_PAPI=1
else
  echo "Wrong experiment name"
  exit 0
fi


which cmake
which gcc
which g++
which gdb
which make
if [ -z "${MKL_DIR}" ]; then
  echo "MKL_DIR is already  set to: ${MKL_DIR}"
else
  export MKL_DIR=$MKLROOT
fi
#### Build
mkdir build
# shellcheck disable=SC2164
cd build
#make clean
#rm -rf *.txt
echo $MKL_DIR
if [ $USE_PAPI -eq 1 ]; then
  cmake -DCMAKE_PREFIX_PATH="$MKL_DIR/lib/intel64;$MKL_DIR/include;$MKL_DIR/../compiler/lib/intel64;_deps/openblas-build/lib/;${SCRATCH}/programs/papi/include/;"  -DPROFILING_WITH_PAPI=ON -DCMAKE_BUILD_TYPE=Release -DPAPI_PREFIX=${SCRATCH}/programs/papi/  ..
else
  cmake -DCMAKE_PREFIX_PATH="$MKL_DIR/lib/intel64;$MKL_DIR/include;$MKL_DIR/../compiler/lib/intel64;_deps/openblas-build/lib/;"  -DCMAKE_BUILD_TYPE=Release ..
fi
make -j 40


cd ..

#BINPATH=./build/example/
DATE=$(date -d "today" +"%Y%m%d%H%M")
if [ -z $JOB_ID ]; then
  LOGS="./build/logs/"
else
  LOGS="./build/logs-$JOB_ID"
fi
#LOGS="./build/logs-${DATE}/"
SCRIPTPATH=./scripts/
if [ -z "$MATLIST_FOLDER" ]; then
  MATLIST_FOLDER=$UFDB
fi
if [ -z $ID ]; then
  MATLIST=$MATLIST_FOLDER/mat_list.txt
else
  MATLIST=$MATLIST_FOLDER/mat_list$ID.txt
fi

mkdir $LOGS
# performing the experiments
if [ $DOWNLOAD -eq 1 ]; then
    python3 $SCRIPTPATH/dl_matrix.py $UFDB $MATLIST
fi
NUM_THREAD=$THRD
export OMP_NUM_THREADS=$NUM_THREAD
MKL_NUM_THREADS=$NUM_THREAD; export MKL_NUM_THREADS
OMP_NUM_THREADS=$NUM_THREAD; export OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE;
export OMP_DYNAMIC=FALSE;
#export MKL_VERBOSE=1

bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST $BCOL $LOGS $INTERNAL_JOB_ID
  # plotting
#  python3 $SCRIPTPATH/plot.py $LOGS $BASELINE
#else
#  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 4 > $LOGS/spmv_spmv_4.csv
#  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 32 > $LOGS/spmv_spmv_32.csv
#  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 64 > $LOGS/spmv_spmv_64.csv
#  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 128 > $LOGS/spmv_spmv_128.csv
#  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 256 > $LOGS/spmv_spmv_256.csv
#fi

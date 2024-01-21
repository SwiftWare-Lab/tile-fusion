#!/bin/bash

UFDB="./data/ss-graphs/"
BCOL=4
EXP="spmm_spmm"
THRD=20
DOWNLOAD=0
ID=0
while getopts ":t:dc:m:i:e:" arg; do

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
  BINPATH=./build/example/
elif [ $EXP == "spmv_spmv" ]; then
  BINPATH=./build/spmv-spmv/
  BINFILE="spmv_spmv_demo"
elif [ $EXP == "jacobi" ]; then
  BINPATH=./build/jacobi/
  BINFILE="jacobi_demo"
  MODE=4
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
cmake -DCMAKE_PREFIX_PATH="$MKL_DIR/lib/intel64;$MKL_DIR/include;$MKL_DIR/../compiler/lib/intel64;_deps/openblas-build/lib/;/home/m/mmehride/kazem/programs/metis-5.1.0/libmetis;/home/m/mmehride/kazem/programs/metis-5.1.0/include/;"  -DCMAKE_BUILD_TYPE=Release ..
make -j 40


cd ..

#BINPATH=./build/example/
DATE=$(date -d "today" +"%Y%m%d%H%M")
LOGS="./build/logs-${DATE}/"
SCRIPTPATH=./scripts/
if [ $ID -eq 0 ]; then
  MATLIST=$UFDB/mat_list.txt
else
  MATLIST=$UFDB/mat_list$ID.txt
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

bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST $BCOL $LOGS
  # plotting
#  python3 $SCRIPTPATH/plot.py $LOGS $BASELINE
#else
#  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 4 > $LOGS/spmv_spmv_4.csv
#  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 32 > $LOGS/spmv_spmv_32.csv
#  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 64 > $LOGS/spmv_spmv_64.csv
#  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 128 > $LOGS/spmv_spmv_128.csv
#  bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE $THRD $MATLIST 256 > $LOGS/spmv_spmv_256.csv
#fi
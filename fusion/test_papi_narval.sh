#!/bin/bash


#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH --constraint=rome
#SBATCH --mem=254000M
#SBATCH -t 47:59:00

UFDB="./data/ss-graphs/"
BCOL=4
EXP="spmm_spmm"
THRD=20
DOWNLOAD=0
ID=0
BINPATH="./build/example"
USE_PAPI=0
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
  BINPATH="./build/example/"
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
  USE_PAPI=1
else
  echo "Wrong experiment name"
  exit 0
fi


#module load NiaEnv/.2022a
#module load StdEnv/2023
module load StdEnv/2020
#module load intel/2023.2.1
module load intel/2022.1.0

echo "========> ${MKLROOT}"
echo " -------> ${MKL_DIR}"
export MKL_DIR=$MKLROOT




PAPI_INSTALL=0
if [ ${PAPI_INSTALL} -eq 1 ]; then
	echo "---- Installing PAPI ----"
	# Install PAPI library
	#git clone https://bitbucket.org/icl/papi.git  
	git clone https://github.com/icl-utk-edu/papi.git
	cd papi/src
	mkdir -p -- ${SCRATCH}/programs/papi
	./configure --prefix=${SCRATCH}/programs/papi/
	make
	make install
	cd ../../
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
make clean
#rm -rf *.txt
echo $MKL_DIR
cmake -DCMAKE_CXX_COMPILER=icc -DCMAKE_C_COMPILER=icc -DCMAKE_PREFIX_PATH="$MKL_DIR/lib/intel64;$MKL_DIR/include;$MKL_DIR/../compiler/lib/intel64;_deps/openblas-build/lib/;${SCRATCH}/programs/papi/include/;"  -DPROFILING_WITH_PAPI=ON -DCMAKE_BUILD_TYPE=Release -DPAPI_PREFIX=${SCRATCH}/programs/papi/  ..
#make -j 40

make -j 40  spmm_spmm_papi_profiler
cd ..

DATE=$(date -d "today" +"%Y%m%d%H%M")
LOGS="./build/logs-${DATE}/"
SCRIPTPATH=./scripts/
if [ $ID -eq 0 ]; then
  MATLIST=$UFDB/mat_list.txt
else
  MATLIST=$UFDB/mat_list$ID.txt
fi

mkdir $LOGS

bash $SCRIPTPATH/run_exp.sh ./build/example/spmm_spmm_papi_profiler $UFDB $MODE $THRD $MATLIST $BCOL $LOGS



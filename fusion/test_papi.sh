#!/bin/bash


#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehid20@gmail.com
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH -t 11:59:00
#SBATCH --constraint=cascade


module load NiaEnv/.2022a
module load intel/2022u2
export MKL_DIR=$MKLROOT
module load cmake


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
cmake -DCMAKE_PREFIX_PATH="$MKL_DIR/lib/intel64;$MKL_DIR/include;$MKL_DIR/../compiler/lib/intel64;_deps/openblas-build/lib/;$SCRATCH/programs/papi/include/;"  -DPROFILING_WITH_PAPI=ON -DCMAKE_BUILD_TYPE=Release -DPAPI_PREFIX=${SCRATCH}/programs/papi/  ..
#make -j 40

make -j 40  spmm_spmm_papi_profiler
./example/spmm_spmm_papi_profiler -ah

cd ..
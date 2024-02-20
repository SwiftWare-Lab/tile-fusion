#!/bin/bash


#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH --constraint=rome
#SBATCH --mem=254000M
#SBATCH -t 47:59:00




#module load NiaEnv/.2022a
#module load StdEnv/2023
module load StdEnv/2020
#module load intel/2023.2.1
module load intel/2022.1.0

echo "========> ${MKLROOT}"
echo " -------> ${MKL_DIR}"
export MKL_DIR=$MKLROOT




PAPI_INSTALL=#1
if [ ${PAPI_INSTALL} -eq 1 ]; then
	echo "---- Installing PAPI ----"
	# Install PAPI library
	#git clone https://bitbucket.org/icl/papi.git  
	git clone https://github.com/icl-utk-edu/papi.git
	cd papi/src
	mkdir -p -- ${HOME}/programs/papi
	./configure --prefix=${HOME}/programs/papi/
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
cmake -DCMAKE_CXX_COMPILER=icc -DCMAKE_C_COMPILER=icc -DCMAKE_PREFIX_PATH="$MKL_DIR/lib/intel64;$MKL_DIR/include;$MKL_DIR/../compiler/lib/intel64;_deps/openblas-build/lib/;${HOME}/programs/papi/include/;"  -DPROFILING_WITH_PAPI=ON -DCMAKE_BUILD_TYPE=Release -DPAPI_PREFIX=${HOME}/programs/papi/  ..
#make -j 40

make -j 40  spmm_spmm_papi_profiler
./example/spmm_spmm_papi_profiler -ah

cd ..


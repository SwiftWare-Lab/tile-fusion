#!/bin/bash


#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=kazem.cheshmi@gmail.com
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH -t 11:59:00
#SBATCH --constraint=cascade

DOWNLOAD=0
DOWNLOAD=$1

module load NiaEnv/.2022a
module load intel/2022u2
export MKL_DIR=$MKLROOT
module load cmake
module load gcc

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
cmake -DCMAKE_PREFIX_PATH="$MKLROOT/lib/intel64;$MKLROOT/include;$MKLROOT/../compiler/lib/intel64;_deps/openblas-build/lib/;/home/m/mmehride/kazem/programs/metis-5.1.0/libmetis;/home/m/mmehride/kazem/programs/metis-5.1.0/include/;"  -DCMAKE_BUILD_TYPE=Release ..
make -j 40


cd ..

BINPATH=./build/example/
# UFDB=$SCRATCH/UFDB/SPD/  #$HOME/UFDB/SPD/
#UFDB=/scratch/m/mmehride/kazem/UFDB/SPD
UFDB=./data
LOGS=./build/logs/
SCRIPTPATH=./scripts/
MATLIST=./scripts/mat_list.txt

if [ $DOWNLOAD -eq 1 ]; then
    python3 $SCRIPTPATH/dl_matrix.py $UFDB $MATLIST
fi


THRD=20
NUM_THREAD=20
export OMP_NUM_THREADS=20



MKL_NUM_THREADS=$NUM_THREAD; export MKL_NUM_THREADS
OMP_NUM_THREADS=$NUM_THREAD; export OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE;
export OMP_DYNAMIC=FALSE;
#export MKL_VERBOSE=1


mkdir $LOGS

MODE=2
# performing the experiments
bash $SCRIPTPATH/run_exp.sh $BINPATH/spmm_spmm_example $UFDB $MODE $THRD $MATLIST 4 > $LOGS/spmv_spmv_4.csv
#bash $SCRIPTPATH/run_exp.sh $BINPATH/spmm_spmm_fusion $UFDB $MODE $THRD $MATLIST 32 > $LOGS/spmv_spmv_32.csv
#bash $SCRIPTPATH/run_exp.sh $BINPATH/spmm_spmm_fusion $UFDB $MODE $THRD $MATLIST 64 > $LOGS/spmv_spmv_64.csv
#bash $SCRIPTPATH/run_exp.sh $BINPATH/spmm_spmm_fusion $UFDB $MODE $THRD $MATLIST 128 > $LOGS/spmv_spmv_128.csv
#bash $SCRIPTPATH/run_exp.sh $BINPATH/spmm_spmm_fusion $UFDB $MODE $THRD $MATLIST 256 > $LOGS/spmv_spmv_256.csv



# plotting
python3 $SCRIPTPATH/plot_example.py $LOGS/spmv_spmv_4.csv

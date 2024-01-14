#!/bin/bash
#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="gcb-end2end"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH -t 11:59:00
#SBATCH --constraint=cascade
DATA="./data/planetoid-graphs/"
MODE="GCNWithDifferentFusionLevels"
THREADS=8
BCOL=100

while getopts ":t:e:m:" arg; do

  case "${arg}" in
    e)
      BCOL=$OPTARG
      ;;
    t)
      THREADS=$OPTARG
      ;;
    m)
      DATA=$OPTARG
      ;;
    *) echo "Usage:
    -e Experiment=tri-banded        Choose a baseline to compare with Fused SpMM SpMM(Current base lines: SpMM_SpMM_Demo_UnFusedParallel,SpMM_SpMM_MKL)
    -c Feature Dimension=4                                         num of the columns of the dense matrix
    -t THRD=40                                        num of threads
    -m DATA=./pyg/data                                    path of matrices data"
      exit 0
  esac
done

MATLIST="$DATA/mat_list.txt"


module load NiaEnv/.2022a
module load intel/2022u2
export MKL_DIR=$MKLROOT
module load gcc
module load cmake
module load python

mkdir build
# shellcheck disable=SC2164
cd build
TORCH_LIB=/home/salehm32/pytorch
cmake -DCMAKE_PREFIX_PATH="$TORCH_LIB;$MKL_DIR/lib/intel64;$MKL_DIR/include;$MKL_DIR/../compiler/lib/intel64;_deps/openblas-build/lib/"  -DCMAKE_BUILD_TYPE=Release ..
make
mkdir logs
cd ..
BINPATH=./build/torch

NUM_THREAD=$THREADS
echo $NUM_THREAD
export OMP_NUM_THREADS=$NUM_THREAD
OMP_NUM_THREADS=$NUM_THREAD; export OMP_NUM_THREADS
MKL_NUM_THREADS=$NUM_THREAD; export MKL_NUM_THREADS
export MKL_DYNAMIC=FALSE;
export OMP_DYNAMIC=FALSE;

#if ! [ -d $DATA ]; then # if data folder does not exist(not applicable for experiments with banded matrices)
#  mkdir $DATA
# python ./scripts/pyg_data_exporter.py ./pyg
# echo "TEST"
#fi

header=1
sr=1
while read line; do
    echo "for $line $BCOL $ED $tn $mw"
    if [ $header -eq 1 ]; then
      $BINPATH/fused_gcn -dp $DATA/$line -nt $THREADS -ah -ed $BCOL > ./build/logs/gcn_end2end.csv
      header=0
    else
      $BINPATH/fused_gcn -dp $DATA/$line -nt $THREADS -ed $BCOL >> ./build/logs/gcn_end2end.csv
    fi
done < $MATLIST

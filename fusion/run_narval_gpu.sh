#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=100000M
#SBATCH --output="fusion.%j.%N.out"
#SBATCH -t 3:59:00

UFDB=$1
EXP=$2
BUILD=$3
PROFILING=$4

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6
module load cmake/3.27.7

export CUDACXX=$(which nvcc)

if [ $BUILD -eq 1 ]; then
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8
cd ..
fi


ID=$SLURM_JOB_ID
if [ $PROFILING -eq 1 ]; then
  MODE=7
else
  MODE=6
fi
MATLIST=$UFDB/mat_list.txt
SCRIPTPATH="./scripts/"
LOGS="./build/logs-$ID/"
mkdir $LOGS
 BINPATH="./build/gpu/"
if [ $EXP == "spmm_spmm" ]; then
 BINFILE="spmm_spmm_demo_gpu"
elif [ $EXP == "spmm" ]; then
  BINFILE="spmm_demo_gpu"
fi

bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE 1 $MATLIST 32 $LOGS 0
bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE 1 $MATLIST 64 $LOGS 0
bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE 1 $MATLIST 128 $LOGS 0
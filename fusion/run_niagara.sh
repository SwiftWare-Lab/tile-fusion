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

BASE_LINE="SpMM_SpMM_Demo_UnFusedParallel"
UFDB=$SCRATCH/UFDB/graphs/
DIM=32
#UFDB=$HOME/UFDB/banded/
while getopts ":b:lm:d:" arg; do
  case "${arg}" in
    b)
      BASE_LINE=$OPTARG
      ;;
    l)
      TEST=1
      ;;
    m)
      UFDB=$OPTARG
      ;;
    d)
      DIM=$OPTARG
      ;;
    *) echo "Usage:
    -b BASELINE=SpMM_SpMM_Demo_UnFusedParallel        Choose a baseline to compare with Fused SpMM SpMM(Current base lines: SpMM_SpMM_Demo_UnFusedParallel,SpMM_SpMM_MKL)
    -l TEST=FALSE                                     Set if you want to run the script for one b_col
    -m UFDB=$SCRATCH/UFDB/AM/                        path of matrices data"
      exit 0
  esac
done


module load NiaEnv/.2022a
#module load intel/2022u2
module load cmake
module load gcc
module load mkl

if [ $TEST -eq 1 ]; then
  bash run.sh -b $BASE_LINE -l -m $UFDB -d $DIM
else
  bash run.sh -b $BASE_LINE -t 40 -m $UFDB
fi
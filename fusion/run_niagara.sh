#!/bin/bash


#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH -t 11:59:00
#SBATCH --constraint=cascade

UFDB=$SCRATCH/data/graphs/
#UFDB=$HOME/UFDB/tri-banded/
EXP=spmv_spmv
BCOL=32
ID=""
THREADS=8
USE_PAPI=0
while getopts ":lm:c:i:e:t:" arg; do
  case "${arg}" in
    l)
      TEST=1
      ;;
    m)
      UFDB=$OPTARG
      ;;
    c)
      BCOL=$OPTARG
      ;;
    e)
      EXP=$OPTARG
      ;;
    i)
      ID=$OPTARG
      ;;
    t)
      THREADS=$OPTARG
      ;;
    *) echo "Usage:
    -l TEST=FALSE                                     Set if you want to run the script for one b_col
    -m UFDB=$SCRATCH/UFDB/AM/                        path of matrices data"
      exit 0
  esac
done


module load NiaEnv/.2022a
module load intel/2022u2
export MKL_DIR=$MKLROOT
module load cmake
#module load gcc

module load cmake
#module load gcc
ID_OPT="-i $ID"
if [ -z "$ID" ]; then
  ID_OPT=""
fi
if [ $TEST -eq 1 ]; then
    bash run.sh -m $UFDB -c $BCOL -e $EXP -t $THREADS -j $SLURM_JOB_ID "$ID_OPT"
else
  bash run.sh -t $THREADS -m $UFDB -c $BCOL -e $EXP -j $SLURM_JOB_ID "$ID_OPT"
fi
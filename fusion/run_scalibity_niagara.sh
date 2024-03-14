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
#SBATCH --array=0-20%6  # Allows no more than 6 of the jobs to run simultaneously

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  SLURM_ARRAY_TASK_ID=""
fi
TASK_ID=$SLURM_ARRAY_TASK_ID
BCOL_ID=$((TASK_ID / 7))
THREAD_ID=$((TASK_ID % 7))
BCOLS=(32 64 128)
NUM_THREAD_LIST=(2 4 8 16 20 32 40)
UFDB=$SCRATCH/data/graphs/
#UFDB=$HOME/UFDB/tri-banded/
EXP=spmv_spmv
BCOL=${BCOLS[${BCOL_ID}]}
NUM_THREAD=${NUM_THREAD_LIST[${THREAD_ID}]}
while getopts ":lm:c:e:" arg; do
  case "${arg}" in
    l)
      TEST=1
      ;;
    m)
      UFDB=$OPTARG
      ;;
    e)
      EXP=$OPTARG
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

if [ $TEST -eq 1 ]; then
    bash run.sh -m $UFDB -c 8 -i 200 -e $EXP -t 8 -j $SLURM_ARRAY_JOB_ID -z $THREAD_ID
else
  bash run.sh -t $NUM_THREAD -m $UFDB -c $BCOL -i 200  -e $EXP -j $SLURM_ARRAY_JOB_ID -z $THREAD_ID
fi
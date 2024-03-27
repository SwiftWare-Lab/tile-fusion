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
#SBATCH --constraint=rome
#SBATCH --mem=254000M
#SBATCH --array=0-59%6  # Allows no more than 6 of the jobs to run simultaneously

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  SLURM_ARRAY_TASK_ID=""
fi
TASK_ID=$SLURM_ARRAY_TASK_ID
BCOL_ID=$((TASK_ID / 6))
THREAD_ID=$((TASK_ID % 6))
BCOLS=(32 64 128)
NUM_THREAD_LIST=(2 4 8 16 32 64)
UFDB=$SCRATCH/graphs/
#UFDB=$HOME/UFDB/tri-banded/
EXP=spmv_spmv
BCOL=64
NUM_THREAD=${NUM_THREAD_LIST[${THREAD_ID}]}
while getopts ":lm:c:e:i:" arg; do
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
    i)
      MAT_ID=$OPTARG
      ;;
    *) echo "Usage:
    -l TEST=FALSE                                     Set if you want to run the script for one b_col
    -m UFDB=$SCRATCH/UFDB/AM/                        path of matrices data"
      exit 0
  esac
done
MATLIST_FOLDER=$UFDB/mat_lists/
MAT_ID=$((TASK_ID / 7))
module load NiaEnv/.2022a
module load intel/2022u2
export MKL_DIR=$MKLROOT
module load cmake
#module load gcc

if [ $TEST -eq 1 ]; then
    bash run.sh -m $UFDB -c 8 -i $MAT_ID -e $EXP -t 8 -j $SLURM_ARRAY_JOB_ID -z $THREAD_ID -l $MATLIST_FOLDER
else
  bash run.sh -t $NUM_THREAD -m $UFDB -c $BCOL -i $MAT_ID  -e $EXP -j $SLURM_ARRAY_JOB_ID -z $THREAD_ID -l $MATLIST_FOLDER
fi
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

UFDB=$SCRATCH/data/graphs/
#UFDB=$HOME/UFDB/tri-banded/
EXP=spmv_spmv
BCOL=32
ID=0
USE_PAPI=0
while getopts ":lm:c:i:e:p" arg; do
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
    p)
      USE_PAPI=1
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
  if [ $USE_PAPI -eq 1 ]; then
    bash run.sh -m $UFDB -c 8 -i $ID -e $EXP -t 8 -p
  else
    bash run.sh -m $UFDB -c 8 -i $ID -e $EXP -t 8
  fi
else
  if [ $USE_PAPI -eq 1 ]; then
    bash run.sh -t 20 -m $UFDB -c $BCOL -i $ID  -e $EXP -p
  else
  bash run.sh -t 20 -m $UFDB -c $BCOL -i $ID  -e $EXP
  fi
fi
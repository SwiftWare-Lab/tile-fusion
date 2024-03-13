#!/bin/bash


#SBATCH --cpus-per-task=64
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH --constraint=rome
#SBATCH --mem=254000M
#SBATCH -t 47:59:00



UFDB=$SCRATCH/data/graphs/
#UFDB=$HOME/UFDB/tri-banded/
EXP=spmv_spmv
BCOL=32
while getopts ":lm:c:e:" arg; do
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
    *) echo "Usage:
    -l TEST=FALSE                                     Set if you want to run the script for one b_col
    -m UFDB=$SCRATCH/UFDB/AM/                        path of matrices data"
      exit 0
  esac
done


#module load NiaEnv/.2022a
#module load StdEnv/2023
module load StdEnv/2020
module load gcc/11.3.0
#echo "========> ${MKLROOT}"
#echo " -------> ${MKL_DIR}"
#export MKL_DIR=$MKLROOT

module load cmake
#module load gcc #we need to add -march=core-avx2 to CXXFLAGS

if [[ $TEST -eq 1 ]]; then
  bash run.sh -m $UFDB -c 8 -e $EXP
else
  bash run.sh -t 32 -m $UFDB -c $BCOL -e $EXP
fi





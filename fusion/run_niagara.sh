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
while getopts ":b:l" arg; do
  case "${arg}" in
    b)
      BASE_LINE=$OPTARG
      ;;
    l)
      TEST=1
      ;;
    *) echo "usage: "
      exit 1
  esac
done


module load NiaEnv/2022a
module load intel/2022u2
module load cmake
module load gcc/12.2.0
module load mkl/2022.1.0

if [ $TEST -eq 1 ]; then
  bash run.sh -b $BASE_LINE -l
else
  bash run.sh -b $BASE_LINE -t 40
fi
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

BASE_LINE="SpMM_SpMM_Demo_UnFusedParallel"
while getopts ":b:" arg; do
  case "${arg}" in
    b)
      BASE_LINE=$OPTARG
      ;;
    *) echo "usage: "
      exit 1
  esac
done


module load NiaEnv/.2022a
module load intel/2022u2
module load cmake
module load gcc

bash run.sh -b $BASE_LINE
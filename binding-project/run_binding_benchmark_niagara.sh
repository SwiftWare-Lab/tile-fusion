#!/bin/bash

#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="gcn-end2end"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH -t 1:59:00
#SBATCH --constraint=cascade


DATA_FOLDER=''
MAT_LIST=''

DATA_FOLDER="./data/ss-graphs/"
BCOL=32
THRD=20
while getopts ":t:c:m:" arg; do

  case "${arg}" in
    c)
      BCOL=$OPTARG
      ;;
    t)
      THRD=$OPTARG
      ;;
    m)
      DATA_FOLDER=$OPTARG
      ;;
    *) echo "Usage:"
      exit 0
  esac
done


module load NiaEnv/.2022a
module load intel/2022u2
export MKL_DIR=$MKLROOT
module load gcc
module load cmake
module load python

which python
which intel
which cmake

bash run_binding_benchmark.sh -t $THRD -m $DATA_FOLDER -c $BCOL -v ./venv
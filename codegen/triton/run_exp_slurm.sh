#!/bin/bash

#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=40000M
#SBATCH --output="fusion.%j.%N.out"
#SBATCH -t 3:59:00

MAT_FILE=$1
MAT_DIR=$2

module load python/3.13.0
module load cuda/12.6.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt

python sddmm_spmm.py $MAT_FILE $MAT_DIR
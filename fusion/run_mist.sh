#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH -t 3:59:00

UFDB=$1
BCOL=$2
BUILD=$3

module load MistEnv/2021a
module load cuda/11.8.0
module load gcc/11.4.0
module load cmake/3.27.8

export CUDACXX=$(which nvcc)

if [ $BUILD -eq 1 ]; then
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8
cd ..
fi


ID=$SLURM_JOB_ID
MODE=6
MATLIST=$UFDB/mat_list.txt
SCRIPTPATH="./scripts/"
LOGS="./build/logs/"
mkdir $LOGS
 BINPATH="./build/gpu/"
BINFILE="spmm_spmm_demo_gpu"

bash $SCRIPTPATH/run_exp.sh $BINPATH/$BINFILE $UFDB $MODE 1 $MATLIST $BCOL $LOGS $ID

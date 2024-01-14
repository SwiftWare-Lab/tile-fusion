while getopts ":t:e:m:" arg; do

  case "${arg}" in
    e)
      BCOL=$OPTARG
      ;;
    t)
      THREADS=$OPTARG
      ;;
    m)
      DATA=$OPTARG
      ;;
    *) echo "Usage:
    -e Experiment=tri-banded        Choose a baseline to compare with Fused SpMM SpMM(Current base lines: SpMM_SpMM_Demo_UnFusedParallel,SpMM_SpMM_MKL)
    -c Feature Dimension=4                                         num of the columns of the dense matrix
    -t THRD=40                                        num of threads
    -m DATA=./pyg/data                                    path of matrices data"
      exit 0
  esac
done

MATLIST="$DATA/mat_list.txt"


module load NiaEnv/.2022a
module load intel/2022u2
export MKL_DIR=$MKLROOT
module load python

source $SCRATCH/.virtualenvs/end2end/bin/activate

NUM_THREAD=$THREADS
echo $NUM_THREAD
export OMP_NUM_THREADS=$NUM_THREAD
OMP_NUM_THREADS=$NUM_THREAD; export OMP_NUM_THREADS
MKL_NUM_THREADS=$NUM_THREAD; export MKL_NUM_THREADS
export MKL_DYNAMIC=FALSE;
export OMP_DYNAMIC=FALSE;

python GCN/gcn-training-example-pyg.py --hidden_channels $BCOL --threads $NUM_THREAD
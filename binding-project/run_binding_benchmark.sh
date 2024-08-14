#!/bin/bash



DATA_FOLDER=''
MAT_LIST=''
VIRTUALENV_DIR="$HOME/projects/fused-gnn/venv"
DATA_FOLDER="./data/ss-graphs/"
BCOL=32
THRD=20
while getopts ":t:c:m:v:" arg; do

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
    v)
      VIRTUALENV_DIR=$OPTARG
      ;;
    *) echo "Usage:"
      exit 0
  esac
done


MAT_LIST="$DATA_FOLDER/mat_list.txt"



#export MKL_DIR=$MKLROOT

source $VIRTUALENV_DIR/bin/activate

mkdir build
cd build || return

mkdir logs


TORCH_LIB="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
export TORCH_LIB
cmake -DCMAKE_PREFIX_PATH="$MKL_DIR/lib/intel64;$MKL_DIR/include;$MKL_DIR/../compiler/lib/intel64;_deps/openblas-build/lib/;$TORCH_LIB"  -DCMAKE_BUILD_TYPE=Release ..

make -j 16

cd ..

MKL_NUM_THREADS=$THRD; export MKL_NUM_THREADS
OMP_NUM_THREADS=$THRD; export OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE;
export OMP_DYNAMIC=FALSE;


HEADER=1
# shellcheck disable=SC2162
#while read line; do
#    mat="$DATA_FOLDER/$line"
#    if [ $HEADER -eq 1 ]; then
#      python binding_benchmark.py $mat $BCOL $THRD $HEADER > ./build/logs/benchmark_results.csv
#      HEADER=0
#    else
#      python binding_benchmark.py $mat $BCOL $THRD $HEADER >> ./build/logs/benchmark_results.csv
#    fi
#done < "${MAT_LIST}"

echo 'impl,matrix,time' > ./build/logs/e2e_$BCOL.csv

#python gcn-e2e-training/gcn-dgl.py --threads $THRD --hidden_channels $BCOL --dataset $DATA_FOLDER >> ./build/logs/e2e_$BCOL.csv
python gcn-e2e-training/fused-gcn-training.py --threads $THRD --hidden_channels $BCOL --dataset $DATA_FOLDER >> ./build/logs/e2e_$BCOL.csv
#python gcn-e2e-training/gcn-pyg.py --threads $THRD --hidden_channels $BCOL --dataset $DATA_FOLDER >> ./build/logs/e2e_$BCOL.csv
deactivate
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

module load NiaEnv/.2022a
module load intel/2022u2
export MKL_DIR=$MKLROOT
module load cmake
module load gcc

mkdir build
# shellcheck disable=SC2164
cd build
cmake -DCMAKE_PREFIX_PATH="$MKL_DIR/lib/intel64;$MKL_DIR/include;$MKL_DIR/../compiler/lib/intel64;_deps/openblas-build/lib/;/home/m/mmehride/kazem/programs/metis-5.1.0/libmetis;/home/m/mmehride/kazem/programs/metis-5.1.0/include/;"  -DCMAKE_BUILD_TYPE=Release ..
make
mkdir logs
cd ..
BINPATH=./build/example

THREADS=$1

NUM_THREAD=$THREADS
export OMP_NUM_THREADS=$NUM_THREAD
OMP_NUM_THREADS=$NUM_THREAD; export OMP_NUM_THREADS
export OMP_DYNAMIC=FALSE;

if ! [ -d ./pyg/data ]; then
 python ./scripts/pyg_data_exporter.py ./pyg
 echo "TEST"
fi
header=1

#for w in {500, 1000}; do
#      k=4
    if [ $header -eq 1 ]; then
      $BINPATH/gcn_demo -sm ./pyg/data/pubmed/PubMed.mtx -nt $THREADS -fm ./pyg/data/pubmed/features.mtx -ah -ip 500 > ./build/logs/gcn_demo.csv
      {
      $BINPATH/gcn_demo -sm ./pyg/data/pubmed_ordered/PubMed_Ordered.mtx -nt $THREADS -fm ./pyg/data/pubmed/features.mtx -ip 500
      $BINPATH/gcn_demo -sm ./pyg/data/cora/Cora.mtx -nt $THREADS -fm ./pyg/data/cora/features.mtx -ip 500
      $BINPATH/gcn_demo -sm ./pyg/data/cora_ordered/Cora_Ordered.mtx -nt $THREADS -fm ./pyg/data/cora/features.mtx -ip 500
      } >> ./build/logs/gcn_demo.csv
      echo ""
      header=0
    else
      {
      $BINPATH/gcn_demo -sm ./pyg/data/pubmed/PubMed.mtx -nt $THREADS -fm ./pyg/data/pubmed/features.mtx -ip $w
      $BINPATH/gcn_demo -sm ./pyg/data/pubmed_ordered/PubMed_Ordered.mtx -nt $THREADS -fm ./pyg/data/pubmed/features.mtx -ip $w
      $BINPATH/gcn_demo -sm ./pyg/data/cora/Cora.mtx -nt $THREADS -fm ./pyg/data/cora/features.mtx -ip $w
      $BINPATH/gcn_demo -sm ./pyg/data/cora_ordered/Cora_Ordered.mtx -nt $THREADS -fm ./pyg/data/cora/features.mtx -ip $w
      } >> ./build/logs/gcn_demo.csv
      echo ""
    fi
#done


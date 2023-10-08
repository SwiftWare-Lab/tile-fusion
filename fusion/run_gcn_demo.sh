#!/bin/bash
#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
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
while read line; do
  for sr in {0.1,0.4,0.7,1}; do
    for w in {50,100,250,500,1000,5000}; do
      echo "for $line $sr $w"
        if [ $header -eq 1 ]; then
          $BINPATH/gcn_demo -sm ./pyg/data/$line -nt $THREADS -ah -ip $w -sr $sr > ./build/logs/gcn_demo_$sr.csv
          header=0
        else
          $BINPATH/gcn_demo -sm ./pyg/data/$line -nt $THREADS -ip $w -sr $sr >> ./build/logs/gcn_demo_$sr.csv
        fi
    done
  done
done < ./pyg/data/mat_list.txt

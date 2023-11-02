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
DATA="./data/planetoid/graphs/"
MODE="GCNFusedSequential"
THREADS=8
BCOL=100

while getopts ":e:t:f:m:" arg; do

  case "${arg}" in
    e)
      MODE=$OPTARG
      ;;
    f)
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
module load cmake

mkdir build
# shellcheck disable=SC2164
cd build
cmake -DCMAKE_PREFIX_PATH="$MKL_DIR/lib/intel64;$MKL_DIR/include;$MKL_DIR/../compiler/lib/intel64;_deps/openblas-build/lib/;/home/m/mmehride/kazem/programs/metis-5.1.0/libmetis;/home/m/mmehride/kazem/programs/metis-5.1.0/include/;"  -DCMAKE_BUILD_TYPE=Release ..
make
mkdir logs
cd ..
BINPATH=./build/gcn

NUM_THREAD=$THREADS
echo $NUM_THREAD
export OMP_NUM_THREADS=$NUM_THREAD
OMP_NUM_THREADS=$NUM_THREAD; export OMP_NUM_THREADS
MKL_NUM_THREADS=$NUM_THREAD; export MKL_NUM_THREADS
export MKL_DYNAMIC=FALSE;
export OMP_DYNAMIC=FALSE;

if ! [ -d $DATA ]; then # if data folder does not exist(not applicable for experiments with banded matrices)
  mkdir $DATA
 python ./scripts/pyg_data_exporter.py ./pyg
 echo "TEST"
fi

if [ $MODE == "GCNFusedSequential" ]; then
  echo "Experiment: GCNFusedSequential"
  for sr in {0.1,0.4,0.7,1}; do
    header=1
    while read line; do
      echo $line
      for t in {4,8,16,32,64,128,256,512,1024}; do
        echo "for $line $sr $t"
          if [ $header -eq 1 ]; then
            $BINPATH/gcn_demo -sm $DATA/$line -nt $THREADS -ah -tn $t -sr $sr -bc $BCOL -en $MODE > ./build/logs/gcn_demo_$sr.csv
            header=0
          else
            $BINPATH/gcn_demo -sm $DATA/$line -nt $THREADS -tn $t -sr $sr -bc $BCOL -en $MODE >> ./build/logs/gcn_demo_$sr.csv
          fi
      done
    done < $MATLIST
  done
fi

if [ $MODE == "GCNFusedParallel" ]; then
  echo "Experiment: GCNFusedParallel"
#  for sr in {0.1,0.4,0.7,1}; do
   sr=1
   header=1
   while read line; do
     for w in {10,50,100,250,500,1000,2000,4000}; do
       for BCOL in {128,256,512,1024,2048}; do
         for EDIM in {4,8,32,64,128,256}; do
          echo "for $line $w $BCOL $EDIM"
          if [ $header -eq 1 ]; then
          $BINPATH/gcn_demo -sm $DATA/$line -nt $THREADS -ah -ip $w -sr $sr -bc $BCOL -ed $EDIM -en $MODE > ./build/logs/gcn_demo.csv
          header=0
          else
           $BINPATH/gcn_demo -sm $DATA/$line -nt $THREADS -ip $w -sr $sr -bc $BCOL -ed $EDIM -en $MODE >> ./build/logs/gcn_demo.csv
          fi
       done
     done
     done
   done < $MATLIST
#  done
fi

if [ $MODE == "GCNFusedBandedSpecific" ]; then
  echo "Experiment: GCNFusedBandedSpecific"
  header=1
  sr=1
  while read line; do
    echo $line
    for t in {4,8,16,32,64,128,256,512,1024}; do
      echo "for $line $sr $t"
        if [ $header -eq 1 ]; then
          $BINPATH/gcn_demo -sm $DATA/$line -nt $THREADS -ah -tn $t -sr $sr -bc $BCOL -en $MODE > ./build/logs/gcn_demo_$sr.csv
          header=0
        else
          $BINPATH/gcn_demo -sm $DATA/$line -nt $THREADS -tn $t -sr $sr -bc $BCOL -en $MODE >> ./build/logs/gcn_demo_$sr.csv
        fi
    done
  done < $MATLIST
fi

if [ $MODE == "GCNSingleLayerCompare" ]; then
  echo "Experiment: GCNSingleLayerCompare"
  header=1
  sr=1
  while read line; do
#    for BCOL in {500,1000,3000}; do
#      for ED in {8,32,64,128}; do
        for tn in {8,16,32,64,128,256,512,1024,2048,4096}; do
          echo "for $line $BCOL $ED $tn"
          if [ $header -eq 1 ]; then
            $BINPATH/gcn_layer_demo -sm $DATA/$line -nt $THREADS -tn $tn -ah -sr $sr -bc 1000 -ed 64 -en $MODE > ./build/logs/gcn_single_layer_demo.csv
            header=0
          else
            $BINPATH/gcn_layer_demo -sm $DATA/$line -nt $THREADS -tn $tn -sr $sr -bc 1000 -ed 64 -en $MODE >> ./build/logs/gcn_single_layer_demo.csv
          fi
        done
#      done
#    done
  done < $MATLIST
fi

if [ $MODE == "GCNWithDifferentFusionLevels" ]; then
 echo "Experiment: GCNWithDifferentFusionLevels"
  sr=1
  header=1
  while read line; do
    for BCOL in {500,3000}; do
      for EDIM in {8,32,64}; do
        for tn in {16,32,64,128,256,512,1024,2048,4096}; do
          echo "for $line $BCOL $EDIM $tn"
          if [ $header -eq 1 ]; then
            $BINPATH/gcn_demo -sm $DATA/$line -nt $THREADS -tn $tn -ah -sr $sr -bc $BCOL -en $MODE -ed $EDIM > ./build/logs/gcn_demo.csv
            header=0
          else
            $BINPATH/gcn_demo -sm $DATA/$line -nt $THREADS -tn $tn -sr $sr -bc $BCOL -en $MODE -ed $EDIM >> ./build/logs/gcn_demo.csv
          fi
        done
      done
    done
  done < $MATLIST
fi

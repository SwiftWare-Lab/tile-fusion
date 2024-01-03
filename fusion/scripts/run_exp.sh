#!/bin/sh

BINLIB=$1
PATHMAIN=$2
TUNED=$3
THRDS=$4
MATLIST=$5
BCOL=$6
LOGS=./build/logs/


#echo $BINLIB $PATHMAIN
#load module intel
export OMP_NUM_THREADS=$THRDS
export MKL_NUM_THREADS=$THRDS

header=1

# shellcheck disable=SC2039
if [ "$TUNED" ==  1 ]; then
  while read line; do
    mat=$line
    k=4
    if [ $header -eq 1 ]; then
      $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -ah -bc $BCOL -ip 1000 > $LOGS/spmv_spmv_$BCOL.csv
      echo "TEST1"
      header=0
    else
      $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -bc $BCOL -ip 1000 >> $LOGS/spmv_spmv_$BCOL.csv
      echo "TEST2"
    fi
  done < ${MATLIST}
fi



# shellcheck disable=SC2039
if [ "$TUNED" ==  2 ]; then
  while read line; do
    mat=$line
    # shellcheck disable=SC2039
    for w in {10,50,100,1000,5000}; do
      k=4
      if [ $header -eq 1 ]; then
        $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -ah -bc $BCOL -ip $w > $LOGS/spmv_spmv_$BCOL.csv
        echo ""
        header=0
      else
        $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -bc $BCOL -ip $w >> $LOGS/spmv_spmv_$BCOL.csv
        echo ""
      fi
    done
  done < ${MATLIST}
fi


# shellcheck disable=SC2039
if [ "$TUNED" ==  3 ]; then
  while read line; do
    mat=$line
    # shellcheck disable=SC2039
#    for w in {100,1000,5000,10000,500000}; do
      k=4
      for ntile in {4,8,16,32,64,128,256,512,1024,2048,4096,8192}; do
#        if [ $ntile -gt $BCOL ]; then
#          continue
#        fi
      echo "for $line $BCOL $w $ntile"
      if [ $header -eq 1 ]; then
        $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -ah -bc $BCOL -ip $ntile -tm $ntile -tn $ntile > $LOGS/spmv_spmv_$BCOL.csv
        header=0
      else
        $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -bc $BCOL -ip $ntile -tm $ntile -tn $ntile >> $LOGS/spmv_spmv_$BCOL.csv
      fi
      done
#    done
  done < ${MATLIST}
fi

# shellcheck disable=SC2039
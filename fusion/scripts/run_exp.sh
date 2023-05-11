#!/bin/sh

BINLIB=$1
PATHMAIN=$2
TUNED=$3
THRDS=$4
MATLIST=$5
BCOL=$6


#echo $BINLIB $PATHMAIN
#load module intel
export OMP_NUM_THREADS=$THRDS
export MKL_NUM_THREADS=$THRDS

header=1

if [ "$TUNED" ==  1 ]; then
  while read line; do
    mat=$line
    k=4
    $BINLIB  $PATHMAIN/$mat $THRDS $k $k $header $BCOL
    echo ""
    if [ $header -eq 1 ]; then
       header=0
    fi
  done < ${MATLIST}
fi
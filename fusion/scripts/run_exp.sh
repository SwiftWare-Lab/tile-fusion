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

# shellcheck disable=SC2039
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



# shellcheck disable=SC2039
if [ "$TUNED" ==  2 ]; then
  while read line; do
    mat=$line
    # shellcheck disable=SC2039
    for w in {10,50,100,1000,5000}; do
      k=4
      $BINLIB  $PATHMAIN/$mat $THRDS $k $k $header $BCOL $w
      echo ""
      if [ $header -eq 1 ]; then
         header=0
      fi
    done
  done < ${MATLIST}
fi


# shellcheck disable=SC2039
if [ "$TUNED" ==  3 ]; then
  while read line; do
    mat=$line
    # shellcheck disable=SC2039
    for w in {100,1000,5000,10000,500000}; do
      k=4
      for ntile in {8,16,32,64,128}; do
        if [ $ntile -gt $BCOL ]; then
          continue
        fi
      $BINLIB  $PATHMAIN/$mat $THRDS $k $k $header $BCOL $w $ntile
      echo ""
      if [ $header -eq 1 ]; then
         header=0
      fi
      done
    done
  done < ${MATLIST}
fi
#!/bin/sh

BINLIB=$1
PATHMAIN=$2
TUNED=$3
THRDS=$4
MATLIST=$5
BCOL=$6
LOGS=$7
ID=$8

#echo $BINLIB $PATHMAIN
#load module intel
export OMP_NUM_THREADS=$THRDS
export MKL_NUM_THREADS=$THRDS

header=1

if [ -z $ID ]; then
  OUTPUT_FILE=$LOGS/spmv_spmv_$BCOL.csv
else
  OUTPUT_FILE=$LOGS/spmv_spmv_${BCOL}_${ID}.csv
fi

# shellcheck disable=SC2039
if [ "$TUNED" ==  1 ]; then
  while read line; do
    mat=$line
    k=4
    if [ $header -eq 1 ]; then
      $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -ah -bc $BCOL -ip 1000 > $OUTPUT_FILE
      echo "TEST1"
      header=0
    else
      $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -bc $BCOL -ip 1000 >> $OUTPUT_FILE
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
        $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -ah -bc $BCOL -ip $w > $OUTPUT_FILE
        echo ""
        header=0
      else
        $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -bc $BCOL -ip $w >> $OUTPUT_FILE
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
#      for ntile in {15000,32000,100000,500000,1000000,2600000}; do
        for ntile in {8,16,32,64,128,256,512,1024,2048,4096}; do
#        if [ $ntile -gt $BCOL ]; then
#          continue
#        fi
      echo "for $line $BCOL $w $ntile"
      if [ $header -eq 1 ]; then
        $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -ah -bc $BCOL -ip $ntile -tm 1000000 -tn 32 -ed $BCOL > $OUTPUT_FILE
        header=0
      else
        $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -bc $BCOL -ip $ntile -tm 1000000 -tn 32 -ed $BCOL >> $OUTPUT_FILE
      fi
      done
#    done
  done < ${MATLIST}
fi

if [ "$TUNED" ==  4 ]; then
  while read line; do
    tokens=$(echo $line | tr "," "\n")
    mat=$(echo $tokens | awk '{print $1}')
    ntile=$(echo $tokens | awk '{print $2}')
      echo "for $mat $ntile"
      if [ $header -eq 1 ]; then
        $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -ah -bc $BCOL -ip $ntile -tm $ntile -tn 64 > $OUTPUT_FILE
        header=0
      else
        $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -bc $BCOL -ip $ntile -tm $ntile -tn 64 >> $OUTPUT_FILE
      fi
  done < ${MATLIST}
fi

if [ "$TUNED" == 5 ]; then
    while read line; do
       mat=$line
        echo "for $mat $ntile"
        if [ $header -eq 1 ]; then
          $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -ah -bc $BCOL > $OUTPUT_FILE
          header=0
        else
          $BINLIB  -sm $PATHMAIN/$mat -nt $THRDS -bc $BCOL >> $OUTPUT_FILE
        fi
    done < ${MATLIST}
  fi

# shellcheck disable=SC2039
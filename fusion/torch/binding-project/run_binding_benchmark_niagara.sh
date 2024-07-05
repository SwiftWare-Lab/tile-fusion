#!/bin/bash



DATA_FOLDER=''
MAT_LIST=''

DATA_FOLDER="./data/ss-graphs/"
BCOL=32
THRD=20
while getopts ":t:c:m:" arg; do

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
    *) echo "Usage:"
      exit 0
  esac
done



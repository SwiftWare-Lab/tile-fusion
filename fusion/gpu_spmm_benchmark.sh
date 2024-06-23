#!/bin/bash

EXP=$1
bash run.sh -t 1 -m ./data/ss-graphs -c 32 -e $EXP
bash run.sh -t 1 -m ./data/ss-graphs -c 64 -e $EXP
bash run.sh -t 1 -m ./data/ss-graphs -c 128 -e $EXP
#bash run.sh -t 1 -m ./data/ss-graphs -c 256 -e $EXP
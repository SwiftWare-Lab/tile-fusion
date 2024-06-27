#!/bin/bash

EXP=$1
bash run.sh -t 1 -m ./data/end2end-test-data -c 32 -e $EXP
bash run.sh -t 1 -m ./data/end2end-test-data -c 64 -e $EXP
bash run.sh -t 1 -m ./data/end2end-test-data -c 128 -e $EXP
#bash run.sh -t 1 -m ./data/ss-graphs -c 256 -e $EXP
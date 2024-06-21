#!/bin/bash

bash run.sh -t 1 -m ./data/ss-graphs -c 32 -e gpu_spmm
bash run.sh -t 1 -m ./data/ss-graphs -c 64 -e gpu_spmm
bash run.sh -t 1 -m ./data/ss-graphs -c 128 -e gpu_spmm
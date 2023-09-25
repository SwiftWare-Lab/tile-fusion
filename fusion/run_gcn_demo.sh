#!/bin/bash

mkdir build
# shellcheck disable=SC2164
cd build
cmake ..
make
mkdir logs
cd ..
BINPATH=./build/example

if ! [ -d ./pyg/data ]; then
 python ./scripts/pyg_data_exporter.py ./pyg
 echo "TEST"

fi
$BINPATH/gcn_demo -sm ./pyg/data/cora/Cora.mtx -nt 8 -fm ./pyg/data/cora/features.mtx -ah > ./build/logs/gcn_demo.csv

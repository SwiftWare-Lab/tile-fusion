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
$BINPATH/gcn_demo -sm ./pyg/data/pubmed/PubMed.mtx -nt 8 -fm ./pyg/data/pubmed/features.mtx -ah -ip 1000 > ./build/logs/gcn_demo.csv
{
$BINPATH/gcn_demo -sm ./pyg/data/pubmed_ordered/PubMed_Ordered.mtx -nt 8 -fm ./pyg/data/pubmed/features.mtx -ip 1000
$BINPATH/gcn_demo -sm ./pyg/data/cora/Cora.mtx -nt 8 -fm ./pyg/data/cora/features.mtx -ip 1000
$BINPATH/gcn_demo -sm ./pyg/data/cora_ordered/Cora_Ordered.mtx -nt 8 -fm ./pyg/data/cora/features.mtx -ip 1000
} >> ./build/logs/gcn_demo.csv
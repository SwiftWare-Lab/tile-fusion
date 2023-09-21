#!/bin/bash

if ! [ -d ./fusion/pyg/data ]; then
 cd ../modeling
 python pyg_data_exporter.py
 cd ../fusion
fi
./gcn_demo -sm ./pyg/data/cora/Cora.mtx -nt 8 -fm ./pyg/data/cora/features.mtx -ah

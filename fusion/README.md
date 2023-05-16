# Fused GNN
We explore possibilities of inter-layer fusion in GNN on hetergenous devices, CPU-GPU

## Requirements
- Python 3.6 for plotting
- CMake
- C++ compiler


## Build
```bash
git clone --recursive https://github.com/SwiftWare-Lab/fused-gnn.git
cd fused-gnn/fusion
mkdir build
cd build
cmake ..
make
```

## Download dataset
You also need to download the data where you like. The default location 
is set to data in the current directory.
```bash
python scripts/dl_matrix.py
```
The script downloads SPD matrices. It can be changed by updating the `dl_matrix.py`. 

## On Niagara
 You can run the script using:
```bash
bash run_niagara.sh 1
```
or you can use:
```bash
sbatch run_niagara.sh 1
```
to run as job. The passed argument `1` specifies whether the matrices should be downloaded or not. 

The `run_niagara.sh` script should work in a linux machine. Make sure to set paths before running the script.

## Plotting

After running the `run_niagara.sh` script, you can plot the results using the `plot.py` script.

```bash
python plot.py where/spmv_spmv_4.csv
```
The plot should be modified to save a graph


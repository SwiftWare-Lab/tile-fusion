# Fused GNN
We explore possibilities of inter-layer fusion in GNN on hetergenous devices, CPU-GPU

## Requirements
- Python 3.6 for plotting
- CMake
- C++ compiler

## Build
```bash
mkdir build
cd build
cmake ..
make
```

## On Niagara
```bash
bash run_niagara.sh
```
or you can use:
```bash
sbatch run_niagara.sh
```
to run as job.

The `run_niagara.sh` script should work in a linux machine.

## Plotting

After running the `run_niagara.sh` script, you can plot the results using the `plot.py` script.

```bash
python plot.py where/spmv_spmv_4.csv
```
The plot should be modified to save a graph


# Fused GNN

We explore possibilities of inter-layer fusion in GNN on hetergenous devices, CPU-GPU

### Requirements

- Python 3.6 for plotting
- CMake
- C++ compiler

### Build

```bash
git clone --recursive https://github.com/SwiftWare-Lab/fused-gnn.git
cd fused-gnn/fusion
mkdir build
cd build
cmake ..
make
```

## SpMM SpMM Fusion

### Download dataset

You also need to download the data where you like. The default location
is set to data in the current directory.

```bash
python scripts/dl_matrix.py
```

The script downloads SPD matrices. It can be changed by updating the `dl_matrix.py`.

### On Local

you can build and run an experiment using:

```bash
bash run.sh -l -b SpMM_SpMM_MKL -d 4
```

learn about different parameters by running:

```bash
bash run.sh -h
```

### On Niagara

You can test the script on niagara using:

```bash
bash run_niagara.sh -b SpMM_SpMM_Demo_UnFusedParallel -l
```

or you can use:

```bash
sbatch run_niagara.sh -b SpMM_SpMM_Demo_UnFusedParallel
```

note that in this method you should have run dl_matrix.py before:

```bash
python scripts/dl_matrix.py $SCRATCH/UFDB/SPD/ scripts/mat_list.txt
```

### Plotting

After running the `run_niagara.sh` script, you can plot the results using the `plot.py` script.

```bash
python plot.py where/spmv_spmv_4.csv
```

The plot should be modified to save a graph

## GCN Fusion

### experiment on tri-banded Matrices

First you need to create matrices using this script

```bash
python scripts/gen_matrix_folder.py -sl 1000 5000 10000 100000 -f ./data/tri-banded -b 3
```

### experiment on torch-geometric Graph Adjacency Matrices

First you need to download matrices using this script

```bash
python scripts/dl_pyg_graphs.py data/pyg-graphs
```

Make sure you have mkl installed and setup its environment variables.
Then run the code using:

```bash
bash run_gcn_demo.sh -t 8 -m ./data/tri-banded -e GCNSingleLayerCompare
```

set the thread number using -t, matrices folder using -m, and name of experiment using -e.
Some example experiments:

- GCNSingleLayerCompare
- GCNWithDifferentFusionLevels

for plotting experiment you can use the plot_gcn.py script in script/plotter folder:

```bash
python scripts/plotter/plot_gcn.py ~/where/logs/ scripts/plotter/single_layer_config.yaml
```

First argument is the folder containing the logs and the second argument is the config file. There is a sample config 
file for single layer experiments in the script/plotter folder. You can generate config file for different experiments 
based on the sample config file.
Plots will be stored in the same folder as the log folder.


## Profiling with PAPI
You will need to build the code with PAPI support. Please check the 
swiftware benchmark repo for details. The 'test_papi.sh' script provides
an example for running papi.

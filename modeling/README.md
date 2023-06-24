# Modeling

This part of the code contain python models.

## SAGEGraph-modeling

this is a simple model of SAGEGraph network.

### Requirements
For running this project you need to install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

then you can install project environment using:
```
conda env create -f environment.yml
```

and activate it using:
```
conda activate fused-gnn
```
### Run the Training process for cora(default) or pubmed dataset
```
python SAGEGraph-GCN/train.py -d cora
```


## Reference PyTorch GraphSAGE Implementation
### Author: William L. Hamilton


Basic reference PyTorch implementation of [GraphSAGE](https://github.com/williamleif/GraphSAGE).
This reference implementation is not as fast as the TensorFlow version for large graphs, but the code is easier to read and it performs better (in terms of speed) on small-graph benchmarks.
The code is also intended to be simpler, more extensible, and easier to work with than the TensorFlow version.

Currently, only supervised versions of GraphSAGE-mean and GraphSAGE-GCN are implemented. 

#### Requirements

pytorch >0.2 is required.

#### Running examples

Execute `python -m graphsage.model` to run the Cora example.
It assumes that CUDA is not being used, but modifying the run functions in `model.py` in the obvious way can change this.
There is also a pubmed example (called via the `run_pubmed` function in model.py).

Execute `python gcn_layer.py` for an example of one layer of GCN. 

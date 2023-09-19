# Modeling

This part of the code contain python models.

## Requirements
For running this project you need to use virtualenv with base python>3.8.
After this install every requirement with pip.
(this part will be completed soon)

## GraphSAGE-modeling

this is a simple model of GraphSAGE GNN.
### Run the Training process for cora(default) or pubmed dataset
```
python graphsage/train.py -d pubmed
```

## GCN-modeling

this is a simple model of GCN.
### Run the Training process for cora(default) or pubmed dataset
```
python GCN/train.py -d pubmed
```
(this only runs the custom model now which is using simple gcn_layer)
## Reference PyTorch GraphSAGE Implementation
### Author: William L. Hamilton


Basic reference PyTorch implementation of [GraphSAGE](https://github.com/williamleif/GraphSAGE).
This reference implementation is not as fast as the TensorFlow version for large graphs, but the code is easier to read and it performs better (in terms of speed) on small-graph benchmarks.
The code is also intended to be simpler, more extensible, and easier to work with than the TensorFlow version.

Currently, only supervised versions of GraphSAGE-mean and GraphSAGE-GCN are implemented. 

#### Requirements

pytorch >0.2 is required.

#### Running examples
```
python graphsage/reference/model.py
```
It assumes that CUDA is not being used, but modifying the run functions in `model.py` in the obvious way can change this.
There is also a pubmed example (called via the `run_pubmed` function in model.py).

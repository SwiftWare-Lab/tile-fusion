
import torch
from torch_geometric.datasets import Planetoid
# from SpMMProfiler import SpMMProfiler
import sys
import time
import argparse
from pathlib import Path
import os

datasets = {
    'cora' : {
        'root': 'data/cora','name': 'Cora', 'batch_count': 256, 'epoch_num' : 100
    },
    'pubmed' : {
        'root':'data/pubmed', 'name':'PubMed', 'batch_count': 1024, 'epoch_num' : 200
    }
}
def get_coordinate_str(index, edge_index) -> str:
    return '{} {}\n'.format(edge_index[0][index].item(),edge_index[1][index].item())

def export_adjacency_matrices():
    for _, v in datasets.items():
        dataset = Planetoid(root=v['root'], name=v['name'])
        data = dataset[0]
        rows = list(data.x.shape)[0]
        cols = rows
        nnz = list(data.edge_index.shape)[1]
        coordinates = [get_coordinate_str(i, data.edge_index) for i in range(nnz)]
        data_folder = os.path.join('..', 'fusion', 'pyg', v['root'])
        os.makedirs(data_folder, exist_ok=True)
        file_path = os.path.join(data_folder, v['name'] + '.mtx')
        with open(file_path , 'w') as mtx_file:
            mtx_file.write("%%MatrixMarket matrix coordinate pattern general\n")
            mtx_file.write('{} {} {}\n'.format(rows, cols, nnz))
            mtx_file.writelines(coordinates)


def export_features():
    for _, v in datasets.items():
        dataset = Planetoid(root=v['root'], name=v['name'])
        data = dataset[0]
        features_shape = list(data.x.shape)
        rows = features_shape[0]
        cols = features_shape[1]
        values = [str(data.x[i][j].item())+'\n' for j in range(cols) for i in range(rows)]
        data_folder = os.path.join('..', 'fusion', 'pyg', v['root'])
        os.makedirs(data_folder, exist_ok=True)
        file_path = os.path.join(data_folder, 'features' + '.mtx')
        with open(file_path , 'w') as mtx_file:
            mtx_file.write("%%MatrixMarket matrix array real general\n")
            mtx_file.write('{} {}\n'.format(rows, cols))
            mtx_file.writelines(values)


if __name__ == '__main__':
    export_features()
    export_adjacency_matrices()
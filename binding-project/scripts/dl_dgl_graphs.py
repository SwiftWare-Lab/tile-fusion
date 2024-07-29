import dgl.data as datasets
import os
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from scipy.sparse.csgraph import reverse_cuthill_mckee
from torch_sparse import coalesce
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
import sys

#def get_coordinate_str(index, edge_index) -> str:
#    return '{} {}\n'.format(edge_index[0][index].item() + 1, edge_index[1][index].item() + 1)

os.environ["OMP_NUM_THREADS"] = "4"
folder_name = sys.argv[1]
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
raw_folder_name = folder_name + '/raw'

datasets = [
    # (datasets.AMDataset(), 'AM'),
    (datasets.BGSDataset(), 'BGS')
]

matrix_list_file_name = folder_name + '/mat_list.txt'


for dataset in datasets:
    with open(matrix_list_file_name, 'a') as f:
        name = dataset[1]
        graph = dataset[0][0]
        category = dataset[0].predict_category
        # print(category)
        adj_matrices = {}
        for etype in graph.etypes:
            adj_matrices[etype] = graph.adjacency_matrix(etype=etype)

        # Determine the sizes of the node sets
        node_types = graph.ntypes
        num_nodes = {ntype: graph.num_nodes(ntype) for ntype in node_types}

        # Create a block matrix with dimensions based on the number of nodes
        total_nodes = sum(num_nodes.values())
        block_matrices = []
        for etype in graph.etypes:
            src_type, _, dst_type = etype
            num_src = num_nodes[src_type]
            num_dst = num_nodes[dst_type]

            # Create a block matrix for this edge type
            adj_coo = adj_matrices[etype]
            # Initialize a zero matrix of the appropriate size
            block_matrix = sp.lil_matrix((total_nodes, total_nodes))
            # Insert the adjacency matrix for this edge type into the appropriate block
            block_matrix[:num_src, num_src:num_src+num_dst] = adj_coo.tocoo()
            block_matrices.append(block_matrix)

        # Combine all block matrices into a single sparse matrix
        full_matrix = sp.bmat(block_matrices, format='coo')

        # Convert to CSR format if needed
        # full_matrix_csr = full_matrix.tocsr()

        # export the full sparse matrix
        mat_folder = folder_name + '/' + name
        mmwrite(mat_folder + '/{}.mtx'.format(name), full_matrix)



        # rows = list(data.x.shape)[0]
        # cols = rows
        # nnz = list(data.edge_index.shape)[1]
        # mat_folder = folder_name + '/' + name
        # if not os.path.exists(mat_folder):
        #     os.mkdir(mat_folder)
        # # print(data.edge_attrs)
        # # print(data.edge_index)
        # coo = coalesce(data.edge_index, None, data.x.size(0), data.x.size(0))
        # coo = coo[0]
        # print("----Ordering and writing {} to mtx file-----".format(name))
        # adj = csr_matrix((np.ones(coo[0].shape[0]), (coo[0].numpy(), coo[1].numpy())))
        # features = data.x.numpy()
        # labels = data.y.numpy()
        # if len(labels.shape) > 1:
        #     labels = labels[:, 0]
        # print(features.shape)
        # labels = labels[np.newaxis, ...]
        # print(labels.shape)
        # mmwrite(mat_folder + '/features.mtx', features)
        # mmwrite(mat_folder + '/labels.mtx', labels)
        # mmwrite(mat_folder + '/{}.mtx'.format(name), adj)
        # f.write(name + '/' + name + '.mtx' + '\n')
        # perm = reverse_cuthill_mckee(adj)
        # adj = adj[perm, :][:, perm]
        # features = features[perm, :]
        # labels = labels[:, perm]
        # name = name + '_ordered'
        # mat_folder = folder_name + '/' + name
        # if not os.path.exists(mat_folder):
        #     os.mkdir(mat_folder)
        # mmwrite(mat_folder + '/features.mtx', features)
        # mmwrite(mat_folder + '/labels.mtx', labels)
        # mmwrite(mat_folder + '/{}.mtx'.format(name), adj)
        # f.write(name + '/' + name + '.mtx' + '\n')




        # mmwrite(mat_folder + '/{}_ordered.mtx'.format(name), adj)
        # f.write(mat_folder + '/{}_ordered.mtx\n'.format(name))
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
        g = dataset[0][0]
        category = dataset[0].predict_category
        # Extract adjacency matrices for each canonical edge type
        adj_matrices = {}
        for canonical_etype in g.canonical_etypes:
            adj_matrices[canonical_etype] = g.adjacency_matrix(etype=canonical_etype)

        # Determine the sizes of the node sets
        node_types = g.ntypes
        num_nodes = {ntype: g.num_nodes(ntype) for ntype in node_types}

        # Create a block matrix with dimensions based on the number of nodes
        # Initialize a zero matrix of the appropriate size
        total_nodes = sum(num_nodes.values())
        block_matrix = sp.lil_matrix((total_nodes, total_nodes))

        # Track the starting index for each node type in the block matrix
        node_type_start_idx = {}
        current_idx = 0
        for ntype in node_types:
            node_type_start_idx[ntype] = current_idx
            current_idx += num_nodes[ntype]

        # Populate the block matrix with the adjacency matrices
        for canonical_etype, adj_coo in adj_matrices.items():
            src_type, _, dst_type = canonical_etype
            src_start_idx = node_type_start_idx[src_type]
            dst_start_idx = node_type_start_idx[dst_type]
            adj_coo = adj_coo.coo()
            coo = coalesce(adj_coo, None, num_nodes[src_type], num_nodes[src_type])
            coo = coo[0]
            # Insert the adjacency matrix into the block matrix
            block_matrix[src_start_idx:src_start_idx+num_nodes[src_type],
            dst_start_idx:dst_start_idx+num_nodes[dst_type]] = coo.toarray()

        # Combine all block matrices into a single sparse matrix
        full_matrix = sp.bmat(block_matrix, format='coo')

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
import torch_geometric.datasets as datasets
import os
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


folder_name = sys.argv[1]
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
raw_folder_name = folder_name + '/raw'

datasets = [
    datasets.Coauthor(root=raw_folder_name + '/coauthor_cs/', name='CS', transform=None),
    datasets.Coauthor(root=raw_folder_name + '/coauthor_physics/', name='Physics', transform=None),
    datasets.CoraFull(root=raw_folder_name + '/cora_full/', transform=None),
    datasets.Flickr(root=raw_folder_name + '/flickr/', transform=None),
    datasets.Yelp(root=raw_folder_name + '/yelp/', transform=None),
    #datasets.Planetoid(root=raw_folder_name + '/planetoid/pubmed/', name='Pubmed', transform=None),
    #datasets.Planetoid(root=raw_folder_name + '/planetoid/cora/', name='Cora', transform=None),
    datasets.GitHub(root=raw_folder_name + '/github/', transform=None),
    datasets.FacebookPagePage(root=raw_folder_name + '/facebook_page_page/', transform=None),
    datasets.DeezerEurope(root=raw_folder_name + '/deezer_europe/', transform=None),
    datasets.Reddit2(root=raw_folder_name + '/reddit2/', transform=None)
]

matrix_list_file_name = folder_name + '/mat_list.txt'


for dataset in datasets:
    with open(matrix_list_file_name, 'a') as f:
        name = dataset.root.split('/')[-1]
        data = dataset[0]
        rows = list(data.x.shape)[0]
        cols = rows
        nnz = list(data.edge_index.shape)[1]
        mat_folder = folder_name + '/' + name
        if not os.path.exists(mat_folder):
            os.mkdir(mat_folder)
        # print(data.edge_attrs)
        # print(data.edge_index)
        coo = coalesce(data.edge_index, None, data.x.size(0), data.x.size(0))
        coo = coo[0]
        print("----Ordering and writing {} to mtx file-----".format(name))
        adj = csr_matrix((np.ones(coo[0].shape[0]), (coo[0].numpy(), coo[1].numpy())))
        features = data.x.numpy()
        labels = data.y.numpy()
        if len(labels.shape) > 1:
            labels = labels[:, 0]
        print(features.shape)
        labels = labels[np.newaxis, ...]
        print(labels.shape)
        mmwrite(mat_folder + '/features.mtx', features)
        mmwrite(mat_folder + '/labels.mtx', labels)
        mmwrite(mat_folder + '/{}.mtx'.format(name), adj)
        f.write(name + '/' + name + '.mtx' + '\n')
        perm = reverse_cuthill_mckee(adj)
        adj = adj[perm, :][:, perm]
        features = features[perm, :]
        labels = labels[:, perm]
        name = name + '_ordered'
        mat_folder = folder_name + '/' + name
        if not os.path.exists(mat_folder):
            os.mkdir(mat_folder)
        mmwrite(mat_folder + '/features.mtx', features)
        mmwrite(mat_folder + '/labels.mtx', labels)
        mmwrite(mat_folder + '/{}.mtx'.format(name), adj)
        f.write(name + '/' + name + '.mtx' + '\n')
        # mmwrite(mat_folder + '/{}_ordered.mtx'.format(name), adj)
        # f.write(mat_folder + '/{}_ordered.mtx\n'.format(name))

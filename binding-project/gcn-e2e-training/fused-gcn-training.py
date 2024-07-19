import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from scipy.io import mmread
from torch_geometric.datasets import Planetoid
import torch_geometric.datasets as datasets
from torch_geometric.logging import init_wandb, log
from torch_geometric import utils as pygUtils
from torch_geometric.nn import GCNConv
import numpy as np
import os

from FusedGCNModule import FusedGCN


# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels,
#                              bias=False)
#         self.conv2 = GCNConv(hidden_channels, out_channels, bias=False)
#         self.conv1_time = 0
#         self.conv2_time = 0
#
#     def forward(self, x, edge_index, edge_weight=None):
#         # x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv1(x, edge_index, edge_weight)
#         # for param in self.conv1.parameters():
#         #     mmwrite('weight1.mtx', param.data.detach().numpy())
#         x = x.relu()
#         # mmwrite('output1.mtx', x.detach().numpy())
#         # x = F.dropout(x, p=0.5, training=self.training)x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         # for param in self.conv2.parameters():
#         #     mmwrite('weight2.mtx', param.data.detach().numpy())
#         # mmwrite('output2.mtx',x.detach().numpy() )
#
#         return x

def convert_scipy_coo_to_torch_csr(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    edge_index = torch.IntTensor(indices)
    # print(edge_index)
    v = torch.FloatTensor(values)

    csr = pygUtils.to_torch_csr_tensor(edge_index, v)
    return csr

def train():
    model.train()
    optimizer.zero_grad()
    out = model(feature)
    # print(out)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./data')
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
device = torch.device('cpu')
torch.set_num_threads(args.threads)
train_mask = range(200)
# init_wandb(
#     name=f'GCN-{args.dataset}',
#     lr=args.lr,
#     epochs=args.epochs,
#     hidden_channels=args.hidden_channels,
#     device=device,
# )
mat_file_path = args.dataset + '/mat_list.txt'
with open(mat_file_path) as mat_file:
    matrices = mat_file.readlines()
    for mat in matrices:
        mat = mat.rstrip()
        datafolder = args.dataset
        mat_folder = mat.split('/')[0]
        adj_path = os.path.join(args.dataset, mat)
        feature_path = os.path.join(args.dataset, mat_folder, 'features.mtx')
        labels_path = os.path.join(args.dataset, mat_folder, 'labels.mtx')
        adj = convert_scipy_coo_to_torch_csr(mmread(adj_path))
        feature = torch.from_numpy(mmread(feature_path).astype(np.float32))
        if feature.size(1) > 128:
            feature = feature[:, :128]
        labels = torch.from_numpy(mmread(labels_path).astype(np.int64))
        labels = torch.squeeze(labels)
        name = mat_folder
        print(mat_folder)
        # if args.use_gdc:
        #     transform = T.GDC(
        #         self_loop_weight=1,
        #         normalization_in='sym',
        #         normalization_out='col',
        #         diffusion_kwargs=dict(method='ppr', alpha=0.05),
        #         sparsification_kwargs=dict(method='topk', k=128, dim=0),
        #         exact=True,
        #     )
        #     data = transform(data)

        # adj = pygUtils.to_torch_csr_tensor(data.edge_index)
        # crow_indices = adj.crow_indices().type(torch.int32)
        # col_indices = adj.col_indices().type(torch.int32)
        # adj = torch.sparse_csr_tensor(crow_indices, col_indices, adj.values())
        num_classes = len(np.unique(labels))
        model = FusedGCN(
            feat_dim=feature.size(1),
            embed_dim=args.hidden_channels,
            num_classes=num_classes,
            adj=adj,
            num_threads=args.threads,
        ).to(device)

        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=args.lr)  # Only perform weight-decay on first convolution.

        # @torch.no_grad()
        # def test():
        #     model.eval()
        #     pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)
        #
        #     accs = []
        #     for mask in [data.train_mask, data.val_mask, data.test_mask]:
        #         accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        #     return accs

        best_val_acc = test_acc = 0
        times = []
        for epoch in range(0, 100):
            start = time.time()
            loss1 = train()
            times.append(time.time() - start)
            # log(Epoch=epoch, Loss=loss1)
        # print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
        print(f'FusedGCNConv,{name},{torch.tensor(times).sum():.4f}')

    # print('total conv1 time: ', model.conv1_time)
    # print('total conv2 time: ', model.conv2_time)

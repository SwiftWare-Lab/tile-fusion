import argparse
import os.path as osp
import time

import dgl
import torch
import torch.nn.functional as F
import os
import torch_geometric.transforms as T
from torch_geometric import utils as pygUtils
import torch_geometric.datasets as datasets
from dgl.nn.pytorch.conv import GraphConv
from torch_geometric.utils.convert import to_dgl
import numpy as np
from scipy.io import mmread

class DGLGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels,
                               bias=False)
        self.conv2 = GraphConv(hidden_channels, out_channels, bias=False)
        self.conv1_time = 0
        self.conv2_time = 0

    def forward(self, x, edge_index, edge_weight=None):
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(edge_index, x)
        # for param in self.conv1.parameters():
        #     mmwrite('weight1.mtx', param.data.detach().numpy())
        x = x.relu()
        # mmwrite('output1.mtx', x.detach().numpy())
        # x = F.dropout(x, p=0.5, training=self.training)x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(edge_index,x)
        # for param in self.conv2.parameters():
        #     mmwrite('weight2.mtx', param.data.detach().numpy())
        # mmwrite('output2.mtx',x.detach().numpy() )

        return x


def train():
    model.train()
    optimizer.zero_grad()
    out = model(feature, adj)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./data')
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
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
# raw_folder_name = osp.join(osp.dirname(osp.realpath(__file__)), '../../modeling', 'data')
# dataset_list = [
#     datasets.Coauthor(root=raw_folder_name + '/coauthor_cs/', name='CS', transform=None),
#     datasets.Coauthor(root=raw_folder_name + '/coauthor_physics/', name='Physics', transform=None),
#     datasets.CoraFull(root=raw_folder_name + '/cora_full/', transform=None),
#     datasets.Flickr(root=raw_folder_name + '/flickr/', transform=None),
#     datasets.Yelp(root=raw_folder_name + '/yelp/', transform=None),
#     # datasets.Planetoid(root=raw_folder_name + '/planetoid/pubmed/', name='Pubmed', transform=None),
#     datasets.GitHub(root=raw_folder_name + '/github/', transform=None),
#     datasets.FacebookPagePage(root=raw_folder_name + '/facebook_page_page/', transform=None),
#     datasets.DeezerEurope(root=raw_folder_name + '/deezer_europe/', transform=None),
#     datasets.Reddit(root=raw_folder_name + '/reddit2/', transform=None)
# ]
# for dataset in dataset_list:
#     name = dataset.root.split('/')[-1]
#     data = dataset[0].to(device)
#     dgl_graph = to_dgl(data)
#     dgl_graph = dgl.add_self_loop(dgl_graph)
#     if args.use_gdc:
#         transform = T.GDC(
#             self_loop_weight=1,
#             normalization_in='sym',
#             normalization_out='col',
#             diffusion_kwargs=dict(method='ppr', alpha=0.05),
#             sparsification_kwargs=dict(method='topk', k=128, dim=0),
#             exact=True,
#         )
#         data = transform(data)
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
        adj = dgl.from_scipy(mmread(adj_path))
        feature = torch.from_numpy(mmread(feature_path).astype(np.float32))
        if feature.size(1) > 128:
            feature = feature[:, :128]
        labels = torch.from_numpy(mmread(labels_path).astype(np.int64))
        labels = torch.squeeze(labels)
        name = mat_folder
        print(mat_folder)

    num_classes = len(np.unique(labels))
    model = DGLGCN(
        in_channels=feature.size(1),
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
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
    print(f'DGL GraphConv,{name},{torch.tensor(times).sum():.4f}')

    # print('total conv1 time: ', model.conv1_time)
    # print('total conv2 time: ', model.conv2_time)

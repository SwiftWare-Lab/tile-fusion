import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from scipy.io import mmwrite
from torch_geometric.datasets import Planetoid
import torch_geometric.datasets as datasets
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from dgl.nn.pytorch.conv import GraphConv
from torch_geometric.utils.convert import to_dgl
import numpy as np

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
    out = model(data.x, dgl_graph)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
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
raw_folder_name = osp.join(osp.dirname(osp.realpath(__file__)), '../../modeling', 'data')
dataset_list = [
    datasets.Coauthor(root=raw_folder_name + '/coauthor_cs/', name='CS', transform=None),
    datasets.Coauthor(root=raw_folder_name + '/coauthor_physics/', name='Physics', transform=None),
    datasets.CoraFull(root=raw_folder_name + '/cora_full/', transform=None),
    # datasets.Flickr(root=raw_folder_name + '/flickr/', transform=None),
    # datasets.Yelp(root=raw_folder_name + '/yelp/', transform=None),
    datasets.Planetoid(root=raw_folder_name + '/planetoid/pubmed/', name='Pubmed', transform=None),
    datasets.Planetoid(root=raw_folder_name + '/planetoid/cora/', name='Cora', transform=None),
    datasets.GitHub(root=raw_folder_name + '/github/', transform=None),
    datasets.FacebookPagePage(root=raw_folder_name + '/facebook_page_page/', transform=None),
    datasets.DeezerEurope(root=raw_folder_name + '/deezer_europe/', transform=None)
    # datasets.Reddit2(root=raw_folder_name + '/reddit2/', transform=None)
]
for dataset in dataset_list:
    name = dataset.root.split('/')[-1]
    data = dataset[0].to(device)
    dgl_graph = to_dgl(data)
    if args.use_gdc:
        transform = T.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        )
        data = transform(data)



    model = DGLGCN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
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

import argparse
import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
import os
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from scipy.io import mmread

def convert_scipy_coo_to_torch_csr(coo):
    values = coo.data
    adj_cs = coo.tocsr()
    adj_cs.setdiag([1.] * adj_cs.shape[0])
    crow_indices = torch.tensor(adj_cs.indptr, dtype=torch.int32)
    col_indices = torch.tensor(adj_cs.indices, dtype=torch.int32)
    values = torch.tensor(adj_cs.data, dtype=torch.float32)
    adj = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(coo.shape[0], coo.shape[1]))
    # print(adj)
    return adj

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             bias=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, bias=False)
        self.conv1_time = 0
        self.conv2_time = 0

    def forward(self, x, edge_index, edge_weight=None):
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        # for param in self.conv1.parameters():
        #     mmwrite('weight1.mtx', param.data.detach().numpy())
        x = x.relu()
        # mmwrite('output1.mtx', x.detach().numpy())
        # x = F.dropout(x, p=0.5, training=self.training)x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # for param in self.conv2.parameters():
        #     mmwrite('weight2.mtx', param.data.detach().numpy())
        # mmwrite('output2.mtx',x.detach().numpy() )

        return F.log_softmax(x, dim=1)

    def get_op_time(self):
        return self.conv1.op_time + self.conv2.op_time

def train(backward_time=[0]):
    model.train()
    optimizer.zero_grad()
    out = model(feature, adj)
    loss = F.cross_entropy(out, labels)
    t1 = time.time()
    loss.backward()
    backward_time[0] += time.time() - t1
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

mat_file_path = args.dataset + '/mat_list.txt'
torch.set_num_threads(args.threads)
with open(mat_file_path) as mat_file:
    matrices = mat_file.readlines()
    for mat in matrices:
        mat = mat.rstrip()
        datafolder = args.dataset
        mat_folder = mat.split('/')[0]
        adj_path = os.path.join(args.dataset, mat)
        adj = convert_scipy_coo_to_torch_csr(mmread(adj_path))
        feature = torch.FloatTensor(adj.shape[0], args.hidden_channels).uniform_(-1, 1)
        labels = torch.randint(0, args.hidden_channels, (adj.shape[0],), dtype=torch.long)
        labels = torch.squeeze(labels)
        name = mat_folder
    # edge_index_sparse = to_torch_sparse_tensor(data.edge_index)
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




        model = GCN(
            in_channels=feature.size(1),
            hidden_channels=args.hidden_channels,
            out_channels=args.hidden_channels
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
        backward_time = [0]
        for epoch in range(0, 100):
            start = time.time()
            loss1 = train(backward_time)
            times.append(time.time() - start)
            # log(Epoch=epoch, Loss=loss1)
        # print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
        print(f'torch_geometric GCNConv,{name},{np.array(times).sum():.4f}')
        # print("gemm-spmm in forward", model.get_op_time())
        # print("backward time", backward_time[0])

        # print('total conv1 time: ', model.conv1_time)
        # print('total conv2 time: ', model.conv2_time)
import timeit

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.profile import profileit
from time import time



dataset = Planetoid("Planetoid", name="PubMed", transform=T.ToSparseTensor())
data = dataset[0]
#>>> Data(adj_t=[2708, 2708, nnz=10556], x=[2708, 1433], y=[2708], ...)


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        self.l1_time = 0
        self.l2_time = 0

    def forward(self, x, adj_t):
        # start timer
        t = time()
        x = self.conv1(x, adj_t)
        #print(len(adj_t.values()))
        x = F.relu(x)
        self.l1_time = time() - t
        t = time()
        x = self.conv2(x, adj_t)
        ret_val = F.log_softmax(x, dim=1)
        self.l2_time = time() - t
        return ret_val

model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def old_train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    # get the size of adjaceny matrix nnz
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return float(loss)


#@profileit()
def train(mod, opt, x_in, e_in, y_in):
    mod.train()
    opt.zero_grad()
    out = mod(x_in, e_in)
    # get the size of adjaceny matrix nnz
    loss = F.nll_loss(out, y_in)
    loss.backward()
    opt.step()
    return float(loss)


for epoch in range(1, 201):
    #loss = old_train(data)
    loss = train(model, optimizer, data.x, data.adj_t, data.y)
    print(model.l1_time , model.l2_time)
    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
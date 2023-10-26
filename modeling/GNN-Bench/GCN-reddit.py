import copy
import os.path as osp
import time

import scipy.sparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

from scipy.io import mmwrite
from scipy.sparse import coo_matrix
import numpy as np


def store_mtx(out_path, row_idx, col_ix, nnz_val):
    # convert to scinpysparse matrix
    if len(nnz_val) == 0:
        nnz_val = [1] * len(row_idx)
    A = coo_matrix((nnz_val, (row_idx, col_ix)), shape=(data.num_nodes, data.num_nodes))
    # store to file
    mmwrite(out_path, A)


def store_mtx_custom(out_path, row_idx, col_ix, nnz_val):
    if len(nnz_val) == 0:
        # open file
        f = open(out_path, "w")
        # write header
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(str(data.num_nodes) + " " + str(data.num_nodes) + " " + str(len(row_idx)) + "\n")
        # write data
        for i in range(len(row_idx)):
            f.write(str(row_idx[i]+1) + " " + str(col_ix[i]+1) + " " + str(1) + "\n")
        # close file
        f.close()
    else:
        # open file
        f = open(out_path, "w")
        # write header
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(str(data.num_nodes) + " " + str(data.num_nodes) + " " + str(len(row_idx)) + "\n")
        # write data
        for i in range(len(row_idx)):
            f.write(str(row_idx[i]+1) + " " + str(col_ix[i]+1) + " " + str(nnz_val[i]) + "\n")
        # close file
        f.close()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[2, 10], shuffle=True, **kwargs)
#store_mtx_custom("Original_reddit.mtx", np.array(data.edge_index[0]), np.array(data.edge_index[1]), [])
#exit(1)

subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)

# No need to maintain these features during evaluation:
del subgraph_loader.data.x, subgraph_loader.data.y
# Add global node index information.
subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)

mtx_no = 0
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        global mtx_no
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            #print(edge_index)
            # print nnz of edge_index
            #print(len(edge_index[0]))
            #print(len(edge_index[1]))
            #print(edge_index[0])
            #print(edge_index[1])
            path = "reddit_" + str(mtx_no) + ".mtx"
            if mtx_no % 50 == 0:
                store_mtx_custom(path, np.array(edge_index[0]), np.array(edge_index[1]), [])
            mtx_no += 1
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                #print(batch)
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


model = SAGE(dataset.num_features, 256, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        print("batch size: ",  batch.batch_size)
        #print(len(batch.edge_index.values()))
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]

        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test():
    model.eval()
    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    y = data.y.to(y_hat.device)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs


times = []
for epoch in range(1, 11):
    start = time.time()
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
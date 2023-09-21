import torch
from torch_geometric.datasets import Planetoid
from custom.model import CustomGCN
from pyG.model import GeometricGCN
# from SpMMProfiler import SpMMProfiler
from torchviz import make_dot
import sys
from sklearn.metrics import f1_score
import time
import numpy as np
import argparse
from pathlib import Path

# here we can configure in every epoch how many vertices will be in respective batch and how many epochs will be run
datasets = {
    'cora': {
        'root': 'data/cora', 'name': 'Cora', 'batch_count': 256, 'epoch_num': 100
    },
    'pubmed': {
        'root': 'data/pubmed', 'name': 'PubMed', 'batch_count': 1024, 'epoch_num': 200
    }
}


def train(visualize, mask):
    model.train()
    optimizer.zero_grad()
    out = model(x)
    # out = model(data)
    if visualize:
        make_dot(out, dict(model.named_parameters())).render("GCN-custom", format="png")
    loss = loss_function(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(mask):
    model.eval()
    with torch.no_grad():
        out = model(x)
        # out = model(data)
        pred = out[mask].argmax(dim=1)
        return f1_score(data.y[mask], pred, average="micro")


parser = argparse.ArgumentParser(Path(__file__).name)
parser.add_argument('-d', '--data', type=str, default='cora', help='dataset to use (e.g. cora, pubmed)')
parser.add_argument('-v', '--visualize', action='store_true', help='visualize model dag')
args = parser.parse_args()

dataset = Planetoid(root=datasets[args.data]['root'], name=datasets[args.data]['name'])

device = torch.device('cpu')
data = dataset[0].to(device)
x = data.x

adj_matrix = torch.sparse_coo_tensor(data.edge_index, [1. for i in range(data.edge_index.size(dim=1))])

num_nodes = adj_matrix.size(dim=1)
# here we can configure number of nodes to be used for training, test and validation
num_test = 2000
num_val = 500
num_train = num_nodes - num_test - num_val
print(num_train)
perm = torch.randperm(num_nodes)
train_mask = perm[:num_train]
val_mask = perm[num_train:num_train + num_val]
test_mask = perm[num_train + num_val:]

if args.visualize:
    epochs = 1
else:
    epochs = datasets[args.data]['epoch_num']
# batch_count = datasets[args.data]['batch_count']
batch_count = 100
# profiler =  SpMMProfiler()
model = CustomGCN(dataset.num_features, 128, adj_matrix, dataset.num_classes)
# model = GeometricGCN(dataset.num_features, 128, dataset.num_classes)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.7)

best_val_acc = test_acc = 0
times = []
loss_function = torch.nn.CrossEntropyLoss()

# every batch vertices are chosen randomly between training nodes
for epoch in range(epochs):
    train_perm = torch.randperm(num_train)
    batch = train_mask[train_perm[:batch_count]]
    start_time = time.time()
    loss = train(args.visualize, batch)
    end_time = time.time()
    times.append(end_time - start_time)
    print('epoch: ', epoch, '- loss: ', loss, '- time: ', end_time - start_time)

val_output = test(val_mask)
print("Validation F1:", val_output)

test_output = test(test_mask)
print("Test F1:", test_output)

print("Average batch time:", np.mean(times))
# print ("Average SpMM calculation Times in a epoch: ", profiler.get_average_spmm_time())

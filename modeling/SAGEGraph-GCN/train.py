
import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from SAGEGraph import SAGEGraph

def train():
    model.train()
    optimizer.zero_grad()
    out = model(x)
    print(out[train_mask])
    print(data.y[train_mask])
    loss = F.nll_loss(out[train_mask].float(), data.y[train_mask]) # TODO: need to be fixed
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1)
        acc = pred[test_mask] == data.y[test_mask]
        acc = int(acc.sum()) / int(test_mask.sum())
    return acc

dataset = Planetoid(root='data/cora', name='Cora')


device = torch.device('cpu')
data = dataset[0].to(device)
x = data.x
x = x.requires_grad_()
print(x)

adj_matrix = torch.sparse_coo_tensor(data.edge_index, [1. for i in range(data.edge_index.size(dim=1))], requires_grad=True)


num_nodes = adj_matrix.size(dim=1)
num_train = int(0.6 * num_nodes)
num_val = int(0.2 * num_nodes)
num_test = num_nodes - num_train - num_val
perm = torch.randperm(num_nodes)
train_mask = perm[:num_train]
val_mask = perm[num_train:num_train+num_val]
test_mask = perm[num_train+num_val:]

model = SAGEGraph(dataset.num_features, 16, adj_matrix, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

best_val_acc = test_acc = 0
for epoch in range(200):
    loss = train()
    val_acc = test()
    print('epoch: ', epoch, '- acc: ', val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = test()


print('Test accuracy: {:.4f}'.format(test_acc))
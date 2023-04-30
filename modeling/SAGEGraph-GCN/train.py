
import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from SAGEGraph import SAGEGraph
from torchviz import make_dot
import sys




def train(visualize):
    model.train()
    optimizer.zero_grad()
    out = model(x)
    if visualize:
        make_dot(out, dict(model.named_parameters())).render("SAGEGraph", format="png")
    loss = loss_function(out[train_mask], data.y[train_mask]) # TODO: need to be fixed
    loss.backward()
    optimizer.step()
    return loss.item()

def test(mask):
    model.eval()
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1)
        acc = pred[mask] == data.y[mask]
        acc = int(acc.sum()) / int(mask.size(dim=0))
    return acc

dataset = Planetoid(root='data/cora', name='Cora')


device = torch.device('cpu')
data = dataset[0].to(device)
x = data.x

adj_matrix = torch.sparse_coo_tensor(data.edge_index, [1. for i in range(data.edge_index.size(dim=1))])

num_nodes = adj_matrix.size(dim=1)
num_train = int(0.6 * num_nodes)
num_val = int(0.2 * num_nodes)
num_test = num_nodes - num_train - num_val
perm = torch.randperm(num_nodes)
train_mask = perm[:num_train]
val_mask = perm[num_train:num_train+num_val]
test_mask = perm[num_train+num_val:]

visualize = False
epochs = 200
if len(sys.argv) > 1 and sys.argv[1] == "v":
    visualize = True
    epochs = 1

model = SAGEGraph(dataset.num_features, 128, adj_matrix, dataset.num_classes)

optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=0.4)

best_test_acc = val_acc = 0
loss_function = torch.nn.CrossEntropyLoss()
for epoch in range(epochs):
    loss = train(visualize)
    test_acc = test(test_mask)
    print('epoch: ', epoch, '- acc: ', test_acc)
    best_model_state = dict()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        val_acc = test(val_mask)
        

print('Best test accuracy: {:.4f}'.format(best_test_acc))

print('Val accuracy: {:.4f}'.format(val_acc))
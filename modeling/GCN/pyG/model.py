import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GeometricGCN(torch.nn.Module):
    def __init__(self, feat_dim, embed_dim , num_classes):
        super(GeometricGCN, self).__init__()
        self.conv1 = GCNConv(feat_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
from SAGEConv import SAGEConv
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F

class SAGEGraph(torch.nn.Module):
    def __init__(self, feat_dim, embed_dim, adj_matrix, num_classes):
        super(SAGEGraph, self).__init__()
        self.conv1 = SAGEConv(feat_dim, embed_dim, adj_matrix)
        self.conv2 = SAGEConv(embed_dim, embed_dim, adj_matrix)
        self.linear = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        return x




import SAGEConv
import torch
from torch_geometric.nn import global_mean_pool

class SAGEGraph(torch.nn.Module):
    def __init__(self, feat_dim, embed_dim, adj_matrix, num_classes):

        super(SAGEGraph, self).__init__()
        self.conv1 = SAGEConv(feat_dim, embed_dim, adj_matrix)
        self.conv2 = SAGEConv(embed_dim, embed_dim, adj_matrix)
        self.linear = torch.nn.Linear(embed_dim, num_classes)
    
    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch=None)
        
        x = self.linear(x)
        return x

from SAGEConv import SAGEConv
import torch
import torch.nn.functional as F
from torchviz import make_dot

class SAGEGraph(torch.nn.Module):
    def __init__(self, feat_dim, embed_dim, adj_matrix, num_classes):
        super(SAGEGraph, self).__init__()
        self.conv1 = SAGEConv(feat_dim, embed_dim, adj_matrix)
        self.conv2 = SAGEConv(embed_dim, embed_dim, adj_matrix)
        self.linear = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, features, batch_nodes):

        x = self.conv1(features)
        x = self.conv2(x, True, batch_nodes)
        x = self.linear(x)
        
        return F.log_softmax(x, dim=1)




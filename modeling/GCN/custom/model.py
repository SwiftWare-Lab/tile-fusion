import torch.nn as nn
from custom.GCNConv import GCNConv 
import torch
import torch.nn.functional as F

class CustomGCN(nn.Module):
    
    def __init__(self, feat_dim, embed_dim, adjacency_matrix, num_classes):
        super(CustomGCN, self).__init__()    
        self.conv1 = GCNConv(feat_dim, embed_dim, adjacency_matrix)
        self.conv2 = GCNConv(embed_dim, num_classes, adjacency_matrix)
        self.linear = torch.nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = F.relu(x)
        x = self.conv2.forward(x)
        return F.log_softmax(x, dim=1)
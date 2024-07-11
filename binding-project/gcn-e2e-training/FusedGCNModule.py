import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from FusedGCNLayer import FusedGCNLayer
import numpy as np
class FusedGCN(torch.nn.Module):
    def __init__(self, feat_dim, embed_dim , num_classes, adj, num_threads):
        super(FusedGCN, self).__init__()
        self.conv1 = FusedGCNLayer(feat_dim, embed_dim, 128, adj, num_threads)
        self.conv2 = FusedGCNLayer(embed_dim, num_classes,  128, adj, num_threads)

    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)
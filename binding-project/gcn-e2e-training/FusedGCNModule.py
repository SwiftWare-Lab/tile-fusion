import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from FusedGCNLayer import FusedGCNLayer
from FusedGCNLayer import FirstLayerGCNCached
import numpy as np
class FusedGCN(torch.nn.Module):
    def __init__(self, feat_dim, embed_dim , num_classes, adj, feature, num_threads):
        super(FusedGCN, self).__init__()
        schedule = torch.ops.sw_gcn.inspect_vt_ro(adj, feat_dim, embed_dim, 300000, num_threads)
        self.conv1 = FusedGCNLayer(feat_dim, embed_dim, 128, adj, schedule)
        self.conv2 = FusedGCNLayer(embed_dim, num_classes,  128, adj, schedule)

    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        return x

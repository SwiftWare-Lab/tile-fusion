import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from FusedGCNLayer import FusedGCNLayer, UnFusedGCNLayer
from FusedGCNLayer import FirstLayerGCNCached
import numpy as np
class FusedGCN(torch.nn.Module):
    def __init__(self, feat_dim, embed_dim , num_classes, adj, feature, num_threads):
        super(FusedGCN, self).__init__()
        schedule = torch.ops.sw_gcn.inspect_vt_ro(adj, feat_dim, embed_dim, 500000, num_threads)
        self.conv1 = FusedGCNLayer(feat_dim, embed_dim, 128, adj, schedule, num_threads)
        self.conv2 = FusedGCNLayer(embed_dim, num_classes,  128, adj, schedule, num_threads)

    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        return x


class UnFusedGCN(torch.nn.Module):
    def __init__(self, feat_dim, embed_dim , num_classes, adj, feature, num_threads):
        super(UnFusedGCN, self).__init__()
        self.conv1 = UnFusedGCNLayer(feat_dim, embed_dim, adj, num_threads)
        self.conv2 = UnFusedGCNLayer(embed_dim, num_classes,  adj, num_threads)

    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        return x
import torch
import random


class GraphSage(torch.nn.Module):

    def __init__(self, adj_mtx, features, feat_dim, hidden_dim, out_dim):
        super(GraphSage, self).__init__()
        self.adj_mtx = adj_mtx
        self.features = features
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.w1 = torch.nn.Parameter(torch.randn(feat_dim, hidden_dim))
        self.w2 = torch.nn.Parameter(torch.randn(hidden_dim, out_dim))

    def forward(self, nodes):
        return

    # layer2: Compute embeddings by aggregating result of layer 1
    def layer2_forward(self, nodes):
        layer2_out = []
        for i in range(len(nodes)):
            node = nodes[i]
            layer1_nodes = self.adj_mtx[node].nonzeros()
            layer1_nodes.extend(node)
            layer1_embeds = self.layer1_forward(layer1_nodes)
            aggregated_features = self.aggregate_features(len(self.adj_mtx[node].nonzeros()), layer1_embeds[:-1])
            layer2_out.append(torch.cat([layer1_embeds[-1], aggregated_features], dim=1))
        return torch.stack(layer2_out).matmul(self.w2)

    # layer1: Compute embeddings by aggregating neighbors in depth 1
    def layer1_forward(self, nodes):
        layer1_feat = []
        for i in range(len(nodes)):
            node = nodes[i]
            neigh_feat = self.features[self.adj_mtx[node].nonzeros()]
            aggregated_features = self.aggregate_features(len(self.adj_mtx[node].nonzeros()), neigh_feat)
            layer1_feat.append(torch.cat([self.features[node], aggregated_features], dim=1))
        return torch.stack(layer1_feat).matmul(self.w1)

    def aggregate_features(self, neighbors_num, features):
        return features.sum(0).div(neighbors_num)

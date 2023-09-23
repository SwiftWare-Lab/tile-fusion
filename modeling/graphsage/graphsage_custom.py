import torch
import torch.nn.functional as F
import random


class GraphSage(torch.nn.Module):

    def __init__(self, adj_mtx, features, feat_dim, hidden_dim, out_dim):
        super(GraphSage, self).__init__()
        self.adj_mtx = adj_mtx
        self.features = features
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden_dim, feat_dim + feat_dim))
        torch.nn.init.xavier_uniform(self.w1)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim + hidden_dim))
        torch.nn.init.xavier_uniform(self.w2)
        self.cw = torch.nn.Parameter(torch.FloatTensor(hidden_dim, out_dim))
        torch.nn.init.xavier_uniform(self.cw)

    def forward(self, nodes):
        return self.layer2_forward(nodes).matmul(self.cw)

    # layer2: Compute embeddings by aggregating result of layer 1
    def layer2_forward(self, nodes):
        layer2_out = []
        for i in range(len(nodes)):
            node = nodes[i]
            neigh_nodes = self.adj_mtx[node].coalesce().indices().squeeze(0)
            layer1_nodes = torch.cat([neigh_nodes, node.unsqueeze(dim=0)])
            layer1_embeds = self.layer1_forward(layer1_nodes)
            neigh_num = len(self.adj_mtx[node].coalesce().indices().squeeze(0))
            aggregated_features = self.aggregate_features(neigh_num, layer1_embeds[:-1])
            layer2_out.append(torch.cat([layer1_embeds[-1], aggregated_features], dim=0))
        return F.relu(self.w2.matmul(torch.stack(layer2_out).t())).t()

    # layer1: Compute embeddings by aggregating neighbors in depth 1
    def layer1_forward(self, nodes):
        layer1_feat = []
        for i in range(len(nodes)):
            node = nodes[i]
            neigh_feat = self.features[self.adj_mtx[node].coalesce().indices().squeeze(0)]
            aggregated_features = self.aggregate_features(len(self.adj_mtx[node].coalesce().indices().squeeze(0)),
                                                          neigh_feat)
            layer1_feat.append(torch.cat([self.features[node], aggregated_features], dim=0))
        return F.relu(self.w1.matmul(torch.stack(layer1_feat).t())).t()

    def aggregate_features(self, neighbors_num, features):
        return features.sum(0) / neighbors_num

import torch
import random

def layer2_forward(nodes ,adj_list, input_channel, output_channel):
    
    for x in nodes:
        layer1_nodes = adj_list[x]
        layer1_nodes.append(x)
        layer1_features = layer1_forward(layer1_nodes)
    aggregated_features = aggregate_features(layer1_features[:-1])

def layer1_forward(nodes, adj_list, features, output_dim):
    for x in nodes:
        aggregated_features = aggregate_features(adj_list[x])
    return    
def aggregate_features(neighbors):
    return

class GraphSage(torch.nn.Module):

    def __init__(self, adj_mtx, features, feat_dim, out_dim):
        self.adj_mtx = adj_mtx
        self.features = features
        self.feat_dim = feat_dim
        self.out_dim = out_dim


    def forward(self, nodes):
        return

    def layer2_forward(self, nodes):
        for x in nodes:
            layer1_nodes = self.adj_mtx[x].nonzeros()
            layer1_nodes.append(x)
            layer1_embeds = layer1_forward(layer1_nodes)
            aggregated_features = aggregate_features(len(self.adj_mtx[x].nonzeros()),None,layer1_embeds[:-1])
            layer2_features = concat(layer1_embeds[-1],aggregated_features)
    
    def layer1_forward(self,nodes):
        for x in nodes:
            neigh_feat = self.features[self.adj_mtx[x].nonzeros()]           
            aggregated_features = aggregate_features(len(self.adj_mtx[x].nonzeros()), None, neigh_feat)
            layer1_features = concat(features[x], aggregated_features)
            
    def aggregate_features(self, neighbors_num, features):
        return features.sum(2).div(neighbors_num)
import torch
import torch.nn as nn
from torch.nn import init
import math

class GCNConv(nn.Module):
    def __init__(self, feat_dim, embed_dim, adjacency_matrix) -> None:
        super(GCNConv, self).__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.adj = adjacency_matrix

        self.weight = nn.Parameter(
                torch.FloatTensor(self.feat_dim, embed_dim))
        init.xavier_uniform(self.weight)
        
    def forward(self, x):
        N = x.shape[0]
        F = x.shape[1]
        F_ = self.weight.shape[1]
        # Initialize the output matrix
        out = torch.zeros(N, F_)

        # Loop over the nodes
        for i in range(N):
        # Get the indices of the neighbors of node i
            neighbors = self.adj[i].coalesce().indices().squeeze(0)
            # Get the degree of node i
            deg_i = torch.sparse.sum(self.adj[i])

            # Loop over the neighbors
            for j in neighbors:
            # Get the degree of neighbor j
                deg_j = torch.sparse.sum(self.adj[j])
                # Compute the normalized message from neighbor j
                message = x[j].matmul(self.weight).div(math.sqrt(deg_i * deg_j))

                # Add the message to the output of node i
                out[i] += message

                # Add the bias term to the output of node i
                # out[i] += b
        return out
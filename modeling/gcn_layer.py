import math
import torch


def gcn_layer(x, adj, W, b):
    # Assume x is a N x F matrix of node features, where N is the number of nodes and F is the feature dimension
    # Assume adj is a N x N sparse matrix of adjacency, where adj[i,j] = 1 if there is an edge from node i to node j, and 0 otherwise
    # Assume W is a F x F' weight matrix, where F' is the output feature dimension
    # Assume b is a F' dimensional bias vector
    N = x.shape[0]
    F = x.shape[1]
    F_ = W.shape[1]
    # Initialize the output matrix
    out = torch.zeros(N, F_)
    deg = torch.zeros(N)
    
    #preprocessing part

    for i in range(N):
        deg[i] = adj[i].sum().item()

    # Loop over the nodes
    for i in range(N):
    # Get the indices of the neighbors of node i
        neighbors = adj[i].nonzero(as_tuple=False).squeeze()

        # Loop over the neighbors
        for j in neighbors:
        # Get the degree of neighbor j

            # Compute the normalized message from neighbor j
            message = x[j] @ W / math.sqrt(deg[i] * deg[j])

            # Add the message to the output of node i
            out[i] += message

            # Add the bias term to the output of node i
            out[i] += b
    return out


# test gcn_layer
N = 100
F = 10
F_ = 5
x = torch.randn(N, F)
adj = torch.randint(2, (N, N))
W = torch.randn(F, F_)
b = torch.randn(F_)
out = gcn_layer(x, adj, W, b)
print(out)

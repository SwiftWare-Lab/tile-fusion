import torch
import torch.nn.functional as F
from torchviz import make_dot

class SAGEConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, adj_matrix):
        super(SAGEConv, self).__init__()
        self.encode = torch.nn.Linear(in_channels, out_channels)
        self.transform = torch.nn.Linear(in_channels + out_channels, out_channels, bias = False)
        self.adj_matrix = adj_matrix
    
    def forward(self, features, batch=False, batch_nodes=None):
        if batch: 
            batch_adj_matrix = self.adj_matrix.index_select(0, batch_nodes)
            batch_features = features.index_select(0, batch_nodes)
        else:
            batch_adj_matrix = self.adj_matrix
            batch_features = features

        neighbor_features = torch.sparse.mm(batch_adj_matrix, features)
        # Mean pooling aggregator
        neighbor_counts = torch.sparse.sum(batch_adj_matrix, dim=1).to_dense().unsqueeze(dim=1)
        pooled_features = torch.div(neighbor_features, neighbor_counts)

        aggregated_features = self.encode(pooled_features) #MM 
        
        
        updated_features = F.relu(self.transform(torch.cat([batch_features, aggregated_features], dim=1)))#MM

        return updated_features
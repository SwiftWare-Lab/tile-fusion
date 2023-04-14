import torch
import torch.nn.functional as F

class SAGEConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, adj_matrix):
        super(SAGEConv, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(in_channels + out_channels, out_channels)
        self.adj_matrix = adj_matrix
    
    def forward(self, x):
        neighbor_features = torch.sparse.mm(self.adj_matrix, x) #SpMM

        # TODO: torch.sum does not work on torch sparse kernel
        # #Mean pooling aggregator
        # neighbor_counts = torch.sum(self.adj_matrix, dim=1).unsqueeze(1)
        # pooled_features = torch.div(neighbor_features, neighbor_counts)

        aggregated_features = F.relu(self.linear1(neighbor_features)) #MM 
        
        
        updated_features = F.relu(self.linear2(torch.cat([x, aggregated_features], dim=1)))#MM
    
        return updated_features
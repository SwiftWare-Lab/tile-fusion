import torch
import torch.nn.functional as F

class SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix):
        super(SAGEConv, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(in_channels + out_channels, out_channels)
        self.adj_matrix = adj_matrix
    
    def forward(self, x):
        # TODO: we need sampling nodes here
        neighbor_features = torch.matmul(self.adj_matrix, x) #SpMM
        aggregated_features = F.relu(self.linear1(neighbor_features)) #MM 
        
        updated_features = F.relu(self.linear2(torch.cat([x, aggregated_features], dim=1)))
        return updated_features
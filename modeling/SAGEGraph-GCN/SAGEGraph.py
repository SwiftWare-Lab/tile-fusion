from SAGEConv import SAGEConv
import torch
import torch.nn.functional as F

class SAGEGraph(torch.nn.Module):
    def __init__(self, feat_dim, embed_dim, adj_matrix, num_classes):
        super(SAGEGraph, self).__init__()
        self.conv1 = SAGEConv(feat_dim, embed_dim, adj_matrix)
        self.conv2 = SAGEConv(embed_dim, embed_dim, adj_matrix)
        self.conv3 = SAGEConv(embed_dim, embed_dim, adj_matrix)
        self.conv4 = SAGEConv(embed_dim, embed_dim, adj_matrix)
        self.linear = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        
        return F.log_softmax(x, dim=1)




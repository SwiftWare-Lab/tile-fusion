import torch

torch.ops.load_library("build/lib/libsw_gcn.so")


class FusedGCNLayer(torch.nn.Module):

    def __init__(self, feat_dim, embed_dim, m_tile_size, adj, num_threads):
        super(FusedGCNLayer, self).__init__()
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(embed_dim, feat_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.schedule = torch.ops.sw_gcn.inspect(adj, m_tile_size)
        self.num_threads = num_threads
        self.adj = adj

    def forward(self, x):
        x = torch.ops.sw_gcn.fusedGeMMSpMM(self.adj, x, self.weight, self.schedule, self.num_threads)
        return x

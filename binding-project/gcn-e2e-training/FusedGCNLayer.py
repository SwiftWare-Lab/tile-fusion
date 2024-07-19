import torch

torch.ops.load_library("build/lib/libsw_gcn.so")


class FusedGCNLayer(torch.nn.Module):

    def __init__(self, feat_dim, embed_dim, m_tile_size, adj, num_threads):
        super(FusedGCNLayer, self).__init__()
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(embed_dim, feat_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        # print(feat_dim, embed_dim)
        # print(adj._nnz())
        # print(adj.size(0))
        self.adj = adj
        self.schedule_data = torch.ops.sw_gcn.inspect_vt_ro(adj, feat_dim, embed_dim, 1000000)
        # print(schedule)
        self.num_threads = num_threads
        self.ro_adj = torch.sparse_csr_tensor(self.schedule_data[0], self.schedule_data[1], self.schedule_data[2])
        self.schedule = self.schedule_data[3:]

    def forward(self, x):
        x = torch.ops.sw_gcn.fusedGeMMSpMM_vt_ro(self.adj, self.ro_adj, x, self.weight, self.schedule, self.num_threads)
        return x

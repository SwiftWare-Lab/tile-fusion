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
        # print(adj)
        # if (feat_dim > embed_dim):
        self.schedule = torch.ops.sw_gcn.inspect_vt_ro(adj, feat_dim, embed_dim, 1250000, num_threads)
        self.forward_fn = torch.ops.sw_gcn.fusedGeMMSpMM_vt_ro
        # else:
        #     self.schedule = torch.ops.sw_gcn.inspect_vt_ro(adj, embed_dim, feat_dim, 1250000, num_threads)
        #     self.forward_fn = torch.ops.sw_gcn.geMMSpMM_f_bw
        # print(schedule)
        self.num_threads = num_threads
        level_ptr = self.schedule[0]
        mix_ptr = self.schedule[1]
        self.max_tile_size = 0
        for x in range(level_ptr[0],level_ptr[1]):
            tile_size = mix_ptr[x*2 + 2] - mix_ptr[x*2]
            if tile_size > self.max_tile_size:
                self.max_tile_size = tile_size
        # self.ro_adj = torch.sparse_csr_tensor(self.schedule_data[0], self.schedule_data[1], self.schedule_data[2])

    def forward(self, x):
        x = self.forward_fn(self.adj, x, self.weight, self.schedule, self.num_threads, self.max_tile_size)
        # print(x)
        # print(x)
        return x


class FirstLayerGCNCached(torch.nn.Module):

    def __init__(self, feat_dim, embed_dim, m_tile_size, adj, feature, num_threads):
        super(FirstLayerGCNCached, self).__init__()
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(embed_dim, feat_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.af = torch.matmul(adj, feature)
        # print(schedule)
        self.num_threads = num_threads
        self.forward_fn = torch.ops.sw_gcn.cachedSpMMGeMM

    def forward(self, x):
        x = self.forward_fn(self.af, self.weight, self.num_threads)
        # print(x)
        return x

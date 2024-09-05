import torch
import scipy.io as sio
import sys


torch.ops.load_library("build/lib/libsw_gcn.so")
# print(torch.utils.cmake_prefix_path)

input_mat = sys.argv[1]
m_tile = int(sys.argv[2])
num_threads = int(sys.argv[3])
mat = sio.mmread(input_mat)
mat_csr = mat.tocsr()
adj = torch.sparse_csr_tensor(torch.tensor(mat_csr.indptr, dtype=torch.int32),
                              torch.tensor(mat_csr.indices, dtype=torch.int32),
                              torch.tensor(mat_csr.data, dtype=torch.float32))
# print(adj)
schedule = torch.ops.sw_gcn.inspect(adj, m_tile)
# print(schedule)
level_ptr = schedule[0]
mix_ptr = schedule[1]
partition = schedule[2]

feature = torch.rand(adj.size(1), 32, dtype=torch.float32)
weight = torch.rand(5, 32, dtype=torch.float32)
# print(feature)
# print(weight)

fused_out = torch.ops.sw_gcn.fusedGeMMSpMM(adj, feature, weight, schedule, num_threads)
# print(fused_out)

#Torch result

unfused_out = torch.mm(adj,torch.mm(feature,weight.t()))

#I have a bug here for the times that rows is not product of mtile
print("-----------------------------Fused Result-------------------------")
print(fused_out)
print("-----------------------------UnFused Result-------------------------")
print(unfused_out)
print("-----------------------------Compare-------------------------")
print(torch.eq(fused_out, unfused_out))
import scipy.io as sio
import sys
import torch
import time

torch.ops.load_library("lib/libsw_gcn.so")
# print(torch.utils.cmake_prefix_path)

input_mat = sys.argv[1]
bcol = int(sys.argv[2])
num_threads = int(sys.argv[3])
header = int(sys.argv[4])

torch.set_num_threads(num_threads)

mat = sio.mmread(input_mat)
mat_csr = mat.tocsr()
adj = torch.sparse_csr_tensor(torch.tensor(mat_csr.indptr, dtype=torch.int32),
                              torch.tensor(mat_csr.indices, dtype=torch.int32),
                              torch.tensor(mat_csr.data, dtype=torch.float32))

feature = torch.rand(adj.size(1), bcol, dtype=torch.float32)
weight = torch.rand(bcol, bcol, dtype=torch.float32)


mat_name = input_mat.split('/')[-1].split('.')[0]

tile_sizes = [8, 16, 32, 64, 128, 256, 512, 2048]
if header:
    print("Impl,Matrix,ExeTime,TileSize")
best_fused_time = float('inf')
best_tile = -1
for m_tile in tile_sizes:
    schedule = torch.ops.sw_gcn.inspect(adj, m_tile)
    level_ptr = schedule[0]
    mix_ptr = schedule[1]
    partition = schedule[2]
    execution_time = 0
    for i in range(7):
        start_time = time.time()
        fused_out = torch.ops.sw_gcn.executeFusedGeMMSpMM(adj, weight, feature, level_ptr, mix_ptr, partition, num_threads)
        end_time = time.time()
        execution_time += end_time - start_time
    avg_exe_time = execution_time / 7
    if avg_exe_time < best_fused_time:
        best_fused_time = avg_exe_time
        best_tile = m_tile

fused_row = ['FusedGeMMSpMM', mat_name, str(best_fused_time), str(best_tile)]

execution_time = 0
for i in range(7):
    start_time = time.time()
    unfused_out = torch.mm(adj, torch.mm(feature, weight))
    end_time = time.time()
    execution_time += end_time - start_time
avg_exe_time = execution_time / 7

unfused_row = ['UnfusedGeMMSpMM', mat_name, str(avg_exe_time), '-1']

print(','.join(fused_row))
print(','.join(unfused_row))



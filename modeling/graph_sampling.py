# import the required libraries
import math
import random
import sys

import numpy as np
import scipy.sparse as sp
import scipy.io as spio


def count_zero_rows(A):
    # get the number of rows and columns of A
    n, m = A.shape
    # iterate over the rows
    zero_rows = 0
    for i in range(n):
        # if the number of nonzeros in the row is zero, increment zero_rows
        if A.indptr[i + 1] - A.indptr[i] == 0:
            zero_rows += 1
    return zero_rows


# graph sampling
def graph_sampling(A, k):
    # get the number of rows and columns of A
    n, m = A.shape
    # create a sparse matrix of size n x m with k nonzeros where all nonzeros are 1
    sampled_graph = sp.random(n, m, density=k / (n * m), format='csr', dtype=np.float32)
    # multiply the sampled graph with A pointwise
    sampled_graph = sampled_graph.multiply(A)
    # return the sampled graph
    return sampled_graph


# graph sampling from the triplet format of A
def graph_sampling_triplet(A, k):
    # get the number of rows and columns of A
    n, m = A.shape
    nnz_A = A.nnz
    # generate k random numbers between 0 and nnz_A
    random_indices = random.sample(range(nnz_A), k)
    # define data and row and col indices
    data = np.zeros(k)
    row_indices = np.zeros(k)
    col_indices = np.zeros(k)
    # iterate over the random indices
    for i in range(k):
        # get the row and column indices of the random index
        row_indices[i] = A.row[random_indices[i]]
        col_indices[i] = A.col[random_indices[i]]
        data[i] = A.data[random_indices[i]]
    # create sparse coo matrix from the data, row and column indices
    sampled_graph = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n, m))
    return sampled_graph


# graph sampling A rows with random_rows mask
def graph_sampling_rows(A, random_rows):
    a_csr = A.tocsr()
    m, n = A.shape
    a_ind_ptr = a_csr.indptr
    a_indice = a_csr.indices
    a_data = a_csr.data
    sampled_ind_ptr = np.zeros(m+1, dtype=np.int32)
    sampled_ind_ptr[0] = 0
    sampled_nnz = 0
    for i in range(m):
        if i in random_rows:
            sampled_nnz = sampled_nnz + a_ind_ptr[i+1] - a_ind_ptr[i]
    sampled_indices = np.zeros(sampled_nnz)
    sampled_data = np.zeros(sampled_nnz)
    for i in range(m):
        if i in random_rows:
            row_nnz_num = a_ind_ptr[i+1] - a_ind_ptr[i]
            sampled_ind_ptr[i+1] = sampled_ind_ptr[i] + row_nnz_num
            sampled_indices[sampled_ind_ptr[i]:sampled_ind_ptr[i+1]] = a_indice[a_ind_ptr[i]:a_ind_ptr[i+1]]
            sampled_data[sampled_ind_ptr[i]:sampled_ind_ptr[i+1]] = a_data[a_ind_ptr[i]:a_ind_ptr[i+1]]
        else:
            sampled_ind_ptr[i+1] = sampled_ind_ptr[i]
    sampled_graph = sp.csr_matrix((sampled_data, sampled_indices, sampled_ind_ptr), shape=(m, n))
    return sampled_graph

#extract the neighbors of the sampled graph
def neighbors_of_sampled_graph(A, random_rows):
    a_csr = A.tocsr()
    m, n = A.shape
    a_ind_ptr = a_csr.indptr
    a_indices = a_csr.indices
    neighbors = set()
    for x in random_rows:
        for i in range(a_ind_ptr[x], a_ind_ptr[x+1]):
            neighbors.add(a_indices[i])
    return neighbors

def check_fusion(W1, W2, row_tile):
    fused_schedule = []
    m, n = W2.shape
    for i in range(0, m, row_tile):
        needed_iterations = []
        for j in range(i, i + row_tile):
            if i + row_tile >= m:
                break
            # take all dependenent rows
            tmp_needed_iter = (W2.indices[W2.indptr[j]:W2.indptr[j + 1]])
            # skip zero rows
            for ni in tmp_needed_iter:
                if W1.indptr[ni + 1] - W1.indptr[ni] != 0:
                    needed_iterations.append(ni)
            if len(needed_iterations) == 0:
                fused_schedule.append(j)
                continue
            # check the tile size
            max_iter, min_iter = max(needed_iterations), min(needed_iterations)
            if min_iter <= i and max_iter < i + row_tile:
                fused_schedule.append(j)
    # skip zero rows
    new_fused_schedule = []
    for i in fused_schedule:
        if W2.indptr[i + 1] - W2.indptr[i] != 0:
            new_fused_schedule.append(i)
    return new_fused_schedule

if __name__ == '__main__':
    n_samples = 50
    sample_ratio = 0.2
    tile_size = 64
    if len(sys.argv) > 1:
        A = spio.mmread(sys.argv[1])
        A = A.tocoo()
    else:
        # generate a random sparse matrix CSR format
        n = 80
        m = 80
        density = 0.1
        # generate two random matrix A and B of float32
        A = sp.random(n, m, density=density, format='coo', dtype=np.float32)
    if len(sys.argv) > 2:
        sample_ratio = float(sys.argv[2])
    if len(sys.argv) > 3:
        tile_size = int(sys.argv[3])
    n_samples = math.ceil(A.shape[0] * sample_ratio)
    sampled_rows = random.sample(range(A.shape[0]), n_samples)
    # sample the graph
    neighbors = neighbors_of_sampled_graph(A, sampled_rows)
    graph_L1 = graph_sampling_rows(A, neighbors)
    sampled_graph_L2 = graph_sampling_rows(A, sampled_rows)

    zero_rows2 = count_zero_rows(sampled_graph_L2)

    for tile_size in {2, 4, 8, 16, 32, 64, 128, 256, 512}:
        fused1 = check_fusion(graph_L1, sampled_graph_L2, tile_size)
        print(A.shape[0], ",", A.shape[1], ",", A.nnz, ",", n_samples, ",", len(fused1), ",", sample_ratio, ",",
              len(fused1) / A.shape[0], ",", len(fused1) / (A.shape[0] - zero_rows2), ",", tile_size)

# define main here
if __name__ == '__main1__':
    n_samples = 50
    sample_ratio = 0.2
    tile_size = 64
    if len(sys.argv) > 1:
        A = spio.mmread(sys.argv[1])
        A = A.tocoo()
    else:
        # generate a random sparse matrix CSR format
        n = 80
        m = 80
        density = 0.1
        # generate two random matrix A and B of float32
        A = sp.random(n, m, density=density, format='coo', dtype=np.float32)
    if len(sys.argv) > 2:
        sample_ratio = float(sys.argv[2])
    if len(sys.argv) > 3:
        tile_size = int(sys.argv[3])
    n_samples = math.ceil(A.nnz * sample_ratio)
    # sample the graph
    sampled_graph_w1 = graph_sampling_triplet(A, n_samples)
    sampled_graph_w2 = graph_sampling_triplet(A, n_samples)
    # convert to CSR format
    sampled_graph_w1 = sampled_graph_w1.tocsr()
    sampled_graph_w2 = sampled_graph_w2.tocsr()
    fused1 = check_fusion(sampled_graph_w1, sampled_graph_w2, tile_size)
    print(A.shape[0], ",", A.shape[1], ",", A.nnz, ",", n_samples, ",", len(fused1), ",", sample_ratio, ",",
          len(fused1) / A.shape[0])

if __name__ == '__main__2':
    n_samples = 50
    sample_ratio = 0.2
    tile_size = 64
    if len(sys.argv) > 1:
        A = spio.mmread(sys.argv[1])
        A = A.tocoo()
    else:
        # generate a random sparse matrix CSR format
        n = 80
        m = 80
        density = 0.1
        # generate two random matrix A and B of float32
        A = sp.random(n, m, density=density, format='coo', dtype=np.float32)
    if len(sys.argv) > 2:
        sample_ratio = float(sys.argv[2])
    if len(sys.argv) > 3:
        tile_size = int(sys.argv[3])
    n_samples = math.ceil(A.nnz * sample_ratio)
    # sample the graph
    sampled_graph_w1 = graph_sampling_triplet(A, n_samples)
    sampled_graph_w2 = graph_sampling_triplet(A, n_samples)
    # convert to CSR format
    sampled_graph_w1 = sampled_graph_w1.tocsr()
    sampled_graph_w2 = sampled_graph_w2.tocsr()
    zero_rows2 = count_zero_rows(sampled_graph_w2)

    for tile_size in {2, 4, 8, 16, 32, 64, 128, 256, 512}:
        fused1 = check_fusion(sampled_graph_w1, sampled_graph_w2, tile_size)
        print(A.shape[0], ",", A.shape[1], ",", A.nnz, ",", n_samples, ",", len(fused1), ",", sample_ratio, ",",
              len(fused1) / A.shape[0], ",", len(fused1) / (A.shape[0] - zero_rows2), ",", tile_size)
    # fused1 = check_fusion(sampled_graph_w1, sampled_graph_w2, tile_size)
    # print(A.shape[0], ",", A.shape[1], ",", A.nnz, ",", n_samples, ",", len(fused1), ",", sample_ratio, ",", len(fused1)/A.shape[0])

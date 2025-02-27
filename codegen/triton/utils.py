import numpy as np
import scipy.io as sio
import os


import scipy.sparse as sp

def convert_csr_matrix_to_power2_dim(A):
    m, n = A.shape
    m_new = int(2 ** np.ceil(np.log2(m)))
    n_new = int(2 ** np.ceil(np.log2(n)))
    new_Aptr = np.zeros(m_new + 1, dtype=np.int32)
    new_Aptr[:m+1] = A.indptr
    new_Aptr[m+1:] = A.indptr[-1]
    new_Adata = A.data
    new_Aindices = A.indices
    # new csr matrix
    new_A = sp.csr_matrix((new_Adata, new_Aindices, new_Aptr),
                          shape=(int(m_new), int(n_new)))
    return new_A



def get_matrix_list(file_path, base_path):
    with open(file_path, 'r') as f:
        matrix_list = f.readlines()
    matrix_list = [x.strip() for x in matrix_list]
    matrix_list = [base_path + x for x in matrix_list]
    # remove bundle1.mtx from the list
    matrix_list = [x for x in matrix_list if "bundle1.mtx" not in x]
    # sort matrix_list by nnz
    nnz_list, new_matrixlist = [], []
    for matrix in matrix_list:
        A = sio.mmread(matrix)
        if A.data.dtype != np.float64:
            print(f"Matrix {matrix} is not double precision")
            A.data = A.data.astype(np.float64)
        nnz_list.append(A.nnz)
    matrix_list = [x for _, x in sorted(zip(nnz_list, matrix_list))]
    return matrix_list
from scipy.sparse import csr_matrix
from scipy.io import mmwrite, mmread
from scipy.sparse.csgraph import reverse_cuthill_mckee
import sys
import os
import numpy as np

folder = sys.argv[1]
mat_list_file_name = os.path.join(folder, 'mat_list.txt')
with open(mat_list_file_name, 'r+') as mtf:
    mat_list = mtf.readlines()
    mat_list = [line.rstrip('\n') for line in mat_list]
    new_mat_list = []
    for mat_path in mat_list:
        mat_name = mat_path.split('/')[0]
        print(mat_name)
        mat_path = os.path.join(folder, mat_path)
        mat = mmread(mat_path)
        mat_csr = mat.tocsr()
        perm = reverse_cuthill_mckee(mat_csr)
        mat_csr = mat_csr[perm, :][:, perm]
        # features = features[perm, :]
        # labels = labels[:, perm]
        mat_name = mat_name + '_ordered'
        mat_folder = os.path.join(folder, mat_name)
        if not os.path.exists(mat_folder):
            os.mkdir(mat_folder)
        mat_path = os.path.join(folder, mat_name, mat_name+'.mtx')
        mat_rel_path = os.path.join(mat_name, mat_name+'.mtx')
        mmwrite(mat_path, mat_csr)
        new_mat_list.append(mat_rel_path+'\n')
    mtf.writelines(new_mat_list)

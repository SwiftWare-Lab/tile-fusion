import  scipy.io as sio
import sys
import os
import math
banded_mat = []
MAX_WIDTH = 2000
folder = sys.argv[1]
mat_list = os.path.join(folder, "banded_mat_list.txt")
with open(mat_list, "r") as file:
    mat_list = file.readlines()
mat_list = [x.strip() for x in mat_list]
print(len(mat_list))
for mat_name in mat_list:
    max_width = 0
    mat_path = os.path.join(folder, mat_name)
    matrix_inf = sio.mminfo(mat_path)
    if matrix_inf[0]> 4000000:
        break
    matrix = sio.mmread(mat_path)
    mtxCsr = matrix.tocsr()
    data = mtxCsr.data
    indices = mtxCsr.indices
    indptr = mtxCsr.indptr
    num_of_rows = mtxCsr.shape[0]
    banded = True
    for i in range(num_of_rows):
        width = indices[indptr[i+1]-1] - indices[indptr[i]]
        # print(width)
        if width > max_width:
            max_width = width
    print(mat_name, max_width)
    if max_width < MAX_WIDTH:
        banded_mat.append(mat_name)
    # if banded:
    #     banded_mat.append(mat_name)
print(banded_mat)
print(len(banded_mat))


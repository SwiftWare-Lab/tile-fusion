
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import scipy.sparse.linalg as splinalg
import time
import sys


def generate_banded_matrix(n):
    triplet_list = []
    for i in range(n):
        for j in range(n):
            if i == j:
                triplet_list.append((i, j, 1))
            elif i == j + 1 or i == j - 1:
                triplet_list.append((i, j, 1))
    return sp.coo_matrix((np.ones(len(triplet_list)), (np.array(triplet_list)[:, 0], np.array(triplet_list)[:, 1])),
                            shape=(n, n))


# main function
if __name__ == '__main__':
    mat = generate_banded_matrix(int(sys.argv[1]))
    # save mat as mtx file
    sio.mmwrite(sys.argv[2], mat)


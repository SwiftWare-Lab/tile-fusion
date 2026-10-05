import sys
import scipy.io as sio
from gen_matrix import generate_banded_matrix_efficient, generate_banded_matrix_sparse_rows
import argparse
import os

def create_matrix_folder(mtx_size_list, folder, bandwidth):
    matrix_names = []
    os.makedirs(folder, exist_ok=True)
    for s in mtx_size_list:
        for bw in bandwidth:
            mat_name = str(bw) + '_banded_' + str(s) + '.mtx'
            matrix_names.append(mat_name)
            mat = generate_banded_matrix_efficient(s, bw)
            # save mat as mtx file
            sio.mmwrite(os.path.join(folder, mat_name), mat)
    with open(os.path.join(folder,"mat_list.txt"), 'w') as mat_list:
        matrix_names = [mn + '\n' for mn in matrix_names]
        mat_list.writelines(matrix_names)


def create_matrix_folder_sparse_rows(mtx_size_list, folder, bandwidth):
    matrix_names = []
    os.makedirs(folder, exist_ok=True)
    for s in mtx_size_list:
        for bw in bandwidth:
            mat_name = str(bw) + '_sbanded_' + str(s) + '.mtx'
            matrix_names.append(mat_name)
            mat = generate_banded_matrix_sparse_rows(s, bw)
            # save mat as mtx file
            sio.mmwrite(os.path.join(folder, mat_name), mat)
    with open(os.path.join(folder,"mat_list.txt"), 'w') as mat_list:
        matrix_names = [mn + '\n' for mn in matrix_names]
        mat_list.writelines(matrix_names)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, default="./data/tri-banded")
    parser.add_argument('-b', '--bandwidth', nargs='+', type=int, default=[3])
    parser.add_argument('-sl', '--size-list', nargs='+', type=int)
    parser.add_argument('-t', '--type', type=str, default='dense_row')
    args = parser.parse_args()
    if args.type == 'dense_row':
        create_matrix_folder(args.size_list, args.folder, args.bandwidth)
    elif args.type == 'sparse_row':
        create_matrix_folder_sparse_rows(args.size_list, args.folder, args.bandwidth)

from ssgetpy import search, fetch
import sys
import os


def dl_save_list(matrix_directory, matrix_list_path):
    # specify what matrices should be downloaded
    result = search(rowbounds=(10000, 50000), colbounds=(10000, 50000), limit=100, kind='Directed Graph')
    # result.download(extract=True, destpath=matrix_directory)
    # generate the list of downloaded matrices
    matrix_list = []
    for matrix in result:
        # join two paths to get the full path of the matrix and append it to the matrix list
        if 'random' not in matrix.kind and 'sequence' not in matrix.kind and 'weighted':
            matrix_list.append(os.path.join(matrix.name, matrix.name + '.mtx'))
            matrix.download(extract=True, destpath=matrix_directory)
    # write the matrix list to a txt file in the matrix_list directory
    with open(matrix_list_path, 'w') as f:
        for item in matrix_list:
            f.write("%s\n" % item)


if __name__ == '__main__':
    dl_save_list(sys.argv[1], sys.argv[2])
from ssgetpy import search, fetch
import sys
import os

matrix_category = 2
download = 1
def dl_save_list(matrix_directory, matrix_list_path):
    # specify what matrices should be downloaded
    if matrix_category ==0:
        result = search(nzbounds=(100000, 100000000), isspd=True, limit=10000000000, dtype='real')
    elif matrix_category == 1:
        result = search(nzbounds=(100000, 100000000), rowbounds=(1, 3000000), limit=1000000000000, dtype='real', kind='graph')
    else:
        result = search(limit=10000000000)

    if download == 1:
        result.download(extract=True, destpath=matrix_directory)
    # generate the list of downloaded matrices
    matrix_list = []
    for matrix in result:
        # join two paths to get the full path of the matrix and append it to the matrix list
        if matrix.rows == matrix.cols:
            matrix_list.append(os.path.join(matrix.name, matrix.name + '.mtx'))
    # write the matrix list to a txt file in the matrix_list directory
    with open(matrix_list_path, 'w') as f:
        for item in matrix_list:
            f.write("%s\n" % item)


if __name__ == '__main__':
    dl_save_list(sys.argv[1], sys.argv[2])


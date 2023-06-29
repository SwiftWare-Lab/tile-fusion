
import os
import sys


def get_dlmc_list(dlmc_folder, rel_path, output_file_list):
    matrix_list = []
    with os.scandir(os.path.join(dlmc_folder, rel_path)) as entries:
        for entry in entries:
            print(entry.name)
            # if entry is csv file
            if entry.name.endswith(".smtx") and entry.is_file():
                matrix_list.append(os.path.join(rel_path, entry.name) )
    # write the matrix list to a txt file in the matrix_list directory
    with open(output_file_list, 'w') as f:
        for item in matrix_list:
            f.write("%s\n" % item)

# /home/kazem/UFDB/DLMC/dlmc transformer/magnitude_pruning/0.9/ dlmc_magnitude_90.txt
# /home/kazem/UFDB/DLMC/dlmc transformer/magnitude_pruning/0.95/ dlmc_magnitude_95.txt
# /home/kazem/UFDB/DLMC/dlmc transformer/magnitude_pruning/0.98/ dlmc_magnitude_98.txt
if __name__ == '__main__':
    get_dlmc_list(sys.argv[1], sys.argv[2], sys.argv[3])

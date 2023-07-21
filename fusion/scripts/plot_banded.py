

import sys

import numpy as np
from numpy import ma
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import os


def filter(df, **kwargs):
    bool_index = None
    for key, value in kwargs.items():
        if isinstance(value, list):
            _bool_index = df[key].isin(value)
        else:
            _bool_index = df[key] == value
        if bool_index is None:
            bool_index = _bool_index
        else:
            bool_index = bool_index & _bool_index
    return df[bool_index]


def take_median(df, **kwargs):
    num_trial = df['Number of Trials']
    time_array = []
    # for each row in dataframe df
    for i in range(num_trial):
        t1 = df['Trial' + str(i) + ' Subregion0 Executor']
        time_array.append(t1)
    return np.median(time_array)


def get_fused_info(mat_list, df_fusion, imp_name, params=None):
    if params is None: # TODO: these params are hardcoded for now
        #params = [40, 400, 4000, 8000, 10000]
        #params = [4, 8, 40, 100, 1000]
        params = [10, 50, 100, 1000, 5000]
        #params = [10, 20, 50, 100, 200]
    fused, fused_40, fused_400, fused_4000, fused_8000, fused_10000 = [], [], [], [], [], []
    for mat in mat_list:
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        fused = cur_mat[cur_mat['Implementation Name'] == imp_name]
        fused_40.append(take_median(fused[fused['LBC WPART'] == params[0]]))
        fused_400.append(take_median(fused[fused['LBC WPART'] == params[1]]))
        fused_4000.append(take_median(fused[fused['LBC WPART'] == params[2]]))
        fused_8000.append(take_median(fused[fused['LBC WPART'] == params[3]]))
        fused_10000.append(take_median(fused[fused['LBC WPART'] == params[4]]))
    return fused_40, fused_400, fused_4000, fused_8000, fused_10000


def get_fused_info(mat_list, df_fusion, imp_name, params=None):
    if params is None: # TODO: these params are hardcoded for now
        #params = [40, 400, 4000, 8000, 10000]
        #params = [4, 8, 40, 100, 1000]
        params = [10, 50, 100, 1000, 5000]
        #params = [10, 20, 50, 100, 200]
    fused, fused_40, fused_400, fused_4000, fused_8000, fused_10000 = [], [], [], [], [], []
    for mat in mat_list:
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        fused = cur_mat[cur_mat['Implementation Name'] == imp_name]

        # fused_40.append(take_median(fused[fused['LBC WPART'] == params[0]]))
        # fused_400.append(take_median(fused[fused['LBC WPART'] == params[1]]))
        # fused_4000.append(take_median(fused[fused['LBC WPART'] == params[2]]))
        # fused_8000.append(take_median(fused[fused['LBC WPART'] == params[3]]))
        # fused_10000.append(take_median(fused[fused['LBC WPART'] == params[4]]))
    return fused_40, fused_400, fused_4000, fused_8000, fused_10000


# a function to extract the matrix list that has al LBC WPART, some of them are crashed
def get_matrix_list(df_fusion, imp_names=None, params=None):
    if imp_names is None:
        imp_names = df_fusion['Implementation Name'].unique()
    if params is None:
        params = df_fusion['LBC WPART'].unique()
    mat_list, new_mat_list = df_fusion['MatrixName'].unique(), []
    num_imps, num_params = len(imp_names), len(params)
    for mat in mat_list:
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        # see if the matrix has all the LBC WPART
        sw = True
        for imp_name in imp_names:
            for param in params:
                if len(cur_mat[cur_mat['Implementation Name'] == imp_name][cur_mat['LBC WPART'] == param]) == 0:
                    sw = False
                    break

        if sw:
            new_mat_list.append(mat)
    return new_mat_list


def plot_spmm_spmm(logs_folder, file_name, baseline_implementation):
    df_fusion = pd.read_csv(os.path.join(logs_folder,file_name))
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    # mat_list = df_fusion['MatrixName'].unique()
    mat_list = get_matrix_list(df_fusion)
    bCol = df_fusion['bCols'].unique()[0]
    nnz_list = df_fusion['NNZ'].unique()
    ntile_list = df_fusion['NTile'].unique()
    mtile_list = df_fusion['MTile'].unique()
    lbc_wpart_list = df_fusion['LBC WPART'].unique()
    iter_per_part_list = df_fusion['Iter Per Partition'].unique()
    implementations = df_fusion['Implementation Name'].unique()
    mats = df_fusion['MatrixName'].unique()

    # compute median for each entry in df_fusion
    for item, row in df_fusion.iterrows():
        t_array = take_median(row)
        df_fusion.loc[item, 'median'] = t_array

    groups = df_fusion.groupby(['MatrixName', 'Implementation Name', 'bCols'])
    df_best = {}
    timing_per_impl = np.zeros((len(mat_list), len(implementations)))
    for name, group in groups:
        df_best[name] = min(group['median'].values)
        # the index of where matrix name is in mat_list
        mat_idx = np.where(mats == name[0])[0][0]
        # the index of where implementation name is in implementations
        imp_idx = np.where(implementations == name[1])[0][0]
        timing_per_impl[mat_idx][ imp_idx] = df_best[name]

    # get the timing of unfused parallel
    unfused_parallel_idx = np.where(implementations == 'SpMM_SpMM_Demo_UnFusedParallel')[0][0]
    unfused_parallel_timing = timing_per_impl[:, unfused_parallel_idx]
    # get the timing of fused with redundant
    fused_redundant_idx = np.where(implementations == 'SpMM_SpMM_FusedTiledParallel_Redundant')[0][0]
    fused_redundant_timing = timing_per_impl[:, fused_redundant_idx]
    # get the timing of fused without redundant
    fused_without_redundant_idx = np.where(implementations == 'SpMM_SpMM_FusedParallel')[0][0]
    fused_without_redundant_timing = timing_per_impl[:, fused_without_redundant_idx]

    # plot a bar chart with matrices in x and time of the three implementation on y
    x_vals = np.arange(len(mat_list))
    width = 0.3
    fig, ax = plt.subplots(figsize=(15, 15))
    # set font size to be 20
    plt.rcParams.update({'font.size': 20})
    ax.bar(x_vals - width, unfused_parallel_timing, width, label='Unfused Parallel')
    ax.bar(x_vals, fused_redundant_timing, width, label='Fused with Redundant')
    ax.bar(x_vals + width, fused_without_redundant_timing, width, label='Fused without Redundant')
    # label x-axis values with corresponding nnz
    ax.set_xticks(x_vals)
    ax.set_xticklabels(mat_list, rotation=45)
    ax.grid(False)
    # set x and y axis label
    ax.set_xlabel('Matrix Name')
    ax.set_ylabel('Execution Time (sec)')
    ax.set_title('for bCols = ' + str(bCol))
    ax.legend()
    plt.show()


    plot_folder = "plots"
    os.makedirs(plot_folder, exist_ok=True)
    fig.savefig(os.path.join(plot_folder, file_name[:-4] + '.pdf'), bbox_inches='tight')
    #fig.show()


def plot_spmm_spmm_for_logs(logs_folder, baseline_implementation):
    with os.scandir(logs_folder) as entries:
        for entry in entries:
            print(entry.name)
            # if entry is csv file
            if entry.name.endswith(".csv") and entry.is_file():
                plot_spmm_spmm(logs_folder, entry.name, baseline_implementation)


if __name__ == '__main__':
    plot_spmm_spmm_for_logs(sys.argv[1], sys.argv[2])

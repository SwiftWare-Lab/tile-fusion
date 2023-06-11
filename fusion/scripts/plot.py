
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
    num_trial = df['Number of Trials'].unique()[0]
    time_array = []
    # for each row in dataframe df
    for index, row in df.iterrows():
        for i in range(num_trial):
            t1 = row['Trial' + str(i) + ' Subregion0 Executor']
            time_array.append(t1)
    return np.median(time_array)


def get_fused_info(mat_list, df_fusion, imp_name):
    fused, fused_40, fused_400, fused_4000, fused_8000, fused_10000 = [], [], [], [], [], []
    for mat in mat_list:
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        fused = cur_mat[cur_mat['Implementation Name'] == imp_name]
        fused_40.append(take_median(fused[fused['LBC WPART'] == 40]))
        fused_400.append(take_median(fused[fused['LBC WPART'] == 400]))
        fused_4000.append(take_median(fused[fused['LBC WPART'] == 4000]))
        fused_8000.append(take_median(fused[fused['LBC WPART'] == 8000]))
        fused_10000.append(take_median(fused[fused['LBC WPART'] == 10000]))
    return fused_40, fused_400, fused_4000, fused_8000, fused_10000


def plot_spmm_spmm(logs_folder, file_name, baseline_implementation):
    df_fusion = pd.read_csv(os.path.join(logs_folder,file_name))
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    mat_list = df_fusion['MatrixName'].unique()
    nnz_list = df_fusion['NNZ'].unique()
    seq_exe_time, separated_exe_time = [], []
    fused_40, fused_400, fused_4000, fused_8000, fused_10000 = [], [], [], [], []
    for mat in mat_list:
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        seq = cur_mat[cur_mat['Implementation Name'] == 'SpMM_SpMM_Demo']
        seq_exe_time.append(take_median(seq))
        separated = cur_mat[cur_mat['Implementation Name'] == baseline_implementation]
        separated_exe_time.append(take_median(separated))
        # fused = cur_mat[cur_mat['Implementation Name'] == 'SpMM_SpMM_FusedParallel']
        # fused_40.append(take_median(fused[fused['LBC WPART'] == 40]))
        # fused_400.append(take_median(fused[fused['LBC WPART'] == 400]))
        # fused_4000.append(take_median(fused[fused['LBC WPART'] == 4000]))
        # fused_8000.append(take_median(fused[fused['LBC WPART'] == 8000]))
        # fused_10000.append(take_median(fused[fused['LBC WPART'] == 10000]))
    # get fused info
    fused_40, fused_400, fused_4000, fused_8000, fused_10000 = get_fused_info(mat_list, df_fusion, 'SpMM_SpMM_FusedParallel')
    fused_sep_40, fused_sep_400, fused_sep_4000, fused_sep_8000, fused_sep_10000 = get_fused_info(mat_list, df_fusion, 'SpMM_SpMM_Separated_FusedParallel')
    fused_out_40, fused_out_400, fused_out_4000, fused_out_8000, fused_out_10000 = get_fused_info(mat_list, df_fusion, 'SpMM_SpMM_OuterProduct_FusedParallel')

    # geomean speedup of fused vs separated
    gg = gmean(np.array(separated_exe_time) / np.array(fused_40))
    geomean_speedup_40 = np.exp(np.mean(np.log(np.array(separated_exe_time) / np.array(fused_40))))
    geomean_speedup_400 = np.exp(np.mean(np.log(np.array(separated_exe_time) / np.array(fused_400))))
    geomean_speedup_4000 = np.exp(np.mean(np.log(np.array(separated_exe_time) / np.array(fused_4000))))
    geomean_speedup_8000 = np.exp(np.mean(np.log(np.array(separated_exe_time) / np.array(fused_8000))))
    geomean_speedup_10000 = np.exp(np.mean(np.log(np.array(separated_exe_time) / np.array(fused_10000))))
    # take minimum of fused arrays
    min_fused = np.minimum(np.minimum(np.minimum(np.array(fused_40), np.array(fused_400)), np.array(fused_4000)),
                           np.minimum(np.array(fused_8000), np.array(fused_10000)))
    # take the minimum of fused_sep arrays
    min_fused_sep = np.minimum(np.minimum(np.minimum(np.array(fused_sep_40), np.array(fused_sep_400)), np.array(fused_sep_4000)),
                            np.minimum(np.array(fused_sep_8000), np.array(fused_sep_10000)))
    # take the minimum of fused_out arrays
    min_fused_out = np.minimum(np.minimum(np.minimum(np.array(fused_out_40), np.array(fused_out_400)), np.array(fused_out_4000)),
                            np.minimum(np.array(fused_out_8000), np.array(fused_out_10000)))
    # take the min of fused and fused_sep and fused_out
    #min_fused = np.minimum(min_fused, min_fused_sep)
    #min_fused = np.minimum(min_fused, min_fused_out)

    # geomean speedup of fused vs separated
    geomean_speedup_min = np.exp(np.mean(np.log(np.array(separated_exe_time) / np.array(min_fused))))
    geomean_speedup_min_sep = np.exp(np.mean(np.log(np.array(separated_exe_time) / np.array(min_fused_sep))))
    geomean_speedup_min_out = np.exp(np.mean(np.log(np.array(separated_exe_time) / np.array(min_fused_out))))

    print('geomean speedup of fused vs separated: ', geomean_speedup_40, geomean_speedup_400, geomean_speedup_4000,
          geomean_speedup_8000, geomean_speedup_10000, geomean_speedup_min, geomean_speedup_min_sep, geomean_speedup_min_out)
    # geomean speedup of fused vs seq
    x_vals = np.arange(len(mat_list))
    # plot flop_sf vs flop_ulbc vs flop_umkl
    fig, ax = plt.subplots()
    ax.scatter(x_vals, np.array(separated_exe_time) / np.array(min_fused), facecolors='none', edgecolors='b',
               marker='s')
    # set a straight line at 1 as baseline
    ax.plot(x_vals, np.ones(len(mat_list)), 'r--')
    # label x-axis values with corresponding nnz
    #ax.set_xticks(nnz_list)

    ax.grid(False)
    # set x and y axis label
    ax.set_xlabel('Matrix ID', fontsize=20, fontweight='bold')
    ax.set_ylabel('Speedup Fused vs ' + baseline_implementation, fontsize=20, fontweight='bold')
    # set x and y axis tick size
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    # set right and top axis off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set left and bottom axis bold
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    fig.set_size_inches(18, 8)
    # set x tick black
    ax.tick_params(axis='x', colors='black')
    # set y tick black
    ax.tick_params(axis='y', colors='black')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # show legend
    # fig.legend(handles, labels, fontsize=14, ncol=3, loc='upper center', frameon=True, borderaxespad=1)
    ax.legend(loc='upper left', fontsize=20, ncol=3, frameon=True, borderaxespad=1)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    # fig.show()
    plot_folder = "plots"
    os.makedirs(plot_folder, exist_ok=True)
    fig.savefig(os.path.join(plot_folder, file_name[:-4] + '.pdf'), bbox_inches='tight')
    # fig.show()


def plot_spmm_spmm_for_logs(logs_folder, baseline_implementation):
    with os.scandir(logs_folder) as entries:
        for entry in entries:
            print(entry.name)
            # if entry is csv file
            if entry.name.endswith(".csv") and entry.is_file():
                plot_spmm_spmm(logs_folder, entry.name, baseline_implementation)


if __name__ == '__main__':
    plot_spmm_spmm_for_logs(sys.argv[1], sys.argv[2])

import sys

import numpy as np
from numpy import ma
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def plot_spmm_spmm(input_path1):
    df_fusion = pd.read_csv(input_path1)
    mat_list = df_fusion['MatrixName'].unique()
    separated_exe_time, separated2_exe_time = [], []
    for mat in mat_list:
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        separated = cur_mat[cur_mat['Implementation Name'] == 'SpMM_SpMM_Tutorial_Demo_UnFusedParallel']
        separated_exe_time.append(take_median(separated))
        separated2 = cur_mat[cur_mat['Implementation Name'] == 'SpMM_SpMM_Tutorial_Demo_UnFusedParallel2']
        separated2_exe_time.append(take_median(separated2))
    # geomean speedup of seperated2 vs separated
    geomean_speedup = np.exp(np.mean(np.log(np.array(separated_exe_time) / np.array(separated2_exe_time))))
    # take minimum of fused arrays

    print('geomean speedup of seperate v2 vs separated: ', geomean_speedup)
    # geomean speedup of seq2 vs seq
    x_vals = np.arange(len(mat_list))
    # plot flop_sf vs flop_ulbc vs flop_umkl
    fig, ax = plt.subplots()
    ax.scatter(x_vals, np.array(separated2_exe_time)/np.array(separated_exe_time),  facecolors='none', edgecolors='b', marker='s')
    # set a straight line at 1 as baseline
    ax.plot(x_vals, np.ones(len(mat_list)), 'r--')

    ax.grid(False)
    # set x and y axis label
    ax.set_xlabel('Matrix ID', fontsize=20, fontweight='bold')
    ax.set_ylabel('Speedup Seperated V2 vs Separated', fontsize=20, fontweight='bold')
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
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    # show legend
    #fig.legend(handles, labels, fontsize=14, ncol=3, loc='upper center', frameon=True, borderaxespad=1)
    ax.legend(loc='upper left', fontsize=20, ncol=3, frameon=True, borderaxespad=1)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    #fig.show()
    fig.savefig('mm-mm.pdf', bbox_inches='tight')
    #fig.show()

if __name__ == '__main__':
    plot_spmm_spmm(sys.argv[1])
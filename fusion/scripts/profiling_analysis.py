



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


def take_counters(df, counter_names, thread_id="Thread0"):
    num_trial = df['Number of Trials']
    mid_run = num_trial // 2
    counters = {}
    for cnt in counter_names:
        label = "Trial" + str(mid_run) + " Subregion0 " + thread_id + " " + cnt
        counters[cnt] = df[label]
    return counters

# a function to extract the matrix list that has al LBC WPART, some of them are crashed
def get_matrix_list(df_fusion, imp_names=None, params=None):
    if imp_names is None:
        imp_names = df_fusion['Implementation Name'].unique()
    if params is None:
        params = df_fusion['LBC WPART'].unique()
    mat_list, new_mat_list = df_fusion['Matrix Name'].unique(), []
    # num_imps, num_params = len(imp_names), len(params)
    # for mat in mat_list:
    #     cur_mat = df_fusion[df_fusion['Matrix Name'] == mat]
    #     # see if the matrix has all the LBC WPART
    #     sw = True
    #     for imp_name in imp_names:
    #         for param in params:
    #             cond1 = cur_mat['Implementation Name'] == imp_name
    #             cond2 = cur_mat['LBC WPART'] == param
    #             both_cond = cur_mat[cond1 & cond2]
    #             #if len(cur_mat[cur_mat['Implementation Name'] == imp_name][cur_mat['LBC WPART'] == param]) == 0:
    #             if len(both_cond) == 0:
    #                 sw = False
    #                 break
    #     if sw:
    #         new_mat_list.append(mat)
    return mat_list


def plot_spmm_spmm(logs_folder, baseline_implementation):
    df_fusion = pd.read_csv(os.path.join(logs_folder))
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    # mat_list = df_fusion['MatrixName'].unique()
    mat_list = get_matrix_list(df_fusion)
    bCol = df_fusion['bCols'].unique()[0]
    num_threads = df_fusion['nThreads'].unique()[0]
    nnz_list = df_fusion['NNZ'].unique()
    ntile_list = df_fusion['NTile'].unique()
    mtile_list = df_fusion['MTile'].unique()
    lbc_wpart_list = df_fusion['LBC WPART'].unique()
    iter_per_part_list = df_fusion['Iter Per Partition'].unique()
    implementations = df_fusion['Implementation Name'].unique()
    mats = df_fusion['Matrix Name'].unique()

    counter_names = ["PAPI_TOT_INS", "PAPI_L2_TCM", "PAPI_L3_TCM", "PAPI_L1_TCM"]
    # compute median for each entry in df_fusion
    for item, row in df_fusion.iterrows():
        t_array = take_median(row)
        df_fusion.loc[item, 'median time'] = t_array
        counters = take_counters(row, counter_names)
        for counter_name, counter_value in counters.items():
            df_fusion.loc[item, 'median ' + counter_name] = counter_value

    groups = df_fusion.groupby(['Matrix Name', 'bCols'])
    valid_implementation = ['SpMM_SpMM_FusedParallelAvx256_FixedTile', 'SpMM_SpMM_FusedParallel_FixedTile', 'SpMM_SpMM_FusedParallelKTiledAvx256_VariableTile']
    df_best = {}
    timing_per_impl = np.zeros((len(mat_list), len(implementations)))
    for name, group in groups:
        # get counter infor for each valid implementation

        for ccnnt in counter_names:
            if ccnnt == "PAPI_TOT_INS":
                continue
            group_imp = group[group['Implementation Name'] == valid_implementation[0]]
            tot_ins_1 = group_imp['median ' + "PAPI_TOT_INS"].values
            cnt_info_1 = group_imp['median ' + ccnnt].values
            mtile_list_1 = group_imp['MTile'].values
            group_imp = group[group['Implementation Name'] == valid_implementation[1]]
            tot_ins_2 = group_imp['median ' + "PAPI_TOT_INS"].values
            cnt_info_2 = group_imp['median ' + ccnnt].values
            mtile_list_2 = group_imp['MTile'].values
            group_imp = group[group['Implementation Name'] == valid_implementation[2]]
            tot_ins_3 = group_imp['median ' + "PAPI_TOT_INS"].values
            cnt_info_3 = group_imp['median ' + ccnnt].values
            mtile_list_3 = group_imp['MTile'].values
            # plot the counter info for both implementations
            plt.scatter(mtile_list_1, cnt_info_1/tot_ins_1, label=valid_implementation[0])
            plt.scatter(mtile_list_2, cnt_info_2/tot_ins_2, label=valid_implementation[1])
            plt.scatter(mtile_list_3, cnt_info_3/tot_ins_3, label=valid_implementation[2])
            # make xtick to be values in matile_list and in even distribution
            plt.xticks(mtile_list_1)
            plt.xlabel('MTile')
            plt.ylabel(ccnnt + "/ PAPI_TOT_INS")
            plt.title('Counter Info for ' + ccnnt + ' for ' + str(name[0]) + ' with ' + str(name[1]) + ' bCols')
            plt.legend()
            plt.savefig(os.path.join("logs/plots", 'counter_info_' + str(name[0]) + '_' + str(name[1]) + '_' + ccnnt + '.png'))
            #plt.show()
            plt.close()


# the main entry point for this module
if __name__ == "__main__":
    args = sys.argv[1:]
    logs_folder = args[0]
    file_name = "file0"
    baseline_implementation = "test"
    plot_spmm_spmm(logs_folder, baseline_implementation)

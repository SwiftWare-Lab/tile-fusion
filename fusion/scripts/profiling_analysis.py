



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


L1_misses, L2_misses, L3_misses = 'L1D:REPLACEMENT', 'LLC_REFERENCES', 'LLC_MISSES'
tot_inst = "INST_RETIRED:ANY_P"
#https://www.7-cpu.com/cpu/Haswell.html
arch_params = {'L1_ACCESS_TIME': 4, 'L2_ACCESS_TIME': 12, 'L3_ACCESS_TIME': 36, #or 43,
                'MAIN_MEMORY_ACCESS_TIME': 89}

def get_counter_info_per_imp(df_fusion, imp_name, counter_names):
    counter_values = {}
    group_imp = df_fusion[df_fusion['Implementation Name'] == imp_name]
    for ccnnt in counter_names:
        counter_values[ccnnt] = group_imp['median ' + ccnnt].values
    counter_values['MTile'] = group_imp['MTile'].values
    return counter_values


def compute_memory_cycles(counters_val):
    mem_cycle = counters_val[tot_inst] * arch_params['L1_ACCESS_TIME']
    #mem_cycle += counters_val[L1_misses] * arch_params['L2_ACCESS_TIME']
    mem_cycle += counters_val[L2_misses] * arch_params['L3_ACCESS_TIME']
    mem_cycle += counters_val[L3_misses] * arch_params['MAIN_MEMORY_ACCESS_TIME']
    mem_cycle /= counters_val[tot_inst]
    # mem_cycle = arch_params['L1_ACCESS_TIME'] + \
    #             arch_params['MAIN_MEMORY_ACCESS_TIME'] * (counters_val[L3_misses] / counters_val[L2_misses])
    return mem_cycle


def computing_working_set(df_fusion, imp_name, params=None):
    group_imp = df_fusion[df_fusion['Implementation Name'] == imp_name]
    # working set for MxK = MxN * NxK
    M_tile = group_imp['Loop 1 Itarations0'].values + group_imp['Number of Fused Nodes0'].values
    nnz_Tile = group_imp['Loop 1 NNZ0'].values
    K_tile = group_imp['bCols'].values
    density = group_imp['NNZ'].values / (group_imp['nRows'].values * group_imp['nCols'].values)
    active_col = density * group_imp['nCols'].values
    working_set = ((M_tile + active_col) * K_tile * 8) / (1024 * 1024)
    fused_nnz = group_imp['Number of Fused nnz0'].values
    working_set += 2*(nnz_Tile - fused_nnz) * K_tile * 8 / (1024 * 1024)
    return working_set


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
    #counter_names = ["PAPI_TOT_INS", "PAPI_SR_INS", "PAPI_LD_INS", "PAPI_BR_INS"]
    counter_names = ['L1D:REPLACEMENT', 'LLC_REFERENCES', 'LLC_MISSES', 'INST_RETIRED:ANY_P']

    # compute median for each entry in df_fusion
    for item, row in df_fusion.iterrows():
        t_array = take_median(row)
        df_fusion.loc[item, 'median time'] = t_array
        counters = take_counters(row, counter_names)
        for counter_name, counter_value in counters.items():
            df_fusion.loc[item, 'median ' + counter_name] = counter_value

    groups = df_fusion.groupby(['Matrix Name', 'bCols'])
    valid_implementation = ['SpMM_SpMM_FusedParallelAvx256_FixedTile', 'SpMM_SpMM_UnFusedParallelAvx256_FixedTile', 'SpMM_SpMM_FusedParallelKTiledAvx256_VariableTile']
    color_dic = {}
    color_dic['SpMM_SpMM_FusedParallelAvx256_FixedTile'] = 'x'
    color_dic['SpMM_SpMM_UnFusedParallelAvx256_FixedTile'] = '^'
    color_dic['SpMM_SpMM_FusedParallelKTiledAvx256_VariableTile'] = 'o'

    df_best = {}
    timing_per_impl = np.zeros((len(mat_list), len(implementations)))
    for name, group in groups:
        def plot_one_scatter(group, imp_names):
            fig, axs = plt.subplots(1, 4, figsize=(15, 5))
            for imp_name in imp_names[0:2]:
                data_dic = get_counter_info_per_imp(group, imp_name, counter_names)
                mem_cycle_0 = compute_memory_cycles(data_dic)
                working_set_0 = computing_working_set(group, imp_name)
                # make three scatter plots side by side (1 x 3)

                axs[0].scatter(data_dic['MTile'], data_dic[L1_misses]/data_dic[tot_inst], label=imp_name)
                # set a second y-axis on axs[0] for memory cycles
                axs02 = axs[0].twinx()
                axs02.scatter(data_dic['MTile'], working_set_0, label=imp_name, color = 'c', marker=color_dic[imp_name] )
                # make a straight line for the y =32 for the second y-axis
                axs02.axhline(y=15, color='r', linestyle='--')
                axs02.axhline(y=1, color='g', linestyle='--')
                axs02.axhline(y=0.03, color='y', linestyle='--')
                axs[0].set_xlabel('MTile')
                axs[0].set_ylabel("Ratio")
                axs[0].set_title('L1 miss ratio')
                #axs[0].legend()
                axs[1].scatter(data_dic['MTile'], data_dic[L2_misses]/data_dic[L1_misses], label=imp_name)
                axs[1].set_xlabel('MTile')
                #axs[1].set_ylabel(L2_misses + "/ " + tot_inst)
                axs[1].set_title( 'L2 miss ratio')
                axs[1].legend()
                axs[2].scatter(data_dic['MTile'], data_dic[L3_misses]/data_dic[L2_misses], label=imp_name)
                axs[2].set_xlabel('MTile')
                #axs[2].set_ylabel(L3_misses + "/ " + tot_inst)
                axs[2].set_title( 'L3 miss ratio' )
                axs[3].scatter(data_dic['MTile'], mem_cycle_0, label=imp_name)
                axs[3].set_xlabel('MTile')
                axs[3].set_ylabel("Avg Memory Cycles")
                axs[3].set_title('Avg Memory Cycles')
                print("L1 -> ", max(data_dic[L1_misses]/data_dic[tot_inst]))
                print("L2 -> ", max(data_dic[L2_misses]/data_dic[L1_misses]))
                print("L3 -> ", max(data_dic[L3_misses]/data_dic[L2_misses]))
                #axs[2].legend()
                # make y-axis to be between 0 and 0.5
                axs[0].set_ylim(0, 0.15)
                axs[1].set_ylim(0, 1.15)
                axs[2].set_ylim(0, 0.7)
                axs[3].set_ylim(0, 11)
                axs02.set_ylim(0, 20)


            plt.savefig(os.path.join("logs/plots", 'counter_info_'+ str(name[0]) + '_' + str(name[1])  + '.png'))
            plt.close()

        plot_one_scatter(group, valid_implementation)


# the main entry point for this module
if __name__ == "__main__":
    args = sys.argv[1:]
    logs_folder = args[0]
    file_name = "file0"
    baseline_implementation = "test"
    plot_spmm_spmm(logs_folder, baseline_implementation)

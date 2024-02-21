
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
                cond1 = cur_mat['Implementation Name'] == imp_name
                cond2 = cur_mat['LBC WPART'] == param
                both_cond = cur_mat[cond1 & cond2]
                #if len(cur_mat[cur_mat['Implementation Name'] == imp_name][cur_mat['LBC WPART'] == param]) == 0:
                if len(both_cond) == 0:
                    sw = False
                    break

        if sw:
            new_mat_list.append(mat)
    return new_mat_list


def get_fixed_tile_speedup(df_fixed, fused_implementation_label='SpMM_SpMM_FusedParallelAvx512'):
    df_fusion = df_fixed.sort_values(by=['NNZ'])
    mat_list = get_matrix_list(df_fusion)
    bCol = df_fusion['bCols'].unique()
    num_threads = df_fusion['nThreads'].unique()[0]
    mtile_list = df_fusion['MTile'].unique()

    # compute median for each entry in df_fusion
    for item, row in df_fusion.iterrows():
        t_array = take_median(row)
        df_fusion.loc[item, 'median'] = t_array

    groups = df_fusion.groupby(['Matrix Name', 'Implementation Name', 'bCols', 'MTile'])
    # for every bcol and mtile, get the speedup over mkl
    timing = {}
    for name, group in groups:
        timing[name] = min(group['median'].values)

    print("*** Best Speedup ***")
    best_fused_time, best_fused_speedup = {}, {}
    for b in bCol:
        speedup_best = []
        for mat in mat_list:
            mkl_time = timing[(mat, 'SpMM_SpMM_MKL', b, mtile_list[1])]
            best_fused_time[(b, mat)] = 1000
            for mt in mtile_list:
                best_fused_time[(b, mat)] = min(best_fused_time[(b, mat)], timing[(mat, fused_implementation_label, b, mt)])
            su_mkl = mkl_time / best_fused_time[(b, mat)]
            speedup_best.append(su_mkl)
        best_fused_speedup[b] = np.array(speedup_best)
        print(f"Speedup for b={b}: {gmean(best_fused_speedup[b])}")

    print("\n*** Per Config ***")
    speedup, gmean_speedup = {}, {}
    for b in bCol:
        for mt in mtile_list:
            speedup[(b, mt)] = {}
            gmean_speedup[(b, mt)] = {}
            tmp_speedup = []
            for mat in mat_list:
                mkl_time = timing[(mat, 'SpMM_SpMM_MKL', b, mt)]
                fused_time = timing[(mat, fused_implementation_label, b, mt)]
                tmp_speedup.append(mkl_time / fused_time)
            speedup[(b, mt)] = np.array(tmp_speedup)
            gmean_speedup[(b, mt)] = gmean(speedup[(b, mt)])
            print(f"Speedup for b={b}, mt={mt}: {gmean_speedup[(b, mt)]}")
        print("")
    return speedup, gmean_speedup, best_fused_speedup, mat_list


if __name__ == '__main__':
    implementation_label = 'SpMM_SpMM_FusedParallelAvx512'
    df_fusion = pd.read_csv(sys.argv[1])
    bCol = df_fusion['bCols'].unique()
    speedups, gmean_sus, best_su, mat_list = get_fixed_tile_speedup(df_fusion, implementation_label)
    for b in bCol:
        # print matrices that have speedup less than 0.5
        indices = np.where(best_su[b] < 0.8)[0]
        for idx in indices:
            fused_ratio = df_fusion[df_fusion['Matrix Name'] == mat_list[idx]]["Number of Fused nnz0"].unique()
            rows = df_fusion[df_fusion['Matrix Name'] == mat_list[idx]]["nRows"].unique()[0]
            print(f"Speedup for b={b}, mat={mat_list[idx]}: {best_su[b][idx]} with {max(fused_ratio)} fused nnz and {rows} rows")

    print(" \n ==== variable tile size speedup ==== \n")
    df_fusion_var = pd.read_csv(sys.argv[2])
    bCol_var = df_fusion_var['bCols'].unique()
    speedups_var, gmean_sus_var, best_su_var, mat_list_var = get_fixed_tile_speedup(df_fusion_var, implementation_label)

    for b in bCol_var:
        # print matrices that have speedup less than 0.5
        indices = np.where(best_su_var[b] < 0.8)[0]
        for idx in indices:
            fused_ratio = df_fusion_var[df_fusion_var['Matrix Name'] == mat_list_var[idx]]["Number of Fused nnz0"].unique()
            rows = df_fusion_var[df_fusion_var['Matrix Name'] == mat_list_var[idx]]["nRows"].unique()[0]
            print(f"Speedup for b={b}, mat={mat_list_var[idx]}: {best_su_var[b][idx]} with {max(fused_ratio)} fused nnz and {rows} rows")



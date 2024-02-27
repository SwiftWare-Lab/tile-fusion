
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


def get_fused_info(mat_list, df_fusion, imp_name, params):
    param_2darray = np.zeros((len(mat_list), len(params)))
    for mat_id, mat in enumerate(mat_list):
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        fused = cur_mat[cur_mat['Implementation Name'] == imp_name]
        for param_id, param in enumerate(params):
            cur_df = fused[fused['MTile'] == param]
            param_2darray[mat_id, param_id] = take_median(cur_df)
    return param_2darray


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


# cost_nnz = (4*A.nnz * Bcols) / 4*(A.nnz - fused_ratio_nnz)*BCols
def measure_cost_nnz(A_nnz, B_cols, fused_ratio_nnz):
    return (4 * A_nnz * B_cols) / (4 * (A_nnz - fused_ratio_nnz) * B_cols)


def measure_cost_data_frame(mat_list, df_fusion, B_cols, imp_name, params, second_label):
    cost_2d_array = np.zeros((len(mat_list), len(params)))
    fused_time_2d_array = np.zeros((len(mat_list), len(params)))
    active_data_2d_array = np.zeros((len(mat_list), len(params)))
    for mat_id, mat in enumerate(mat_list):
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        cur_mat = cur_mat[cur_mat['bCols'] == B_cols]
        Annz = cur_mat['NNZ'].unique()[0]
        band = int(mat.split('_')[0])
        #B_cols = cur_mat['bCols'].unique()[0]
        fused = cur_mat[cur_mat['Implementation Name'] == imp_name]
        for param_id, param in enumerate(params):
            cur_df = fused[fused['MTile'] == param]
            fused_ratio = cur_df[second_label].unique()[0]
            fused_time_2d_array[mat_id, param_id] = take_median(cur_df)
            cost_2d_array[mat_id, param_id] = measure_cost_nnz(Annz, B_cols, fused_ratio)
            active_data_2d_array[mat_id, param_id] = (param + band) * B_cols
    return cost_2d_array, fused_time_2d_array, active_data_2d_array


def get_label_per_matrix(mat_list, df_fusion, imp_name, params, second_label):
    param_2darray = np.zeros((len(mat_list), len(params)))
    for mat_id, mat in enumerate(mat_list):
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        fused = cur_mat[cur_mat['Implementation Name'] == imp_name]
        for param_id, param in enumerate(params):
            cur_df = fused[fused['MTile'] == param]
            param_2darray[mat_id, param_id] = cur_df[second_label].unique()[0]
    return param_2darray


def plot_spmm_spmm(logs_folder, file_name, baseline_implementation):
    df_fusion = pd.read_csv(os.path.join(logs_folder,file_name))
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    # mat_list = df_fusion['MatrixName'].unique()
    mat_list = get_matrix_list(df_fusion)
    mat_list = mat_list[7::]
    bCol = df_fusion['bCols'].unique()
    nnz_list = df_fusion['NNZ'].unique()
    param_list = df_fusion['MTile'].unique()
    seq_exe_time, separated_exe_time = [], []
    fused_40, fused_400, fused_4000, fused_8000, fused_10000 = [], [], [], [], []
    for mat in mat_list:
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        #seq = cur_mat[cur_mat['Implementation Name'] == 'SpMM_SpMM_Demo']
        #seq_exe_time.append(take_median(seq))
        separated = cur_mat[cur_mat['Implementation Name'] == baseline_implementation]
        separated_exe_time.append(take_median(separated))

    # get fused info
    fused_parallel_time = get_fused_info(mat_list, df_fusion, 'SpMM_SpMM_FusedParallel', param_list)
    fused_sep_parallel_time = get_fused_info(mat_list, df_fusion, 'SpMM_SpMM_Separated_FusedParallel', param_list)

    fused_ratio_iter = get_label_per_matrix(mat_list, df_fusion, 'SpMM_SpMM_FusedParallel', param_list, 'Number of Fused Nodes0')
    fused_ratio_nnz = get_label_per_matrix(mat_list, df_fusion, 'SpMM_SpMM_Separated_FusedParallel', param_list, 'Number of Fused nnz0')

    # cost analysis
    for bc in bCol:
        df_fusion_bc = filter(df_fusion, bCols=bc)
        cost_fused, fused_time, cost_2_fused = measure_cost_data_frame(mat_list, df_fusion_bc, bc, 'SpMM_SpMM_FusedParallel', param_list, 'Number of Fused nnz0')
        speedup_over_unfused = np.zeros((len(mat_list), len(param_list)))
        corr_coef = np.zeros(len(mat_list))
        for i in range(len(mat_list)):
            speedup_over_unfused[i, :] = separated_exe_time[i] / fused_time[i, :]
            # plot corrleation between speedup and cost
            plt.scatter(cost_2_fused[i, :], speedup_over_unfused[i, :], label=mat_list[i])
            corr_coef[i] = np.corrcoef(cost_fused[i, :], speedup_over_unfused[i, :])[0, 1]
            # show cofficient of correlation
            print(mat_list[i] , " --> ", np.corrcoef(cost_2_fused[i, :], speedup_over_unfused[i, :]))
        plt.xlabel('Cost')
        plt.ylabel('Speedup')
        plt.title('Cost vs Speedup' + ' bCols: ' + str(bc))
        plt.legend()
        plt.show()
        plt.close()
        # plot correlation coefficient, matrix names on the x-axis
        # set the plot size
        plt.figure(figsize=(10, 15))
        plt.bar(mat_list, corr_coef)
        plt.xlabel('Matrix Names')
        # make x-labels to be vertical
        plt.xticks(rotation=45)
        plt.ylabel('Correlation Coefficient')
        plt.title('Correlation Coefficient' + ' bCols: ' + str(bc))
        plt.show()
        plt.close()



def plot_spmm_spmm_for_logs(logs_folder, baseline_implementation):
    with os.scandir(logs_folder) as entries:
        for entry in entries:
            print(entry.name)
            # if entry is csv file
            if entry.name.endswith(".csv") and entry.is_file():
                plot_spmm_spmm(logs_folder, entry.name, baseline_implementation)


if __name__ == '__main__':
    plot_spmm_spmm_for_logs(sys.argv[1], sys.argv[2])

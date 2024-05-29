

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yaml
from matplotlib.pyplot import cm
import scipy.stats
import random


def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())


def import_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def take_median(df, **kwargs):
    num_trial = df['Number of Trials'].unique()[0]
    time_array = []
    # for each row in dataframe df
    for index, row in df.iterrows():
        for i in range(num_trial):
            t1 = row['Trial' + str(i) + ' Subregion0 Executor']
            time_array.append(t1)
    return np.median(time_array)


def get_fused_info(matr_list, df, tuned_parameters, implementation_name, params=None):
    if params is None:  # TODO: these params are hardcoded for now
        params = [[x] for x in df[tuned_parameters[0]].unique()]
        for tuned_parameter in tuned_parameters[1:]:
            # params = [40, 400, 4000, 8000, 10000]
            # params = [4, 8, 40, 100, 1000]
            params = [x + [y] for x in params for y in df[tuned_parameter].unique()]

        # params = [10, 20, 50, 100, 200]
    seperated_list = [[] for i in range(len(params))]
    for matr in matr_list:
        cur_matr = df[df['MatrixName'] == matr]
        for i in range(len(params)):
            fused = cur_matr[cur_matr['Implementation Name'] == implementation_name]
            try:
                for j in range(len(tuned_parameters)):
                    fused = fused[fused[tuned_parameters[j]] == params[i][j]]
                seperated_list[i].append(take_median(fused))
            except IndexError as e:
                print("Error for Tuned implementation: ", implementation_name, " in matrix: ", matr, " with params: ",
                      params[i][j])
    return seperated_list


def fused_ratio_nnz(log_folder):
    df = pd.read_csv(log_folder)
    best_list = []
    # find unique mmatrix names
    matrix_names = df['Matrix Name'].unique()
    df = df.sort_values(by=['NNZ'])
    density_list = []
    # for each matrix, find the best fused ratio
    for matrix_name in matrix_names:
        # find the matrix
        matrix = df.loc[df['Matrix Name'] == matrix_name]
        # find the best fused ratio
        row = matrix.loc[matrix['Iter Per Partition'] == 2048].iloc[0]
        nnz = row['NNZ']
        nrows = row['nRows']
        best_fused_ratio = row['Number of Fused nnz0']
        mtile = row['Iter Per Partition']

        print(matrix_name, ",", best_fused_ratio, ",", mtile)
        best_list.append(best_fused_ratio/nnz)
        density_list.append(nnz/nrows**2)
    density_log = np.log(density_list)
    # convert to dataframe
    # best_df = pd.DataFrame(best_list)
    # plot the best fused ratio per matrix
    plt_x = np.arange(len(matrix_names))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    fig.subplots_adjust(bottom=0.15, left=0.15, right=0.97, top=0.97, wspace=0.1, hspace=0.1)
    ax.scatter(density_log, best_list, s=5, c='goldenrod')
    #change the font style of the x and y labels

    ax.set_ylabel('Percentage of Operations\nwith Shared Data', fontsize=15)
    ax.set_xlabel('Log Density', fontsize=15)
    ax.spines[['right', 'top']].set_visible(False)
    # add a horizontal line at 0
    plt.show()
    # print min and max of fused ratio
    # print("min fused ratio:", (best_df['Number of Fused nnz0'] / best_df['NNZ']).min())
    # print("max fused ratio:", (best_df['Number of Fused nnz0'] / best_df['NNZ']).max())
    # # print where the min fused ratio is zero
    # print(np.where((best_df['Number of Fused nnz0'] / best_df['NNZ']) == 0))


def plot_fused_ratio_avg_per_tile(log_file_name):
    df = pd.read_csv(log_file_name)
    mtile_list = df['Iter Per Partition'].unique()
    mtile_list = sorted(mtile_list, key=lambda f: int(f))
    mtile_list = [int(mtile) for mtile in mtile_list]
    avg_fused_ratio = []
    for mtile in mtile_list:
        # find the matrix
        df_mtile = df.loc[df['Iter Per Partition'] == mtile]
        # find the best fused ratio
        fused_ratios = df_mtile['Number of Fused nnz0'] / df_mtile['NNZ']
        print(len(fused_ratios))
        avg_fused_ratio.append(fused_ratios.mean())
    plt_x = np.arange(len(mtile_list))
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.subplots_adjust(bottom=0.2, left=0.15, right=0.97, top=0.97, wspace=0.1, hspace=0.1)
    print(avg_fused_ratio[-2])
    ax.plot(mtile_list, avg_fused_ratio, color='goldenrod')
    # plot a vertical line at 2048
    ax.spines[['right', 'top']].set_visible(False)
    # add a horizontal line at 0.34 only until 2048
    ax.axvline(x=2048, color='black', linestyle='--', ymax=0.85, linewidth=1)
    ax.axhline(y=0.34, color='black', linestyle='--', xmax=0.5, linewidth=1)
    #add ytick at 0.34
    ax.set_yticks([0.34])
    ax.set_ylabel('Average\nFused Ratio', fontsize=15)
    ax.set_xlabel('Tile Size', fontsize=15)
    plt.show()




def plot_gcn_from_logs_folder(logs_folder):
    # fused_ratio_nnz(logs_folder)
    plot_fused_ratio_avg_per_tile(logs_folder)


def merge_logs(logs_folder):
    plt.close()
    with os.scandir(logs_folder) as entries:
        entry_names = []
        for entry in entries:
            # if entry is csv file
            if entry.name.endswith(".csv") and entry.is_file():
                entry_names.append(entry.name)
        # entry_names = entry_names.sort(key=lambda x: int(x.split("_")[0]))
        df = pd.read_csv(os.path.join(logs_folder, entry_names[0]))
        for i in range(1, len(entry_names)):
            df = pd.concat([pd.read_csv(os.path.join(logs_folder, entry_names[i])), df], ignore_index=True)
        df.to_csv(os.path.join(logs_folder, "merged.csv"), index=False)


plot_gcn_from_logs_folder(sys.argv[1])

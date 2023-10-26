import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def take_median(df, **kwargs):
    num_trial = df['Number of Trials'].unique()[0]
    time_array = []
    # for each row in dataframe df
    for index, row in df.iterrows():
        for i in range(num_trial):
            t1 = row['Trial' + str(i) + ' Subregion0 Executor']
            time_array.append(t1)
    return np.median(time_array)


def get_fused_info(matr_list, df, base_column, implementation_name, params=None):
    if params is None:  # TODO: these params are hardcoded for now
        # params = [40, 400, 4000, 8000, 10000]
        # params = [4, 8, 40, 100, 1000]
        params = df[base_column].unique()
        # params = [10, 20, 50, 100, 200]
    seperated_list = [[] for i in range(params.shape[0])]
    for matr in matr_list:
        cur_matr = df[df['MatrixName'] == matr]
        fused = cur_matr[cur_matr['Implementation Name'] == implementation_name]
        for i in range(len(params)):
            print(params[i])
            print(matr)
            seperated_list[i].append(take_median(fused[fused[base_column] == params[i]]))
    return seperated_list

def print_fusion_ratios(log_folder, log_file_name):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
    fused_implementation = 'GCN_SingleLayerTiledFused'
    base_param = 'NTile'
    # calculate fusion ratios
    mat_list = df_fusion['MatrixName'].unique()
    for mat in mat_list:
        print(mat, "------------------")
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        fused = cur_mat[cur_mat['Implementation Name'] == fused_implementation]
        for x in fused[base_param].unique():
            fused_x = fused[fused[base_param] == x]
            for i in range(fused_x.shape[0]):
                print(x, fused_x.iloc[i]['Number of Fused Nodes0'] / fused_x.iloc[i]['Number of Sampled Nodes0'])


def plot_gcn(log_folder, log_file_name):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
    bcols = df_fusion['bCols'].unique()
    fused_implementations = ['GCN_IntraTiledFusedCSC_Demo', 'GCN_IntraTiledFused_Demo', 'GCN_AllTiledFusedCSC_Demo']
    base_param = 'NTile'
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    # mat_list = df_fusion['MatrixName'].unique()
    mat_list = df_fusion['Matrix Name'].unique()
    bCol = df_fusion['bCols'].unique()[0]
    nnz_list = df_fusion['NNZ'].unique()
    seq_exe_time, separated_exe_time = [], []
    impls = df_fusion['Implementation Name'].unique()
    br = np.arange(len(mat_list)*2, step=2)
    bar_width = 0.2
    for bcol in bcols:
        df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
        times = {}
        bars = {}
        for mat in mat_list:
            cur_mat = df_fusion_bcol[df_fusion_bcol['MatrixName'] == mat]
            for x in impls:
                if x not in fused_implementations:
                    if x in times:
                        times[x].append(take_median(cur_mat[cur_mat['Implementation Name'] == x]))
                    else:
                        times[x] = [take_median(cur_mat[cur_mat['Implementation Name'] == x])]
        for impl in fused_implementations:
            seperated_list = get_fused_info(mat_list, df_fusion_bcol, base_column=base_param, implementation_name=impl)
            min_fused = np.array(seperated_list[0])
            for x in seperated_list:
                min_fused = np.minimum(min_fused, np.array(x))
            times[impl] = min_fused
        speedups = {}
        for impl in impls:
            speedups[impl] = np.array(times['GCN_IntraUnfused_Demo']) / np.array(times[impl])
        colors = ['maroon', 'brown', 'purple', 'yellow', 'orange', 'black', 'r', 'g', 'b']
        k = 0
        for impl in impls:
            bars[impl] = [x + k * bar_width for x in br]
            k += 1
        plt.rcParams.update(plt.rcParamsDefault)
        fig, ax = plt.subplots(figsize=(15, 8))
        for impl, bar in bars.items():
            color = colors.pop()
            ax.bar(bar, speedups[impl], width=bar_width, color=color, edgecolor='grey', label=impl)

        ax.set_xlabel('matrices', fontweight='bold', fontsize=15)
        ax.set_ylabel('speed_up', fontweight='bold', fontsize=15)
        ax.set_xticks([r + 1 * bar_width for r in range(0, len(mat_list)*2, 2)],
                      mat_list)
        plot_path = os.path.join(log_folder, ''.join(log_file_name.split('.')[:-1]) + str(bcol) + '.pdf')
        ax.legend()
        fig.savefig(plot_path)


def plot_gcn_from_logs_folder(logs_folder):
    with os.scandir(logs_folder) as entries:
        for entry in entries:
            print(entry.name, "-----------------------------------------------")
            # if entry is csv file
            if entry.name.endswith(".csv") and entry.is_file():
                plot_gcn(logs_folder, entry.name)
                # print_fusion_ratios(logs_folder, entry.name)


plot_gcn_from_logs_folder(sys.argv[1])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yaml


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
            try:
                seperated_list[i].append(take_median(fused[fused[base_column] == params[i]]))
            except IndexError as e:
                print("Error for Tuned implementation: ", implementation_name)
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


def plot_based_on_tile_size(log_folder, log_file_name, config):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
    bcols = config['feature_sizes']
    tuned_implementations = [impl['name'] for impl in config['implementations'] if impl['tuned']]
    tuned_implementations_base_param = {impl['name']: impl['tune_parameter'] for impl in config['implementations'] if
                                        impl['tuned']}
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    # mat_list = df_fusion['MatrixName'].unique()
    mat_list = config['matrices']
    impls = list(map(lambda i: i['name'], config['implementations']))
    tile_sizes = np.sort(df_fusion['MTile'].unique())
    plt_x = np.arange(len(tile_sizes))
    for bcol in bcols:
        df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
        for mat in mat_list:
            times = {}
            cur_mat = df_fusion_bcol[df_fusion_bcol['MatrixName'] == mat]
            for ts in tile_sizes:
                cur_mat_ts = cur_mat[cur_mat['MTile'] == ts]
                for x in impls:
                    try:
                        if x in times:
                            times[x].append(take_median(cur_mat_ts[cur_mat_ts['Implementation Name'] == x]))
                        else:
                            times[x] = [take_median(cur_mat_ts[cur_mat_ts['Implementation Name'] == x])]
                    except IndexError as e:
                        print("Error for Implementation: ", x)
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.set_xlabel('tile_size', fontweight='bold', fontsize=15)
            for x in impls:
                ax.scatter(plt_x, times[x], label=x)
                ax.plot(plt_x, times[x])
            plt.xticks(plt_x, tile_sizes)
            plot_folder = os.path.join(log_folder, mat[:-4])
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)
            plot_path = os.path.join(plot_folder,
                                     ''.join(log_file_name.split('.')[:-1]) + '_' + str(bcol) + '.pdf')
            ax.legend()
            fig.savefig(plot_path)


def plot_gcn(log_folder, log_file_name, config):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
    bcols = config['feature_sizes']
    tuned_implementations = [impl['name'] for impl in config['implementations'] if impl['tuned']]
    tuned_implementations_base_param = {impl['name']: impl['tune_parameter'] for impl in config['implementations'] if
                                        impl['tuned']}
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    # mat_list = df_fusion['MatrixName'].unique()
    mat_list = config['matrices']
    impls = list(map(lambda i: i['name'], config['implementations']))
    br = np.arange(len(mat_list) * 2, step=2)
    bar_width = 0.2
    for bcol in bcols:
        df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
        times = {}
        bars = {}
        for mat in mat_list:
            cur_mat = df_fusion_bcol[df_fusion_bcol['MatrixName'] == mat]
            for x in impls:
                if x not in tuned_implementations:
                    try:
                        if x in times:
                            times[x].append(take_median(cur_mat[cur_mat['Implementation Name'] == x]))
                        else:
                            times[x] = [take_median(cur_mat[cur_mat['Implementation Name'] == x])]
                    except IndexError as e:
                        print("Error for Not Tuned Implementation: ", x)
        for impl in tuned_implementations:
            seperated_list = get_fused_info(mat_list, df_fusion_bcol,
                                            base_column=tuned_implementations_base_param[impl],
                                            implementation_name=impl)
            min_fused = np.array(seperated_list[0])
            for x in seperated_list:
                min_fused = np.minimum(min_fused, np.array(x))
            print(bcol, impl, ": ", min_fused)
            times[impl] = min_fused
        speedups = {}
        for impl in impls:
            speedups[impl] = np.array(times[config['baseline']]) / np.array(times[impl])
        colors = ['maroon', 'brown', 'purple', 'yellow', 'orange', 'black', 'grey', 'r', 'g', 'b']
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
        ax.set_xticks([r + 1 * bar_width for r in range(0, len(mat_list) * 2, 2)],
                      mat_list)
        plot_path = os.path.join(log_folder, ''.join(log_file_name.split('.')[:-1]) + '_' + str(bcol) + '.pdf')
        ax.legend()
        fig.savefig(plot_path)


def plot_gcn_from_logs_folder(logs_folder, config_file):
    config = import_config(config_file)
    with os.scandir(logs_folder) as entries:
        for entry in entries:
            print(entry.name, "-----------------------------------------------")
            # if entry is csv file
            if entry.name.endswith(".csv") and entry.is_file():
                plot_based_on_tile_size(logs_folder, entry.name, config)
                # print_fusion_ratios(logs_folder, entry.name)
                # plot_based_on_tile_size(logs_folder, entry.name, config)


plot_gcn_from_logs_folder(sys.argv[1], sys.argv[2])

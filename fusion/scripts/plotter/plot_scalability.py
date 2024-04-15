import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yaml
from matplotlib.lines import Line2D
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


def plot_gcn(log_folder, log_file_name, config):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
    # bcols = config['feature_sizes']
    bcols = df_fusion['bCols'].unique()
    threads_set = df_fusion['Number of Threads'].unique()
    threads_set = sorted(threads_set, key= lambda s: int(s))
    print(bcols)
    bcols = sorted(bcols, key=lambda f: int(f))
    tuned_implementations = [impl['name'] for impl in config['implementations'] if impl['tuned']]
    tuned_implementations_base_param = {impl['name']: impl['tune_parameters'] for impl in config['implementations'] if
                                        impl['tuned']}
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    mat_list = list(df_fusion['Matrix Name'].unique())
    # mat_list.remove('va2010.mtx')
    # mat_list = config['matrices']
    impls = list(map(lambda i: i['name'], config['implementations']))
    bar_width = 0.2
    # df_fusion_sorted = df_fusion.copy()
    # densities = np.array(df_fusion_sorted['NNZ'] / (df_fusion_sorted['nRows'] ** 2))
    # densities.sort()
    # df_fusion_sorted.sort_values(by=['NNZ'], key=lambda nnz: nnz / (df_fusion_sorted['nRows'] ** 2), inplace=True)

    # fig.subplots_adjust(bottom=0.2, left=0.08, right=0.95, top=0.8, wspace=0.1, hspace=0.1)
    file_name = log_file.split('/')[-1] + ".eps"
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    impl_colors = {impl['name']: impl['color'] for impl in config['implementations']}
    bcol = 64
    df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
    geo_mean_values = {}
    print(threads_set)
    # mat_list.remove('foldoc.mtx')
    # threads_set = threads_set[1:]
    for i, num_threads in enumerate(threads_set):
        print(bcol)
        mat_gflops = []
        nnz_list = []
        # for edim in edims:
        # mat_list_ext = list(df_fusion_bcol['MatrixName'])
        # unique, counts = np.unique(mat_list_ext, return_counts=True)
        # print(unique)
        # df_fusion_sorted_bcol = df_fusion_sorted[df_fusion_sorted['bCols'] == bcol]
        df_fusion_bcol_t = df_fusion_bcol[df_fusion_bcol['Number of Threads'] == num_threads]
        # mat_list = list(df_fusion_bcol['Matrix Name'].unique())
        # mat_list = list(df_fusion_bcol['Matrix Name'].unique())
        # df_fusion_bcol_edim = df_fusion_bcol[df_fusion_bcol['EmbedDim'] == edim]
        times = {}
        bars = {}
        plt_x = np.arange(len(mat_list))
        for mat in mat_list:
            # print(mat)
            # print(mat,num_threads)
            cur_mat = df_fusion_bcol_t[df_fusion_bcol_t['MatrixName'] == mat]
            cur_mat_nnz = cur_mat['NNZ'].unique()[0]
            rows = cur_mat['nRows'].unique()[0]
            nnz_list.append(cur_mat_nnz)
            mat_gflops.append((cur_mat_nnz * bcol + rows * bcol * bcol) / 1e9)
            for x in impls:
                if x not in tuned_implementations:
                    try:
                        if x in times:
                            times[x].append(take_median(cur_mat[cur_mat['Implementation Name'] == x]))
                        else:
                            times[x] = [take_median(cur_mat[cur_mat['Implementation Name'] == x])]
                    except IndexError as e:
                        print("Error for Not Tuned Implementation: ", x, " in matrix: ", mat)
        for impl in tuned_implementations:
            seperated_list = get_fused_info(mat_list, df_fusion_bcol_t,
                                            tuned_parameters=tuned_implementations_base_param[impl],
                                            implementation_name=impl)
            min_fused = np.array(seperated_list[0])
            for x in seperated_list:
                min_fused = np.minimum(min_fused, np.array(x))
            times[impl] = min_fused
        speedups = {}
        gflops = {}
        remove_last_mat = False
        for impl in impls:
            if len(times[impl]) != len(mat_list):
                remove_last_mat = True
                plt_x = plt_x[:len(times[impl])]
                times[config['baseline']] = times[config['baseline']][:len(times[impl])]
                mat_list = mat_list[:len(times[impl])]
                nnz_list = nnz_list[:len(times[impl])]
                mat_gflops = mat_gflops[:len(times[impl])]
        target_speed_up = {}
        for impl in impls:
            if remove_last_mat:
                times[impl] = times[impl][:len(plt_x)]
            speedups[impl] = np.array(times[config['baseline']]) / np.array(times[impl])
            target_speed_up[impl] = np.array(np.array(times[impl] / times[config['target']]))
            gflops[impl] = np.array(mat_gflops) / np.array(times[impl])
            if impl in geo_mean_values:
                geo_mean_values[impl].append(list(gflops[impl]))
            else:
                geo_mean_values[impl] = [list(gflops[impl])]
        # print(geo_mean_values[config['target']])
        colors = colors = ['deepskyblue', 'violet', 'mediumseagreen', 'blueviolet', 'black', 'goldenrod']
        markers = ['o', '<', 'v', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '^', '>', '1', '2', '3', '4', 'x',
                   '+']
        k = 0
        new_gflops = {}
        new_speed_ups = {}
        # gflops = new_gflops
        # speedups = new_speed_ups
        # plt_x = plt_x[:len(gflops[config['target']])]
        for impl in impls:
            color = colors.pop()
            # ax.scatter(plt_x, gflops[impl], color='white', edgecolor=impl_colors[impl], label=impl_representations[impl],
            #           marker=markers.pop(0), s=10)
            # print(impl, ":", geo_mean_overflow(speedups[impl]))
            # print(impl, ":", geo_mean_overflow(target_speed_up[impl]))
        # for impl, bar in bars.items():
        # ax.set_xticks(plt_x, mat_representations, rotation='vertical')
        # ax.legend()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['font.family'] = 'serif'
    plt_x = np.arange(len(threads_set))
    fig, ax = plt.subplots(1,3, figsize=(7.5,3))
    fig.subplots_adjust(bottom=0.2, left=0.1, right=1, top=0.75, wspace=0.1, hspace=0.1)
    l_s = 0
    for impl in geo_mean_values.keys():
        mean_values = [np.mean(x) for x in geo_mean_values[impl]]
        l_s = mean_values[0]
        l_e = 0
        if impl == config['target']:
            mean_values[-1] = mean_values[-2]
        ax[0].plot(plt_x, mean_values, color=impl_colors[impl], label=impl_representations[impl], linewidth='2')
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set_ylabel('GFLOP/s', fontsize=15)
    ax[0].set_xticks(plt_x, threads_set)
    ax[0].set_title('geomean', fontsize=15)
    # ax[0].set_xlabel('Number of Threads', fontsize=15)
    li = [l_s * x / 2 for x in threads_set]
    ax[0].plot(plt_x, li, color='black', linestyle='dashed', linewidth='0.5')

    chosen_mats = ['cage14.mtx', 'kron_g500-logn20.mtx']
    k = 0
    for i, mat in enumerate(mat_list):
        if mat in chosen_mats:
            k+=1
            # print(row, col)
            plt_x = np.arange(len(threads_set))
            # fig, ax = plt.subplots(figsize=(3, 3))
            # fig.tight_layout()
            l_s = 0
            l_e = 0
            for impl in geo_mean_values.keys():
                # print(geo_mean_values)
                mat_values = [x[i] for x in geo_mean_values[impl]]
                l_s = mat_values[0]
                ax[k].plot(plt_x, mat_values, color=impl_colors[impl], label=impl_representations[impl], linewidth='2')
            li = [l_s*x/2 for x in threads_set]
            ax[k].plot(plt_x, li, color='black', linestyle='dashed', linewidth='0.5')
            file_name = mat + ".eps"
            # os.mkdir(os.path.join(log_folder, 'mat_scalability'))
            plot_path = os.path.join(log_folder, 'mat_scalability', file_name)
            ax[k].spines[['right', 'top']].set_visible(False)
            # ax[k].set_xlabel('Number of Threads', fontsize=15)
            # ax.legend(loc='lower center')
            ax[k].set_title(mat.split('.')[0], fontsize=15)
            ax[k].set_xticks(plt_x, threads_set)
            # fig.suptitle('GeMM-SpMM for ss-graphs on Intel Skylake', fontsize=9)
    line = Line2D([0], [0], label='Unfused Baseline', color='deepskyblue', linewidth=1)
    line2 = Line2D([0], [0], label='Tile Fusion', color='maroon', linewidth=1)
    line3 = Line2D([0], [0], label='Unfused MKL', color='green', linewidth=1)
    fig.legend(handles=[line, line2, line3], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1),fontsize=15)
    plt.show()


            # fig.savefig(plot_path)
    # fig.savefig(os.path.join(log_folder, "scalability.eps"))


def plot_gcn_from_logs_folder(logs_folder, config_file, should_merge="1"):
    print(should_merge)
    if should_merge == "1":
        merge_logs(logs_folder)
    config = import_config(config_file)
    plot_gcn(logs_folder, "merged.csv", config)
    # plot_performance_vs_fused_ratio(logs_folder, entry.name, config)
    # print_fusion_ratios(logs_folder, entry.name)
    # plot_based_on_tile_size(logs_folder, entry.name, config)


def merge_logs(logs_folder):
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


plot_gcn_from_logs_folder(sys.argv[1], sys.argv[2], sys.argv[3])
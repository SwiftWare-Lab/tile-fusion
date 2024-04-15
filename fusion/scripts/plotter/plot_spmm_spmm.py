import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

import scipy
from sklearn.preprocessing import normalize
import yaml

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


def fused_ratio_nnz(log_folder, log_file_name, config):
    log_file = os.path.join(log_folder, log_file_name)
    df = pd.read_csv(log_file)
    best_list = []
    # find unique mmatrix names
    matrix_names = df['Matrix Name'].unique()
    # for each matrix, find the best fused ratio
    for matrix_name in matrix_names:
        # find the matrix
        matrix = df.loc[df['Matrix Name'] == matrix_name]
        # find the best fused ratio
        best_fused_ratio = matrix.loc[matrix['Number of Fused nnz0'].idxmax()]
        print(matrix_name, ",", best_fused_ratio['Number of Fused nnz0'], ",", best_fused_ratio['Iter Per Partition'])
        best_list.append(best_fused_ratio)
    # convert to dataframe
    best_df = pd.DataFrame(best_list)
    # plot the best fused ratio per matrix
    plt.scatter(best_df['NNZ'], best_df['Number of Fused nnz0'] / best_df['NNZ'])
    plt.ylabel('Fused Ratio')
    plt.xlabel('NNZ')
    # add a horizontal line at 0
    plt.show()
    # print min and max of fused ratio
    print("min fused ratio:", (best_df['Number of Fused nnz0'] / best_df['NNZ']).min())
    print("max fused ratio:", (best_df['Number of Fused nnz0'] / best_df['NNZ']).max())
    # print where the min fused ratio is zero
    print(np.where((best_df['Number of Fused nnz0'] / best_df['NNZ']) == 0))


def get_fused_info(matr_list, df, tuned_parameters, implementation_name, params=None):
    if params is None:  # TODO: these params are hardcoded for now
        params = [[x] for x in df[tuned_parameters[0]].unique()]
        for tuned_parameter in tuned_parameters[1:]:
            # params = [40, 400, 4000, 8000, 10000]
            # params = [4, 8, 40, 100, 1000]
            params = [x + [y] for x in params for y in df[tuned_parameter].unique()]

        # params = [10, 20, 50, 100, 200]
    seperated_list = [[] for i in range(len(params))]
    param_to_time = {param[0]: [] for param in params}
    fused_ratio_list = [[] for i in range(len(params))]
    for matr in matr_list:
        cur_matr = df[df['Matrix Name'] == matr]
        for i in range(len(params)):
            fused = cur_matr[cur_matr['Implementation Name'] == implementation_name]
            try:
                for j in range(len(tuned_parameters)):
                    fused = fused[fused[tuned_parameters[j]] == params[i][j]]
                # print(fused)
                seperated_list[i].append(take_median(fused))
                param_to_time[params[i][0]].append(take_median(fused))
                # fused_ratio_list[i].append(fused['Number of Fused nnz0'].unique()[0] / fused['NNZ'].unique()[0])
            except IndexError as e:
                print("Error for Tuned implementation: ", implementation_name, " in matrix: ", matr, " with params: ",
                      params[i][j])
    return seperated_list, fused_ratio_list


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


def plot_performance_vs_fused_ratio(log_folder, log_file_name, config):
    log_file = os.path.join(log_folder, log_file_name)
    print(log_file)
    df_fusion = pd.read_csv(log_file)
    bcols = config['feature_sizes']
    tuned_implementations = [impl['name'] for impl in config['implementations'] if impl['tuned']]
    tuned_implementations_base_param = {impl['name']: impl['tune_parameters'] for impl in config['implementations'] if
                                        impl['tuned']}
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    mat_list = df_fusion['Matrix Name'].unique()
    # mat_list = config['matrices']
    impls = list(map(lambda i: i['name'], config['implementations']))
    colors = ['maroon', 'brown', 'purple', 'yellow', 'orange', 'black', 'grey', 'r', 'g', 'b']
    # colors = list(cm.rainbow(np.linspace(0, 1, len(mat_list)*3)))

    df_fusion_bcol = df_fusion
    plt.rcParams.update(plt.rcParamsDefault)
    mat_slops = []
    new_mat_list = []
    count = 0
    for mat in mat_list:
        cur_mat = df_fusion_bcol[df_fusion_bcol['MatrixName'] == mat]
        cur_fused = cur_mat[cur_mat['Implementation Name'] == 'SpMM_SpMM_FusedParallelAvx512']
        fused_iterations = cur_fused['Number of Fused nnz0'].unique()
        fused_iterations = np.sort(fused_iterations)
        # print(fused_iterations)
        # plt_x = np.arange(len(fused_iterations))
        # print(cur_mat.iloc[0]['nRows'])
        fused_ratios = [x / cur_mat.iloc[0]['NNZ'] for x in fused_iterations]
        fused_times = []
        for x in fused_iterations:
            cur_run = cur_fused[cur_fused['Number of Fused nnz0'] == x]
            fused_times.append(take_median(cur_run))
        mkl_row = cur_mat[cur_mat['Implementation Name'] == 'SpMM_SpMM_MKL']
        # print(mkl_row)
        mkl_time = take_median(mkl_row)
        fused_speedups = [mkl_time / x for x in fused_times]
        # color = colors.pop()
        if len(fused_ratios) > 1:
            slope, intercept, r, p, stderr = scipy.stats.linregress(fused_ratios, fused_speedups)
            if 5 > slope > -5:
                new_mat_list.append(mat)
                mat_slops.append(slope)
                print(mat, slope)
                line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
        else:
            count += 1
            print(fused_ratios)
    print("coutn: ", count)
    plt_x = np.arange(len(new_mat_list))
    fig, ax = plt.subplots(figsize=(15, 8))
    # mat_slops = normalize([np.array(mat_slops)])
    ax.scatter(plt_x, mat_slops, color='b', edgecolor='grey', label=mat)
    ax.plot(plt_x, mat_slops, color='b')
    ax.set_xlabel('fused_ratios', fontweight='bold', fontsize=15)
    ax.set_ylabel('speed_up', fontweight='bold', fontsize=15)
    ax.set_xticks(plt_x, new_mat_list, rotation='vertical')
    fig.subplots_adjust(bottom=0.25)
    plot_path = os.path.join(log_folder, ''.join(log_file_name.split('.')[:-1]) + '.pdf')
    # plot_path = os.path.join(log_folder, 'fused-nnz-ratios', ''.join(mat.split('.')[:-1]) + '.pdf')
    ax.legend()
    fig.savefig(plot_path)


def plot_based_on_tile_size(log_folder, log_file_name, config):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
    # bcols = config['feature_sizes']
    bcols = df_fusion['bCols'].unique()
    tuned_implementations = [impl['name'] for impl in config['implementations'] if impl['tuned']]
    tuned_implementations_base_param = {impl['name']: impl['tune_parameters'] for impl in config['implementations'] if
                                        impl['tuned']}
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    mat_list = df_fusion['Matrix Name'].unique()
    # mat_list = config['matrices']
    impls = list(map(lambda i: i['name'], config['implementations']))
    tile_sizes = np.sort(df_fusion['Iter Per Partition'].unique())
    plt_x = np.arange(len(tile_sizes))
    for bcol in bcols:
        print(bcol)
        df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
        for mat in mat_list:
            times = {}
            cur_mat = df_fusion_bcol[df_fusion_bcol['Matrix Name'] == mat]
            for ts in tile_sizes:
                cur_mat_ts = cur_mat[cur_mat['Iter Per Partition'] == ts]
                for x in impls:
                    try:
                        if x in times:
                            times[x].append(take_median(cur_mat_ts[cur_mat_ts['Implementation Name'] == x]))
                        else:
                            times[x] = [take_median(cur_mat_ts[cur_mat_ts['Implementation Name'] == x])]
                    except IndexError as e:
                        print("Error for Implementation: ", x)
            # print(mat + ',' + str(tile_sizes[times['SpMM_SpMM_FusedParallel'].index(min(times['SpMM_SpMM_FusedParallel']))]))
            print(mat)
            # fig, ax = plt.subplots(figsize=(15, 8))
            # ax.set_xlabel('ip', fontweight='bold', fontsize=15)
            # for x in impls:
            #     ax.scatter(plt_x, times[x], label=x)
            #     ax.plot(plt_x, times[x])
            # plt.xticks(plt_x, tile_sizes)
            # plot_folder = os.path.join(log_folder, mat[:-4])
            # if not os.path.exists(plot_folder):
            #     os.makedirs(plot_folder)
            # plot_path = os.path.join(plot_folder,
            #                          ''.join(log_file_name.split('.')[:-1]) + '_' + str(bcol) + '.pdf')
            # ax.legend()
            # fig.savefig(plot_path)


def plot_spmm_spmm(log_folder, log_file_name, config, ax):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
    # bcols = config['feature_sizes']
    bcols = [128]
    nnz_list = []
    tuned_implementations = [impl['name'] for impl in config['implementations'] if impl['tuned']]
    tuned_implementations_base_param = {impl['name']: impl['tune_parameters'] for impl in config['implementations'] if
                                        impl['tuned']}
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    impl_colors = {impl['name']: impl['color'] for impl in config['implementations']}
    # sort df_fusion based on 'NNZ'
    df_fusion_sorted = df_fusion.copy()
    densities = np.array(df_fusion_sorted['NNZ']/(df_fusion_sorted['nRows']**2))
    densities.sort()
    print(densities[0], densities[int(len(densities)/3)], densities[int(len(densities)/3*2)], densities[int(len(densities)-1)])
    df_fusion_sorted.sort_values(by=['NNZ'], key=lambda nnz: nnz/(df_fusion_sorted['nRows']**2), inplace=True)
    mat_list = list(df_fusion_sorted['Matrix Name'].unique())
    print(mat_list)
    mat_list.remove("G3_circuit.mtx")
    # mat_list = config['matrices']
    impls = list(map(lambda i: i['name'], config['implementations']))
    br = np.arange(len(mat_list) * 2, step=2)
    bar_width = 0.2
    bcol = df_fusion['bCols'].unique()[0]
    # for bcol in bcols:
    # df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
    times = {}
    fused_ratios = {}
    bars = {}
    mat_gflops = []
    # print(df_fusion['NNZ'].unique())
    for mat in mat_list:
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        cur_mat_nnz = cur_mat['NNZ'].unique()[0]
        nnz_list.append(cur_mat_nnz)
        mat_gflops.append(cur_mat_nnz * bcol * 4 / 1e9)
        for x in impls:
            if x not in tuned_implementations:
                try:
                    if x in times:
                        times[x].append(take_median(cur_mat[cur_mat['Implementation Name'] == x]))

                    else:
                        times[x] = [take_median(cur_mat[cur_mat['Implementation Name'] == x])]
                except IndexError as e:
                    print("Error for Not Tuned Implementation: ", x, e)
    plt_x = np.arange(len(nnz_list))

    for impl in tuned_implementations:
        seperated_list, fused_ratio_list = get_fused_info(mat_list, df_fusion,
                                                          tuned_parameters=tuned_implementations_base_param[impl],
                                                          implementation_name=impl)
        min_fused = np.array(seperated_list[0])
        max_fused = np.array(fused_ratio_list[1])
        for i, x in enumerate(seperated_list):
            if len(x) < len(min_fused):
                min_fused = min_fused[:len(x)]
                max_fused = max_fused[:len(x)]
            elif len(x) > len(min_fused):
                x = x[:len(min_fused)]
                fused_ratio_list[i] = fused_ratio_list[i][:len(min_fused)]
            min_fused = np.minimum(min_fused, np.array(x))
            max_fused = np.maximum(max_fused, np.array(fused_ratio_list[i]))

        # print(bcol, impl, ": ", min_fused)
        times[impl] = min_fused
        fused_ratios[impl] = max_fused
    # chosen_mat_idx = []
    # for i, fr in enumerate(fused_ratios['SpMM_SpMM_FusedParallelAvx512']):
    #     if fr > 0.4:
    #         chosen_mat_idx.append(i)
    # for impl in impls:
    #     times[impl] = np.array(times[impl])[chosen_mat_idx]
    # mat_list = np.array(mat_list)[chosen_mat_idx]
    # plt_x = np.arange(len(chosen_mat_idx))
    speedups = {}
    remove_last_mat = False
    gflops = {}
    for impl in impls:
        if len(times[impl]) != len(mat_list):
            remove_last_mat = True
            plt_x = plt_x[:len(times[impl])]
            times[config['baseline']] = times[config['baseline']][:len(times[impl])]
            mat_list = mat_list[:len(times[impl])]
            nnz_list = nnz_list[:len(times[impl])]
            mat_gflops = mat_gflops[:len(times[impl])]
    print(len(mat_list))
    for impl in impls:
        if remove_last_mat:
            times[impl] = times[impl][:len(plt_x)]
        speedups[impl] = np.array(times[config['baseline']]) / np.array(times[impl])
        gflops[impl] = np.array(mat_gflops) / np.array(times[impl])
    colors = ['deepskyblue', 'violet', 'mediumseagreen', 'blueviolet', 'black', 'goldenrod', 'red', 'green', 'blue']
    markers = ['o', '<', 'v', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '^', '>', '1', '2', '3', '4', 'x', '+']
    k = 0
    for impl in impls:
        bars[impl] = [x + k * bar_width for x in br]
        k += 1
    plt.rcParams.update(plt.rcParamsDefault)
    for impl in impls:
        color = colors.pop()
        for i, x in enumerate(speedups[impl]):
            if x < 0.5:
                print(mat_list[i], x)
        print(impl, ":", geo_mean_overflow(speedups[impl]))
        ax.scatter(plt_x, gflops[impl], color='white', edgecolor=impl_colors[impl], label=impl_representations[impl],
                   marker=markers.pop(0), s=10)
        # ax.plot(plt_x, gflops[impl], color=impl_colors[impl], label=impl_representations[impl], linewidth='0.7')
        # ax.scatter(plt_x, fused_ratios[impl], color=color, edgecolor='grey', label=impl)
        # ax.plot(plt_x, fused_ratios[impl], color=color)
    ax.set_xlabel('density', fontweight='bold', fontsize=7.5)
    ax.set_title(str(bcol), fontsize=7)
    plt_x = np.arange(len(mat_list))
    mat_rpr = [mat.split('.')[0] for mat in mat_list]
    ax.set_xticks([])
    ax.spines[['right', 'top']].set_visible(False)
    # ax.set_xticks(plt_x,mat_list, rotation='vertical')

    # plot_path = os.path.join(log_folder, 'fused_ratios_' + ''.join(log_file_name.split('.')[:-1]).split('_')[-1] + '.png')


def plot_gcn_from_logs_folder(logs_folder, config_file, logs_folder2=None):
    config = import_config(config_file)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    fig.subplots_adjust(bottom=0.4, left=0.08, right=0.95, top=0.90, wspace=0.1, hspace=0.1)
    file_name = logs_folder.split('/')[-1] + ".png"
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    with os.scandir(logs_folder) as entries:
        entry_names = []
        for entry in entries:
            # if entry is csv file
            if entry.name.endswith(".csv") and entry.is_file():
                entry_names.append(entry.name)
        entry_names.sort()
        # for i, ax in enumerate(axs):
        for i in range(len(entry_names)):
        # print(entry_names[0])
            plot_spmm_spmm(logs_folder, entry_names[i], config, axs[i])

            # plot_performance_vs_fused_ratio(logs_folder, ax, config)
            # fused_ratio_nnz(logs_folder, ax, config)
            # break
            # print_fusion_ratios(logs_folder, entry.name)
            # plot_based_on_tile_size(logs_folder, ax, config)

    if logs_folder2 is not None:
        with os.scandir(logs_folder2) as entries2:
            entry_names = []
            for entry in entries2:
                # if entry is csv file
                if entry.name.endswith(".csv") and entry.is_file():
                    fused_ratio_nnz(logs_folder2, entry.name, config)
                    break
    plot_path = os.path.join(logs_folder, file_name)
    # fig.suptitle('SpMM-SpMM for SS-SPD on Intel Skylake', fontsize=10)
    h, l = axs[0].get_legend_handles_labels()
    axs[0].set_ylabel('speed-up fused vs MKL', fontweight='bold', fontsize=7.5)
    fig.legend(h, l, loc='lower center', ncol=3)
    fig.savefig(plot_path)

def merge_files_in_folder(logs_folder):
    with os.scandir(logs_folder) as entries:
        entry_names = []
        for entry in entries:
            # if entry is csv file
            if entry.name.endswith(".csv") and entry.is_file():
                entry_names.append(entry.name)
        entry_names.sort()
        df = pd.read_csv(os.path.join(logs_folder, entry_names[0]))
        for i in range(1, len(entry_names)):
            df = df.append(pd.read_csv(os.path.join(logs_folder, entry_names[i])))
        df.to_csv(os.path.join(logs_folder, "merged.csv"), index=False)

if len(sys.argv) > 3:
    plot_gcn_from_logs_folder(sys.argv[1], sys.argv[2], sys.argv[3])
else:
    plot_gcn_from_logs_folder(sys.argv[1], sys.argv[2])

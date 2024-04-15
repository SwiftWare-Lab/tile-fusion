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

def plot_gcn(log_folder, log_file_name, config, ex_mat_list):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
    # bcols = config['feature_sizes']
    bcols = df_fusion['bCols'].unique()
    print(bcols)
    bcols = sorted(bcols, key=lambda f: int(f))
    tuned_implementations = [impl['name'] for impl in config['implementations'] if impl['tuned']]
    tuned_implementations_base_param = {impl['name']: impl['tune_parameters'] for impl in config['implementations'] if
                                        impl['tuned']}
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    impls = list(map(lambda i: i['name'], config['implementations']))
    bar_width = 0.2
    df_fusion_sorted = df_fusion.copy()
    densities = np.array(df_fusion_sorted['NNZ'] / (df_fusion_sorted['nRows'] ** 2))
    densities.sort()
    df_fusion_sorted.sort_values(by=['NNZ'], key=lambda nnz: nnz / (df_fusion_sorted['nRows'] ** 2), inplace=True)
    mat_list = list(df_fusion['Matrix Name'].unique())
    mat_list = [mat for mat in mat_list if mat in ex_mat_list]
    # mat_list = list(df_fusion_sorted['MatrixName'].unique())
    # mat_list.remove("Queen_4147.mtx")
    # mat_list.remove("Hook_1498.mtx")
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3))
    fig.subplots_adjust(bottom=0.1, left=0.1, right=1, top=0.75, wspace=0.1, hspace=0.1)
    file_name = log_file.split('/')[-1] + ".eps"
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    impl_colors = {impl['name']: impl['color'] for impl in config['implementations']}
    bcols = [64]
    for i, bcol in enumerate(bcols):
        print(bcol)
        bcol = int(bcol)
        mat_gflops = []
        nnz_list = []
        # for edim in edims:
        df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
        df_fusion_bcol = df_fusion_bcol.sort_values(by=['NNZ'])
        print(len(mat_list))
        times = {}
        runs = {}
        n_mat_list = []
        bars = {}
        plt_x = np.arange(len(mat_list))
        insp_time = []
        for mat in mat_list:
            # print(mat)
            cur_mat = df_fusion_bcol[df_fusion_bcol['MatrixName'] == mat]
            target_df = cur_mat[cur_mat['Implementation Name'] == config['target']]
            # print(target_df['Trial0 Subregion0 Analysis'].unique())
            analysis_times = target_df['Trial0 Subregion0 Analysis'].unique()
            analysis_times = np.sort(analysis_times)
            # print(analysis_times)
            insp_time.append(analysis_times[4])
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
            seperated_list = get_fused_info(mat_list, df_fusion_bcol,
                                            tuned_parameters=tuned_implementations_base_param[impl],
                                            implementation_name=impl)
            min_fused = np.array(seperated_list[0])
            for x in seperated_list:
                min_fused = np.minimum(min_fused, np.array(x))
            times[impl] = min_fused
        speedups = {}
        gflops = {}
        target_speed_up = {}

        for impl in impls:
            speedups[impl] = np.array(times[config['baseline']]) / np.array(times[impl])
            target_speed_up[impl] = np.array(np.array(times[impl] / times[config['target']]))
            gflops[impl] = np.array(mat_gflops) / np.array(times[impl])
            runs[impl] = []
        print(insp_time)
        for i, mat in enumerate(mat_list):
            # f = True
            # for x in impls:
            #     if x != config['target'] and target_speed_up[x][i] <= 1.02:
            #         f = False
            #         break
            # if f:
            n_mat_list.append(mat)
            for impl in impls:
                if impl != config['target']:
                    num_runs = int(np.ceil(insp_time[i] / (times[impl][i]-times[config['target']][i])))
                    runs[impl].append(num_runs)
        ids = []
        for imp in impls:
            ids += np.where(np.array(runs[imp]) < -100)
            # runs[imp] = np.array(runs[imp])[ids]
        print(ids)
        k = 0
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['font.family'] = 'serif'
        plt_x = np.arange(len(n_mat_list)-2)
        for impl in impls:
            if impl == config['target']:
                continue
            runs[impl] = runs[impl][:30] + runs[impl][31:41] + runs[impl][42:]
            print(impl, max(runs[impl]))
            # print(runs[impl])
            # print(impl, ":", geo_mean_overflow(target_speed_up[impl]))
            ax.plot(plt_x, runs[impl], color=impl_colors[impl], label=impl_representations[impl], linewidth='1')
        # for impl, bar in bars.items():
        ax.set_xlabel('NNZ', fontsize=15)
        ax.set_title('bCol=cCol='+str(bcol)+',sp', fontsize=15)
        ax.set_xticks([])
        ax.spines[['right', 'top']].set_visible(False)
        # ax.set_xticks(plt_x, mat_representations, rotation='vertical')
        # ax.legend()
        fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1),fontsize=15)
    file_name = log_folder.split('/')[-1] + ".eps"
    plot_path = os.path.join(log_folder, file_name)
    ax.set_ylabel('Number of Runs', fontsize=15)
    plt.show()
    # fig.savefig(plot_path, format='eps')


def plot_gcn_from_logs_folder(logs_folder, config_file, mat_list_file,should_merge="1"):
    print(should_merge)
    with open(mat_list_file) as f:
        mat_list = f.readlines()
    mat_list = [x.strip() for x in mat_list]

    if should_merge == "1":
        merge_logs(logs_folder)
    config = import_config(config_file)
    plot_gcn(logs_folder, "merged.csv", config,mat_list)



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


plot_gcn_from_logs_folder(sys.argv[1], sys.argv[2], sys.argv[3] , sys.argv[4])

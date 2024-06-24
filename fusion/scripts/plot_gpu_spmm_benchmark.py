import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from copy import deepcopy
import yaml

def take_median(df, **kwargs):
    num_trial = df['Number of Trials'].unique()[0]
    time_array = []
    # for each row in dataframe df
    for index, row in df.iterrows():
        for i in range(num_trial):
            t1 = row['Trial' + str(i) + ' Subregion0 Executor']
            time_array.append(t1)
    return np.median(time_array)


def plot_fused_ratio(log_folder, log_file_name, config_file):
    log_file = os.path.join(log_folder, log_file_name)
    df_benchmark = pd.read_csv(log_file)
    df_benchmark = df_benchmark.sort_values(by=['NNZ'])
    #read the config file as yaml into a dictionary
    conf = {}
    bcol = 32
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    impls = [impl['name'] for impl in conf['impls']]
    impl_reprs = {impl['name']: impl['repr'] for impl in conf['impls']}
    bcols = df_benchmark['bCols'].unique()
    fig, axs = plt.subplots(1, 1, figsize=(5, 3))
    # fig.subplots_adjust(bottom=0.1, left=0.06, right=1, top=0.75, wspace=0.1, hspace=0.1)
    times = {}
    mat_list = df_benchmark['Matrix Name'].unique()
    df_benchmark = df_benchmark[df_benchmark['bCols'] == bcol]
    for impl in impls:
        times[impl] = []
    fused_ratio = {impl: [] for impl in impls}
    for mat in mat_list:
        cur_mat = df_benchmark[df_benchmark['Matrix Name'] == mat]
        cur_mat_nnz = cur_mat['NNZ'].unique()[0]
        cur_mat_rows = cur_mat['nRows'].unique()[0]
        for impl in impls:
            cur_impl = cur_mat[cur_mat['Implementation Name'] == impl]
            fused_rows = cur_impl['Number of Fused Rows0'].unique()[0]
            fused_ratio[impl].append(fused_rows / cur_mat_rows)
    # colors = ['red', 'black', 'green', 'goldenrod', 'blue', 'purple', 'pink', 'brown']
    colors = ['red', 'black', 'green', 'goldenrod']
    for impl in impls:
        color = colors.pop()
        axs.plot(mat_list, fused_ratio[impl], label=impl_reprs[impl], color=color, linewidth=1)
    print("----------------------")
    h, l = axs.get_legend_handles_labels()
    fig.legend(loc='upper center', handles=h, labels=l, ncol=5)
    axs.spines[['right', 'top']].set_visible(False)
    axs.set_xticks([])
    plt.show()

def plot_gpu_spmm_benchmark(log_folder, config_file, log_file_name):
    log_file = os.path.join(log_folder, log_file_name)
    df_benchmark = pd.read_csv(log_file)
    df_benchmark = df_benchmark.sort_values(by=['NNZ'])
    #read the config file as yaml into a dictionary
    conf = {}
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    impls = df_benchmark['Implementation Name'].unique()
    impls = [impl['name'] for impl in conf['impls']]
    impl_reprs = {impl['name']: impl['repr'] for impl in conf['impls']}
    mat_list = df_benchmark['Matrix Name'].unique()
    bcols = df_benchmark['bCols'].unique()
    bcols = np.sort(bcols)
    fig, axs = plt.subplots(1, 3, figsize=(16, 2.7))
    fig.subplots_adjust(bottom=0.1, left=0.06, right=1, top=0.75, wspace=0.1, hspace=0.1)
    for i,bcol in enumerate(bcols):
        times = {}
        df_benchmark_bcol = df_benchmark[df_benchmark['bCols'] == bcol]
        mat_list = df_benchmark_bcol['Matrix Name'].unique()
        for impl in impls:
            times[impl] = []
        mat_gflops = []
        for mat in mat_list:
            cur_mat = df_benchmark_bcol[df_benchmark_bcol['Matrix Name'] == mat]
            cur_mat_nnz = cur_mat['NNZ'].unique()[0]
            mat_gflops.append((cur_mat_nnz * bcol + cur_mat_nnz * bcol) / 1e9)
            for impl in impls:
                cur_impl = cur_mat[cur_mat['Implementation Name'] == impl]
                cur_impl = cur_impl[cur_impl['bCols'] == bcol]
                times[impl].append(take_median(cur_impl))
        gflops = {}
        # colors = ['red', 'black', 'green', 'goldenrod', 'blue', 'purple', 'pink', 'brown']
        colors = ['red', 'black', 'green', 'goldenrod']
        print(bcol)
        for impl in impls:
            gflops[impl] = np.array(mat_gflops) / np.array(times[impl])
            print(impl, np.mean(gflops[impl]))
            color = colors.pop()
            axs[i].scatter(mat_list, gflops[impl], label=impl_reprs[impl], s=5,color=color)
        print("----------------------")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(loc='upper center', handles=h, labels=l, ncol=5)
        axs[i].spines[['right', 'top']].set_visible(False)
        axs[i].set_xticks([])
        axs[i].set_title('bCol='+str(bcol))
    plt.show()

def plot_gpu_spmm_speedups_vs_cusparse(log_folder, log_file_name):
    log_file = os.path.join(log_folder, log_file_name)
    df_benchmark = pd.read_csv(log_file)
    df_benchmark = df_benchmark.sort_values(by=['NNZ'])
    mat_list = df_benchmark['Matrix Name'].unique()
    bcols = df_benchmark['bCols'].unique()
    bcols = np.sort(bcols)
    impls = df_benchmark['Implementation Name'].unique()
    baseline_impls = ['GPU_cuSparse_SpMM_CSR_ALG2_Demo', 'GPU_cuSparse_SpMM_CSR_ALG3_Demo', 'GPU_cuSparse_SpMM_CSR_Default_Demo'] #TODO: Fix this
    fig, axs = plt.subplots(7, 3, figsize=(16, 12))
    fig.subplots_adjust(bottom=0.03, left=0.06, right=1, top=0.92, wspace=0.2, hspace=0.2)
    baseline_impl = baseline_impls[0]
    impls = list(impls)
    target_impls = deepcopy(impls)
    target_impls.remove('CPU_SpMM_Demo')
    for bi in baseline_impls:
        target_impls.remove(bi)

    for i,bcol in enumerate(bcols):
        times = {}
        df_benchmark_bcol = df_benchmark[df_benchmark['bCols'] == bcol]
        mat_list = df_benchmark_bcol['Matrix Name'].unique()
        for impl in impls:
            times[impl] = []
        mat_gflops = []
        for mat in mat_list:
            cur_mat = df_benchmark_bcol[df_benchmark_bcol['Matrix Name'] == mat]
            cur_mat_nnz = cur_mat['NNZ'].unique()[0]
            mat_gflops.append((cur_mat_nnz * bcol + cur_mat_nnz * bcol) / 1e9)
            for impl in impls:
                cur_impl = cur_mat[cur_mat['Implementation Name'] == impl]
                cur_impl = cur_impl[cur_impl['bCols'] == bcol]
                times[impl].append(take_median(cur_impl))
        gflops = {}
        for impl in impls:
            gflops[impl] = np.array(mat_gflops) / np.array(times[impl])

        for bi in baseline_impls:
            gflops[bi] = np.array(mat_gflops) / np.array(times[bi])
        speed_ups = {}
        for ii, impl in enumerate(target_impls):
            speed_ups[impl] = np.array(times[baseline_impl]) / np.array(times[impl])
            ax = axs[ii, i]
            impl_repr = impl.split("_")[1]
            if impl_repr == "cuSparse":
                impl_repr += "_" + impl.split("_")[-2]
            # ax.scatter(gflops[baseline_impl], speed_ups[impl], label=impl_repr, s=5)
            ax.scatter(mat_list, speed_ups[impl], label=impl_repr, s=5)
            ax.axhline(y=1, color='black', linestyle='--')
            ax.set_xticks([])
            ax.spines[['right', 'top']].set_visible(False)
            if i == 1:
                ax.set_title(impl_repr)
            else:
                ax.set_title('bCol='+str(bcol))

    fig.suptitle('speed-up over ' + baseline_impl)
    plt.show()
    # plt.show()


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
            print(entry_names[i])
            df = pd.concat([pd.read_csv(os.path.join(logs_folder, entry_names[i])), df], ignore_index=True)
        df.to_csv(os.path.join(logs_folder, "merged.csv"), index=False)


def plot_gcn_from_logs_folder(logs_folder, config_file, should_merge="1"):
    # print(should_merge)
    plt.rcParams["font.family"] = "serif"
    if should_merge == "1":
        merge_logs(logs_folder)
    # config = import_config(config_file)
    plot_fused_ratio(logs_folder, "merged.csv", config_file)
    # plot_gpu_spmm_benchmark(logs_folder, config_file, "merged.csv")
    # plot_gpu_spmm_speedups_vs_cusparse(logs_folder, "merged.csv")

plot_gcn_from_logs_folder(sys.argv[1], sys.argv[2])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yaml
from matplotlib.pyplot import cm
import scipy.stats
import random
from matplotlib.lines import Line2D
import statistics


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
    if params is None:
        params = [[x] for x in df[tuned_parameters[0]].unique() if x != -1]
        for tuned_parameter in tuned_parameters[1:]:
            # params = [40, 400, 4000, 8000, 10000]
            # params = [4, 8, 40, 100, 1000]
            params = [x + [y] for x in params for y in df[tuned_parameter].unique() if y != -1]

        # params = [10, 20, 50, 100, 200]
    seperated_list = [[] for i in range(len(params))]
    for matr in matr_list:
        cur_matr = df[df['MatrixName'] == matr]
        for i in range(len(params)):
            fused = cur_matr[cur_matr['Implementation Name'] == implementation_name]
            try:
                for j in range(len(tuned_parameters)):
                    fused = fused[fused[tuned_parameters[j]] == params[i][j]]
                # if not fused['Tile Size Mean0'].isna().any() and not fused.shape[0] == 0:
                if not fused.shape[0] == 0:
                    seperated_list[i].append(take_median(fused))
                else:
                    seperated_list[i].append(float('inf'))
            except IndexError as e:
                print("Error for Tuned implementation: ", implementation_name, " in matrix: ", matr, " with params: ",
                      params[i][j])
    return seperated_list


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
    # bcols = config['feature_sizes']
    bcols = [32,64,128]
    # tuned_implementations = [impl['name'] for impl in config['implementations'] if impl['tuned']]
    # tuned_implementations_base_param = {impl['name']: impl['tune_parameters'] for impl in config['implementations'] if
    #                                     impl['tuned']}
    # sort df_fusion based on 'NNZ'
    parameter = 'MTile'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    # mat_list = df_fusion['MatrixName'].unique()
    # mat_list = config['matrices']
    mat_list = df_fusion['MatrixName'].unique()
    # edims = config['embed_dimensions']
    impls = list(map(lambda i: i['name'], config['implementations']))
    new_impls = []
    tile_sizes = np.sort(df_fusion[parameter].unique())
    tile_sizes = np.delete(tile_sizes, 0)
    tile_sizes = np.delete(tile_sizes, 0)
    for impl in impls:
        for ts in tile_sizes:
            new_impls.append(impl + '_' + str(ts))
    plt_x = np.arange(len(mat_list))
    bcol = df_fusion['bCols'].unique()[0]
    colors = ['maroon', 'brown', 'purple', 'yellow', 'orange', 'black', 'grey', 'r', 'g', 'b', 'c', 'm', 'y', 'k']
    # for edim in edims:
    df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
    # df_fusion_bcol_edim = df_fusion_bcol[df_fusion_bcol['EmbedDim'] == edim]
    times = {}
    for mat in mat_list:
        cur_mat = df_fusion_bcol[df_fusion_bcol['MatrixName'] == mat]
        for ts in tile_sizes:
            cur_mat_ts = cur_mat[cur_mat[parameter] == ts]
            for x in impls:
                try:
                    impl_ts = x + '_' + str(ts)
                    if impl_ts in times:
                        times[impl_ts].append(take_median(cur_mat_ts[cur_mat_ts['Implementation Name'] == x]))
                    else:
                        times[impl_ts] = [take_median(cur_mat_ts[cur_mat_ts['Implementation Name'] == x])]
                except IndexError as e:
                    print("Error for Implementation: ", x, " in matrix: ", mat, " with tile size: ", ts)
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(bottom=0.4, left=0.06, right=1, top=0.87, wspace=0.1, hspace=0.1)
    speedups = {}
    for x in new_impls:
        speedups[x] = np.array(times[new_impls[0]]) / np.array(times[x])
        print(x, ":", geo_mean_overflow(speedups[x]))
    ax.set_xlabel('Cache Size', fontweight='bold', fontsize=15)
    for x in new_impls:
        # ax.scatter(plt_x, times[x], label=x)
        ax.plot(plt_x, speedups[x], label=x, color=colors.pop(0))
    # plt.xticks(plt_x, mat_list, rotation='vertical')
    # plt.xticks([])
    mat_representations = [mat[:-4] for mat in mat_list]
    ax.set_xticks(plt_x, mat_representations, rotation='vertical')
    plot_path = os.path.join(log_folder,str(bcol) + '_' + '.png')
    ax.legend()
    fig.savefig(plot_path)


def plot_performance_vs_fused_ratio(log_folder, log_file_name, config):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
    # bcols = config['feature_sizes']
    bcols = df_fusion['bCols'].unique()
    tuned_implementations = [impl['name'] for impl in config['implementations'] if impl['tuned']]
    tuned_implementations_base_param = {impl['name']: impl['tune_parameters'] for impl in config['implementations'] if
                                        impl['tuned']}
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    mat_list = list(df_fusion['Matrix Name'].unique())
    # mat_list.remove('kmnist_norm_10NN.mtx')
    # random.shuffle(mat_list)
    # mat_list = mat_list[:10]
    # mat_list = mat_list[:5]
    # mat_list = config['matrices']
    impls = list(map(lambda i: i['name'], config['implementations']))
    # colors = ['maroon', 'brown', 'purple', 'yellow', 'orange', 'black', 'grey', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'cyan', ]
    colors = list(cm.rainbow(np.linspace(0, 1, len(mat_list) * 3)))

    print(len(colors))
    for bcol in bcols:
        df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
        df_fusion_bcol_edim = df_fusion_bcol[df_fusion_bcol['EmbedDim'] == bcol]
        # plt.rcParams.update(plt.rcParamsDefault)
        print(len(mat_list))

        for mat in mat_list:
            fig, ax = plt.subplots(figsize=(15, 8))
            cur_mat = df_fusion_bcol_edim[df_fusion_bcol_edim['MatrixName'] == mat]
            cur_fused = cur_mat[cur_mat['Implementation Name'] == 'GCN_SingleLayerFusedSeperated']
            fused_iterations = cur_fused['FusedIterations0'].unique()
            fused_iterations = np.sort(fused_iterations)
            # print(fused_iterations)
            # print(cur_mat.iloc[0]['nRows'])
            fused_ratios = np.array([x / cur_mat.iloc[0]['nRows'] for x in fused_iterations])
            fused_times = []
            for x in fused_iterations:
                cur_run = cur_fused[cur_fused['FusedIterations0'] == x]
                try:
                    fused_times.append(take_median(cur_run))
                except IndexError as e:
                    continue
            mkl_row = cur_mat[cur_mat['Implementation Name'] == 'GCN_SingleLayerMKL']
            # print(mkl_row)
            try:
                mkl_time = take_median(mkl_row)
            except IndexError as e:
                continue
            fused_speedups = [mkl_time / x for x in fused_times]
            # print(colors)
            random.shuffle(colors)
            color = colors.pop()
            ax.scatter(fused_ratios, fused_speedups, color=color, edgecolor='grey', label=mat)
            # co = np.corrcoef(fused_ratios, fused_speedups)
            print(mat)
            slope, intercept, r, p, stderr = scipy.stats.linregress(fused_ratios, fused_speedups)
            line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
            ax.legend()
            ax.plot(fused_ratios, intercept + slope * fused_ratios, color=color)

            # ax.plot(fused, fused_speedups, color=color)

            plot_path = os.path.join(log_folder, mat.split(".")[0] + "_" + str(bcol) + '.pdf')
            ax.set_xlabel('fused_ratios', fontweight='bold', fontsize=15)
            ax.set_ylabel('speed_up', fontweight='bold', fontsize=15)
            # ax.set_xticks(plt_x,fused_ratios, rotation='vertical')
            fig.subplots_adjust(bottom=0.25)
            fig.savefig(plot_path)


def plot_stack_bar(log_folder, log_file_name, config):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
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
    mat_list = df_fusion['MatrixName'].unique()
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.3))
    fig.subplots_adjust(bottom=0.1, left=0.1, right=1, top=0.75, wspace=0.1, hspace=0.1)
    file_name = log_file.split('/')[-1] + ".eps"
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    impl_colors = {impl['name']: impl['color'] for impl in config['implementations']}
    bcol = 64
    mat_gflops = []
    nnz_list = []
    df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
    df_fusion_bcol = df_fusion_bcol.sort_values(by=['NNZ'])
    print(len(mat_list))
    times = {}
    bars = {}
    times['Coarse Grained Tile'] = []
    plt_x = np.arange(len(mat_list))
    for mat in mat_list:
        # print(mat)
        cur_mat = df_fusion_bcol[df_fusion_bcol['MatrixName'] == mat]
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
        cur_impl = cur_mat[cur_mat['Implementation Name'] == config['target']]
        cur_impl_mtile = cur_impl[cur_impl['Iter Per Partition'] == 2048]
        try:
            times['Coarse Grained Tile'].append(take_median(cur_impl_mtile))
        except IndexError as e:
            print("Error for Not Tuned Implementation: Coarse Grained Tile", " in matrix: ", mat)
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
    for impl in times.keys():
        if remove_last_mat:
            times[impl] = times[impl][:len(plt_x)]
        speedups[impl] = np.array(times[config['baseline']]) / np.array(times[impl])
        target_speed_up[impl] = np.array(np.array(times[impl] / times[config['target']]))
        gflops[impl] = np.array(mat_gflops) / np.array(times[impl])
    k = 0
    print(geo_mean_overflow(speedups['Coarse Grained Tile']))
    check = gflops['Coarse Grained Tile'] > 0
    gflops['Coarse Grained Tile'] = gflops['Coarse Grained Tile'][check]
    gflops['GCN_SingleLayer_UnFused'] = gflops['GCN_SingleLayer_UnFused_sequential'][check]
    gflops['GCN_SingleLayer_FusedSeperated'] = gflops['GCN_SingleLayer_FusedSeperated'][check]
    plt_x = np.arange(len(gflops['Coarse Grained Tile']))
    su1 = gflops['Coarse Grained Tile'] - gflops['GCN_SingleLayer_UnFused']
    check = su1 < 0
    su1[check] = 0
    su2 = gflops['GCN_SingleLayer_FusedSeperated'] - (su1 + gflops['GCN_SingleLayer_UnFused'])
    check = su2 < 0
    su2[check] = 0
    print(len(np.where(su2 > 0)[0]))
    part_speedup = {
        'UnFused Sequential': gflops['GCN_SingleLayer_UnFused'],
        'Tile Fusion - After Step 1': su1,
        'Tile Fusion - After Step 2': su2,
    }
    impl_colors = {
        'UnFused Sequential': 'lightblue',
        'Tile Fusion - After Step 1': 'orange',
        'Tile Fusion - After Step 2': 'maroon',
    }
    # plt.rcParams.update(plt.rcParamsDefault)
    bottom = np.zeros(len(plt_x))
    for impl in part_speedup.keys():
        color = impl_colors[impl]
        # print(impl, ":", geo_mean_overflow(part_speedup[impl]))
        ax.bar(plt_x, part_speedup[impl], color=color, label=impl, width=0.7, bottom=bottom)
        bottom += part_speedup[impl]
    # for impl in impls:
    #     print(impl, ":", geo_mean_overflow(target_speed_up[impl]))
    #     ax.plot(plt_x, gflops[impl], color=impl_colors[impl], label=impl_representations[impl], linewidth='0.6')
    ax.set_xlabel('NNZ', fontsize=15)
    ax.set_ylabel('GFLOP/s', fontsize=15)
    # ax.set_title('bCol=cCol=' + str(bcol) + ',sp', fontsize=15)
    ax.set_xticks([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.15), ncol=3, fontsize=15,frameon=False,labelspacing=0.1, columnspacing=0.1)
    ax.spines[['right', 'top']].set_visible(False)
    file_name = log_folder.split('/')[-1] + '_stacked' + ".eps"
    plot_path = os.path.join(log_folder, file_name)
    # ax.set_ylabel('Normalized Speed-up', fontweight='bold', fontsize=7.5)
    plt.show()
    fig.savefig(plot_path, format='eps')


def plot_gflops_per_bcol(log_folder, log_file_name, config, ex_mat_list):
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
    df_fusion_sorted = df_fusion.copy()
    densities = np.array(df_fusion_sorted['NNZ'] / (df_fusion_sorted['nRows'] ** 2))
    densities.sort()
    df_fusion_sorted.sort_values(by=['NNZ'], key=lambda nnz: nnz / (df_fusion_sorted['nRows'] ** 2), inplace=True)
    mat_list = list(df_fusion['Matrix Name'].unique())
    mat_list = [mat for mat in mat_list if mat in ex_mat_list]
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    fig.tight_layout()
    file_name = log_file.split('/')[-1] + ".eps"
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    impl_colors = {impl['name']: impl['color'] for impl in config['implementations']}
    gflops_bcol = {}
    for impl in impls:
        gflops_bcol[impl] = []
    for i, bcol in enumerate(bcols):
        print(bcol)
        bcol = int(bcol)
        mat_gflops = []
        nnz_list = []
        df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
        df_fusion_bcol = df_fusion_bcol.sort_values(by=['NNZ'])
        print(len(mat_list))
        times = {}
        bars = {}
        plt_x = np.arange(len(mat_list))
        for mat in mat_list:
            # print(mat)
            cur_mat = df_fusion_bcol[df_fusion_bcol['MatrixName'] == mat]
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
            gflops_bcol[impl].append(geo_mean_overflow(gflops[impl]))
    for impl in impls:
        print(impl, ":", gflops_bcol[impl])
    # plt.rcParams.update(plt.rcParamsDefault)
    plt_x = np.arange(len(bcols))
    for impl in impls:
        # ax.scatter(plt_x, gflops[impl], color='white', edgecolor=impl_colors[impl], label=impl_representations[impl],
        #           marker=markers.pop(0), s=10)
        # print(impl, ":", geo_mean_overflow(target_speed_up[impl]))
        ax.plot(plt_x, gflops_bcol[impl], color=impl_colors[impl], label=impl_representations[impl], linewidth='0.6')
    # for impl, bar in bars.items():
    ax.set_xlabel('Dense Columns', fontweight='bold', fontsize=7.5)
    ax.set_ylabel('mean GFLOPs per second', fontweight='bold', fontsize=7.5)
    # ax.set_title(str(bcol), fontsize=7)
    ax.set_xticks(plt_x, bcols)
    ax.spines[['right', 'top']].set_visible(False)
    file_name = 'gflops_per_bcol.eps'
    plot_path = os.path.join(log_folder, file_name)
    # h, l = axs[0].get_legend_handles_labels()
    # fig.legend(h, l, loc='lower center', ncol=3)
    # fig.suptitle('GeMM-SpMM for ss-graphs on Intel Skylake', fontsize=9)
    # plt.show()
    # fig.savefig(plot_path, format='eps')


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
    # mat_list.remove('va2010.mtx')
    # mat_list = config['matrices']
    impls = list(map(lambda i: i['name'], config['implementations']))
    bar_width = 0.2
    df_fusion_sorted = df_fusion.copy()
    densities = np.array(df_fusion_sorted['NNZ'] / (df_fusion_sorted['nRows'] ** 2))
    densities.sort()
    df_fusion_sorted.sort_values(by=['NNZ'], key=lambda nnz: nnz / (df_fusion_sorted['nRows'] ** 2), inplace=True)
    mat_list = list(df_fusion['Matrix Name'].unique())
    # mat_list = [mat for mat in mat_list if mat in ex_mat_list]
    # mat_list = list(df_fusion_sorted['MatrixName'].unique())
    # mat_list.remove("Queen_4147.mtx")
    # mat_list.remove("Hook_1498.mtx")
    fig, axs = plt.subplots(1, 3, figsize=(16, 2.7))
    # fig, axs = plt.subplots(1, 3, figsize=(16, 2))
    # fig.subplots_adjust(bottom=0.1, left=0.06, right=1, top=0.72, wspace=0.1, hspace=0.1)
    fig.subplots_adjust(bottom=0.4, left=0.06, right=1, top=0.87, wspace=0.1, hspace=0.1)
    # fig.subplots_adjust(bottom=0.1, left=0.05, right=1, top=0.9, wspace=0.1, hspace=0.1)
    file_name = log_file.split('/')[-1] + ".eps"
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    impl_colors = {impl['name']: impl['color'] for impl in config['implementations']}
    # mat_list.remove('Queen_4147.mtx')
    # mat_list.remove('Hook_1498.mtx')
    for i, bcol in enumerate(bcols):
        print(bcol)
        bcol = int(bcol)
        mat_gflops = []
        nnz_list = []
        # for edim in edims:
        df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
        df_fusion_bcol = df_fusion_bcol.sort_values(by=['NNZ'])
        # mat_list_ext = list(df_fusion_bcol['MatrixName'])
        # unique, counts = np.unique(mat_list_ext, return_counts=True)
        # print(unique)
        # df_fusion_sorted_bcol = df_fusion_sorted[df_fusion_sorted['bCols'] == bcol]
        # mat_list = list(df_fusion['Matrix Name'].unique())
        # if bcol == 128:
        #     mat_list.remove("kron_g500-logn18.mtx")
        #     mat_list.remove("Emilia_923.mtx")
        #     mat_list.remove("Geo_1438.mtx")
        #     mat_list.remove("audikw_1.mtx")
        #     mat_list.remove("inline_1.mtx")
        #     mat_list.remove("Queen_4147.mtx")
        # mat_list.remove("Queen_4147.mtx")
        # mat_list.remove("Hook_1498.mtx")
        # print(len(mat_list))
        # mat_list = list(df_fusion_bcol['Matrix Name'].unique())
        # df_fusion_bcol_edim = df_fusion_bcol[df_fusion_bcol['EmbedDim'] == edim]
        times = {}
        bars = {}
        plt_x = np.arange(len(mat_list))
        for mat in mat_list:
            # print(mat)
            cur_mat = df_fusion_bcol[df_fusion_bcol['MatrixName'] == mat]
            # print(bcol, mat)
            cur_mat_nnz = cur_mat['NNZ'].unique()[0]
            rows = cur_mat['nRows'].unique()[0]
            nnz_list.append(cur_mat_nnz)
            # mat_gflops.append((cur_mat_nnz * bcol + rows * bcol * bcol) / 1e9)
            mat_gflops.append((cur_mat_nnz * bcol + cur_mat_nnz * bcol) / 1e9)
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
            target_speed_up[impl] = np.array(times[impl]) / np.array(times[config['target']])
            gflops[impl] = np.array(mat_gflops) / np.array(times[impl])
        # print(max(target_speed_up['SpMM_SpMM_UnFusedParallelAvx512']))
        # print(min(target_speed_up['SpMM_SpMM_UnFusedParallelAvx512']))
        # print(statistics.median(target_speed_up['SpMM_SpMM_UnFusedParallelAvx512']))
        # print('mkl',len(np.where(speedups[config['target']] > 1)[0]))
        # print('unfused',len(np.where(target_speed_up['SpMM_SpMM_UnFusedParallelAvx512'] > 1)[0]))
        # print(len(mat_list))
        colors = colors = ['deepskyblue', 'violet', 'mediumseagreen', 'blueviolet', 'black', 'goldenrod']
        markers = ['o', '<', 'v', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '^', '>', '1', '2', '3', '4', 'x',
                   '+']
        # if bcol == 32:
        #     data = {
        #         'unfused_gflops': gflops['GCN_SingleLayer_UnFused'],
        #         'fused_gflops': gflops['GCN_SingleLayer_FusedSeperated'],
        #         'matrices': mat_list,
        #     }
        #     pd.DataFrame(data).to_csv('fused_unfused_gflops.csv', index=False)

        k = 0
        # plt.rcParams.update(plt.rcParamsDefault)
        ax = axs[i]
        # new_gflops = {}
        # new_speed_ups = {}
        # for impl in impls:
        #     new_gflops[impl] = [gflops[impl][p] for p,x in enumerate(speedups[config['target']]) if x > 0.7]
        #     new_speed_ups[impl] = [speedups[impl][p] for p,x in enumerate(speedups[config['target']]) if x > 0.7]
        # gflops = new_gflops
        # speedups = new_speed_ups
        # plt_x = plt_x[:len(gflops[config['target']])]
        for impl in impls:
            # ax.plot(plt_x, gflops[impl], color=impl_colors[impl],
            #            label=impl_representations[impl], linewidth=1)
            print(impl, ":", geo_mean_overflow(speedups[impl]))
        # if impl in ['SpMM_SpMM_Demo_FusedCSCAtomic','SpMM_SpMM_FusedParallel_Redundant']:
        #     gflops[impl] = 4*gflops[impl]
        #     target_speed_up[impl] = target_speed_up[impl]/4
        # print(speedups[config['target']])
        # speedups[config['target']] = np.where(speedups[config['target']] < 1, 1, speedups[config['target']])
        # print(speedups[config['target']])
        # print("gmean", ":", geo_mean_overflow(speedups[config['target']]))
        # print("arithmatic mean", ":", np.mean(speedups[config['target']]))
        # if impl == config['target']:
        #     print(len(np.where(speedups[impl] > 1.15)[0])/len(speedups[impl]))
        #     print(len(np.where(speedups[impl] > 1.3)[0])/len(speedups[impl]))
        #     print(len(np.where(speedups[impl] > 1)[0])/len(speedups[impl]))
        # print("arithmetic: ", impl, ":", np.mean(target_speed_up[impl]))
        # print(impl, min(gflops[impl]), max(gflops[impl]))
        # print(impl, min(speedups[impl]), max(speedups[impl]))
        ax.scatter(gflops[config['baseline']], speedups[config['target']], color=impl_colors[config['target']],
               label=impl_representations[impl], s=5)
        # set limit of y axis to 0-3
        # ax.set_ylim(0, 3)
        # draw horizental line at 1
        # ax.axhline(y=1, color=impl_colors[config['baseline']], linestyle='--')
        # for impl, bar in bars.items():
        ax.set_xlabel('NNZ', fontsize=15)
        ax.set_title('bCol=cCol=' + str(bcol) + ',sp', fontsize=15)
        # set the y scale to 0 to 600
        # ax.set_ylim(0, 700)
        # ax.set_xticks([])
        ax.spines[['right', 'top']].set_visible(False)
        mat_representations = [mat[:-4] for mat in mat_list]
        # ax.set_xticks(plt_x, mat_representations, rotation='vertical')
        ax.set_xticks([])
        # ax.legend()
    file_name = log_folder.split('/')[-1] + ".eps"
    plot_path = os.path.join(log_folder, file_name)

    axs[0].set_ylabel('GFlops', fontsize=15)
    h, l = axs[0].get_legend_handles_labels()
    # line = Line2D([0], [0], label='Unfused Baseline', color='deepskyblue', linewidth=1)
    # h.extend([line])
    fig.legend(handles=h, loc='lower center', ncol=3,fontsize=15)
    # fig.suptitle('GeMM-SpMM for ss-graphs on Intel Skylake', fontsize=9)
    plt.show()
    # plt.savefig(plot_path, format='eps')


def plot_gcn_best_threads(log_folder, log_file_name, config):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
    # bcols = config['feature_sizes']
    bcols = df_fusion['bCols'].unique()
    print(bcols)
    threads = df_fusion['Number of Threads'].unique()
    bcols = sorted(bcols, key=lambda f: int(f))
    tuned_implementations = [impl['name'] for impl in config['implementations'] if impl['tuned']]
    tuned_implementations_base_param = {impl['name']: impl['tune_parameters'] for impl in config['implementations'] if
                                        impl['tuned']}
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    # mat_list.remove('va2010.mtx')
    # mat_list = config['matrices']
    impls = list(map(lambda i: i['name'], config['implementations']))
    bar_width = 0.2
    df_fusion_sorted = df_fusion.copy()
    densities = np.array(df_fusion_sorted['NNZ'] / (df_fusion_sorted['nRows'] ** 2))
    densities.sort()
    df_fusion_sorted.sort_values(by=['NNZ'], key=lambda nnz: nnz / (df_fusion_sorted['nRows'] ** 2), inplace=True)
    mat_list = list(df_fusion['Matrix Name'].unique())
    fig, axs = plt.subplots(1, 3, figsize=(16, 2.7))
    # fig, axs = plt.subplots(1, 3, figsize=(16, 2))
    # fig.subplots_adjust(bottom=0.1, left=0.06, right=1, top=0.72, wspace=0.1, hspace=0.1)
    fig.subplots_adjust(bottom=0.4, left=0.06, right=1, top=0.87, wspace=0.1, hspace=0.1)
    # fig.subplots_adjust(bottom=0.1, left=0.05, right=1, top=0.9, wspace=0.1, hspace=0.1)
    file_name = log_file.split('/')[-1] + ".eps"
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    impl_colors = {impl['name']: impl['color'] for impl in config['implementations']}
    # mat_list.remove('Queen_4147.mtx')
    # mat_list.remove('Hook_1498.mtx')
    for i, bcol in enumerate(bcols):
        print(bcol)
        bcol = int(bcol)
        # for edim in edims:
        df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
        df_fusion_bcol = df_fusion_bcol.sort_values(by=['NNZ'])
        best_gflops = {}
        plt_x = np.arange(len(mat_list))
        for thread in threads:
            mat_gflops = []
            nnz_list = []
            times = {}
            bars = {}
            df_fusion_threads = df_fusion_bcol[df_fusion_bcol['Number of Threads'] == thread]
            for mat in mat_list:
                # print(mat)
                cur_mat = df_fusion_threads[df_fusion_threads['MatrixName'] == mat]
                # print(bcol, mat)
                cur_mat_nnz = cur_mat['NNZ'].unique()[0]
                rows = cur_mat['nRows'].unique()[0]
                nnz_list.append(cur_mat_nnz)
                mat_gflops.append((cur_mat_nnz * bcol + rows * bcol * bcol) / 1e9)
                # mat_gflops.append((cur_mat_nnz * bcol + cur_mat_nnz * bcol) / 1e9)
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
                seperated_list = get_fused_info(mat_list, df_fusion_threads,
                                                tuned_parameters=tuned_implementations_base_param[impl],
                                                implementation_name=impl)
                min_fused = np.array(seperated_list[0])
                for x in seperated_list:
                    min_fused = np.minimum(min_fused, np.array(x))
                times[impl] = min_fused
            speedups = {}
            gflops = {}
            remove_last_mat = False
            target_speed_up = {}
            for impl in impls:
                gflops[impl] = np.array(mat_gflops) / np.array(times[impl])
                if impl == config['baseline']:
                    if thread == 80:
                        best_gflops[impl] = gflops[impl]
                elif impl not in best_gflops:
                    best_gflops[impl] = gflops[impl]
                else:
                    best_gflops[impl] = np.maximum(best_gflops[impl], gflops[impl])

        speedups = {}
        k = 0
        ax = axs[i]
        for impl in impls:
            speedups[impl] = np.array(best_gflops[impl]) / np.array(best_gflops[config['baseline']])
            ax.plot(plt_x, best_gflops[impl], color=impl_colors[impl],
                    label=impl_representations[impl], linewidth=1)
            print(impl, ":", geo_mean_overflow(speedups[impl]))
        ax.set_xlabel('NNZ', fontsize=15)
        ax.set_title('bCol=cCol=' + str(bcol) + ',sp', fontsize=15)
        # set the y scale to 0 to 600
        # ax.set_ylim(0, 700)
        # ax.set_xticks([])
        ax.spines[['right', 'top']].set_visible(False)
        mat_representations = [mat[:-4] for mat in mat_list]
        # ax.set_xticks(plt_x, mat_representations, rotation='vertical')
        ax.set_xticks([])
        # ax.legend()
    file_name = log_folder.split('/')[-1] + ".eps"
    plot_path = os.path.join(log_folder, file_name)

    axs[0].set_ylabel('GFlops', fontsize=15)
    h, l = axs[0].get_legend_handles_labels()
    # line = Line2D([0], [0], label='Unfused Baseline', color='deepskyblue', linewidth=1)
    # h.extend([line])
    fig.legend(handles=h, loc='lower center', ncol=3,fontsize=15)
    # fig.suptitle('GeMM-SpMM for ss-graphs on Intel Skylake', fontsize=9)
    plt.show()
    # plt.savefig(plot_path, format='eps')

def plot_gcn_reordered_graphs(log_folder, log_file_name, config, ex_mat_list):
    log_file = os.path.join(log_folder, log_file_name)
    df_fusion = pd.read_csv(log_file)
    # bcols = config['feature_sizes']
    bcols = df_fusion['bCols'].unique()
    print(bcols)
    bcols = sorted(bcols, key=lambda f: int(f))
    tuned_implementations = [impl['name'] for impl in config['implementations'] if impl['tuned']]
    tuned_implementations_base_param = {impl['name']: impl['tune_parameters'] for impl in config['implementations'] if
                                        impl['tuned']}
    ordered_impls = [impl for impl in config['ordered_impls']]
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    # mat_list.remove('va2010.mtx')
    # mat_list = config['matrices']
    impls = list(map(lambda i: i['name'], config['implementations']))
    bar_width = 0.2
    df_fusion_sorted = df_fusion.copy()
    densities = np.array(df_fusion_sorted['NNZ'] / (df_fusion_sorted['nRows'] ** 2))
    densities.sort()
    df_fusion_sorted.sort_values(by=['NNZ'], key=lambda nnz: nnz / (df_fusion_sorted['nRows'] ** 2), inplace=True)
    mat_list = list(df_fusion['Matrix Name'].unique())
    # mat_list = [mat for mat in mat_list if mat in ex_mat_list]
    # mat_list = list(df_fusion_sorted['MatrixName'].unique())
    # mat_list.remove("coauthor_cs.mtx")
    # mat_list.remove("ogbn_products_ordered.mtx")
    fig, axs = plt.subplots(1, 3, figsize=(16, 2.7),sharey='row')
    # fig, axs = plt.subplots(1, 3, figsize=(16, 2))
    # fig.subplots_adjust(bottom=0.1, left=0.06, right=1, top=0.72, wspace=0.1, hspace=0.1)
    fig.subplots_adjust(bottom=0.3, left=0.06, right=0.95, top=0.75, wspace=0.05, hspace=0.05)
    # fig.subplots_adjust(bottom=0.1, left=0.05, right=1, top=0.9, wspace=0.1, hspace=0.1)
    file_name = log_file.split('/')[-1] + ".eps"
    impl_representations = {impl['name']: impl['representation'] for impl in config['implementations']}
    impl_colors = {impl['name']: impl['color'] for impl in config['implementations']}
    # mat_list.remove('Queen_4147.mtx')
    # mat_list.remove('Hook_1498.mtx')
    for i, bcol in enumerate(bcols):
        print(bcol)
        bcol = int(bcol)
        mat_gflops = []
        nnz_list = []
        # for edim in edims:
        df_fusion_bcol = df_fusion[df_fusion['bCols'] == bcol]
        df_fusion_bcol = df_fusion_bcol.sort_values(by=['NNZ'])
        times = {}
        bars = {}
        plt_x = np.arange(len(mat_list))
        o_counter = 0
        for mat in mat_list:
            if mat.endswith('ordered.mtx'):
                o_counter += 1
            # print(mat)
            cur_mat = df_fusion_bcol[df_fusion_bcol['MatrixName'] == mat]
            # print(bcol, mat)
            cur_mat_nnz = cur_mat['NNZ'].unique()[0]
            rows = cur_mat['nRows'].unique()[0]
            nnz_list.append(cur_mat_nnz)
            # mat_gflops.append((cur_mat_nnz * bcol + rows * bcol * bcol) / 1e9)
            mat_gflops.append((cur_mat_nnz * bcol + cur_mat_nnz * bcol) / 1e9)
            for x in impls:
                if x not in tuned_implementations:
                    try:
                        if x in times:
                            times[x].append(take_median(cur_mat[cur_mat['Implementation Name'] == x]))
                        else:
                            times[x] = [take_median(cur_mat[cur_mat['Implementation Name'] == x])]
                    except IndexError as e:
                        print("Error for Not Tuned Implementation: ", x, " in matrix: ", mat)
        print(o_counter)
        print(len(mat_list))
        for x in impls:
            if x not in tuned_implementations:
                times[x] = np.array(times[x])
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
            # speedups[impl] = np.array(times[config['baseline']]) / np.array(times[impl])
            gflops[impl] = np.array(mat_gflops) / np.array(times[impl])
        colors = ['deepskyblue', 'violet', 'mediumseagreen', 'blueviolet', 'black', 'goldenrod', 'violet', 'mediumseagreen', 'blueviolet']
        markers = ['o', '<', 'v', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '^', '>', '1', '2', '3', '4', 'x',
                   '+']

        k = 0
        # plt.rcParams.update(plt.rcParamsDefault)
        ax = axs[i]
        new_mat_list = []
        new_gflops = {}
        new_impls = []
        new_speedups = {}
        for impl in impls:
            new_impls.append(impl)
            impl_ordered = impl + '_ordered'
            if impl in ordered_impls:
                new_impls.append(impl_ordered)
                new_gflops[impl_ordered] = []
                new_speedups[impl_ordered] = []
                impl_representations[impl_ordered] = impl_representations[impl] + ' (RCM Ordered)'
                impl_colors[impl_ordered] = colors.pop()
            new_gflops[impl] = []
            new_speedups[impl] = []
            for m, mat in enumerate(mat_list):
                mat_type = mat.split('_')[-1].split('.')[0]
                mat_name = mat.split('_')[0] + '.mtx'
                if mat_type == 'ordered':
                    if impl in ordered_impls:
                        new_gflops[impl_ordered].append(gflops[impl][m])
                else:
                    new_gflops[impl].append(gflops[impl][m])
        new_gflops[config['target']] = np.array(new_gflops[config['target']]) * 2
        # new_gflops[config['baseline']] = np.array(new_gflops[config['baseline']]) * 2
        for impl in new_impls:
            speedups[impl] = np.array(new_gflops[impl]) / np.array(new_gflops[config['baseline']])
        # count number of matrices that have speedup greater than 1
        print(len(np.where(speedups[config['target']] > 1)[0]))
        # for cu,su in enumerate(speedups[config['target']]):
        #     if su > 5:
        #         print(mat_list[cu])
        # speedups[config['target']] = speedups[config['target']] + 0.2
        new_mat_list = [mat for mat in mat_list if mat.split('_')[-1].split('.')[0] != 'ordered']
        plt_x = np.arange(len(new_mat_list))
        for impl in new_impls:
            print(impl, ":", geo_mean_overflow(speedups[impl]))
            print(impl, ":", geo_mean_overflow(new_gflops[impl]))
            # print(impl, ":", geo_mean_overflow(new_gflops[impl]))
            # print(impl)
            # ax.plot(plt_x, new_gflops[impl], color=impl_colors[impl],
            #         label=impl_representations[impl], linewidth=1)
        # ax.plot(plt_x, speedups[config['target']], color=impl_colors[config['target']],
        #         label=impl_representations[config['target']], linewidth=1)
        # speedups[config['target']] = speedups[config['target']] + 0.2
        ax.scatter(new_gflops[config['baseline']], speedups[config['target']], color='maroon',
                       label=impl_representations[impl], s=5)
        # draw a horizontal line at 1
        ax.axhline(y=1, color=impl_colors[config['baseline']], linestyle='--')

        ax.set_xlabel('Unfused GFLOP/s', fontsize=15)
        # ax.set_title('bCol=cCol=' + str(bcol),fontsize=15)
        # set the y scale to 0 to 600
        ax.set_ylim(0, 3)
        # sort the mat_list based on the gflops of the baseline
        new_mat_list, _ = zip(*sorted(zip(new_mat_list, new_gflops[config['baseline']]), key=lambda x: x[1]))
        max_gflops = np.max(new_gflops[config['baseline']])
        min_gflops = np.min(new_gflops[config['baseline']])
        # plt_x = np.linspace(min_gflops, max_gflops, 4)
        # ax.set_xticks(plt_x)
        ax.spines[['right', 'top']].set_visible(False)
        mat_representations = [mat[:-4] for mat in new_mat_list]
        # ax.set_xticks(range(len(mat_representations)), mat_representations, rotation='vertical')
        # ax.legend()
    file_name = log_folder.split('/')[-1] + ".eps"
    plot_path = os.path.join(log_folder, file_name)

    axs[0].set_ylabel('Speedup over\n Unfused', fontsize=15)
    h, l = axs[0].get_legend_handles_labels()
    line = Line2D([0], [0], label='Unfused MKL', color='deepskyblue', linewidth=1)
    # h.extend([line])
    # fig.legend(handles=h, loc='lower center', ncol=3,fontsize=15)
    # fig.suptitle('GeMM-SpMM for ss-graphs on Intel Skylake', fontsize=9)
    plt.show()
    # plt.savefig(plot_path, format='eps')


def plot_gcn_from_logs_folder(logs_folder, config_file, mat_list_file, should_merge="1"):
    # print(should_merge)
    plt.rcParams["font.family"] = "serif"
    with open(mat_list_file) as f:
        mat_list = f.readlines()
    mat_list = [x.strip() for x in mat_list]
    if should_merge == "1":
        merge_logs(logs_folder)
    config = import_config(config_file)
    # plot_gcn(logs_folder, "merged.csv", config, mat_list)
    # plot_gcn_best_threads(logs_folder, "merged.csv", config)
    plot_gcn_reordered_graphs(logs_folder, "merged.csv", config, mat_list)
    # plot_gflops_per_bcol(logs_folder, "merged.csv", config,mat_list)
    # plot_stack_bar(logs_folder, "merged.csv", config)
    # plot_performance_vs_fused_ratio(logs_folder, entry.name, config)
    # print_fusion_ratios(logs_folder, entry.name)
    # for entry in os.scandir(logs_folder):
    #     if entry.name.endswith(".csv") and entry.is_file() and entry.name != "merged.csv":
    #         print(entry.name)
    #         plot_based_on_tile_size(logs_folder, entry.name, config)


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


plot_gcn_from_logs_folder(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

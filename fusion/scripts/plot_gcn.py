import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def take_median(df, **kwargs):
    num_trial = df['Number of Trials'].unique()[0]
    time_array = []
    # for each row in dataframe df
    for index, row in df.iterrows():
        for i in range(num_trial):
            t1 = row['Trial' + str(i) + ' Subregion0 Executor']
            time_array.append(t1)
    return np.median(time_array)


def get_fused_info(matr_list, df, params=None):
    if params is None:  # TODO: these params are hardcoded for now
        # params = [40, 400, 4000, 8000, 10000]
        # params = [4, 8, 40, 100, 1000]
        params = [20, 50, 100, 500, 1000]
        # params = [10, 20, 50, 100, 200]
    fused, fused_20, fused_50, fused_100, fused_500, fused_1000 = [], [], [], [], [], []
    for matr in matr_list:
        cur_matr = df[df['MatrixName'] == matr]
        fused = cur_matr[cur_matr['Implementation Name'] == 'GCN_Fused_Demo']
        fused_20.append(take_median(fused[fused['Iter Per Partition'] == params[0]]))
        fused_50.append(take_median(fused[fused['Iter Per Partition'] == params[1]]))
        fused_100.append(take_median(fused[fused['Iter Per Partition'] == params[2]]))
        fused_500.append(take_median(fused[fused['Iter Per Partition'] == params[3]]))
        fused_1000.append(take_median(fused[fused['Iter Per Partition'] == params[4]]))
    return fused_20, fused_50, fused_100, fused_500, fused_1000


def plot_gcn():
    df_fusion = pd.read_csv('./build/logs/gcn_demo.csv')
    # sort df_fusion based on 'NNZ'
    df_fusion = df_fusion.sort_values(by=['NNZ'])
    # mat_list = df_fusion['MatrixName'].unique()
    mat_list = df_fusion['Matrix Name'].unique()
    bCol = df_fusion['bCols'].unique()[0]
    nnz_list = df_fusion['NNZ'].unique()
    seq_exe_time, separated_exe_time = [], []
    impls = df_fusion['Implementation Name'].unique()
    br = np.arange(len(mat_list))
    k = 0
    bar_width = 0.2
    times = {}
    bars = {}
    for mat in mat_list:
        cur_mat = df_fusion[df_fusion['MatrixName'] == mat]
        for x in impls:
            if x != 'GCN_Fused_Demo':
                if x in times:
                    times[x].append(take_median(cur_mat[cur_mat['Implementation Name'] == x]))
                else:
                    times[x] = [take_median(cur_mat[cur_mat['Implementation Name'] == x])]
    fused_20, fused_50, fused_100, fused_500, fused_1000 = get_fused_info(mat_list,
                                                                                                              df_fusion)
    min_fused = np.minimum(
      np.minimum(np.minimum(np.array(fused_20), np.array(fused_50)), np.array(fused_100)),
                   np.minimum(np.array(fused_500), np.array(fused_1000)))
    times['GCN_Fused_Demo'] = min_fused
    colors = ['purple', 'yellow', 'orange', 'black', 'r', 'g', 'b']
    for impl in impls:
        bars[impl] = [x + k * bar_width for x in br]
        k += 1
    plt.rcParams.update(plt.rcParamsDefault)
    fig, ax = plt.subplots(figsize=(15, 8))
    for impl, bar in bars.items():
        color = colors.pop()
        ax.bar(bar, times[impl], width=bar_width, color=color, edgecolor='grey', label=impl)

    ax.set_xlabel('matrices', fontweight='bold', fontsize=15)
    ax.set_ylabel('run_time', fontweight='bold', fontsize=15)
    ax.set_xticks([r + 1 * bar_width for r in range(0, len(mat_list))],
                  mat_list)

    ax.legend()
    fig.savefig('plot.pdf')


plot_gcn()

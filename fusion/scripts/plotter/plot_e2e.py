import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def plot_e2e(log_folder, log_file_name, ax):
    bcol = log_file_name.split('_')[-1].split('.')[0]
    log_file = os.path.join(log_folder, log_file_name)
    df_e2e = pd.read_csv(log_file, sep=',', header=0, index_col=False)
    # sort df_fusion based on 'NNZ'
    mat_list = list(df_e2e['Graph'].unique())
    mat_list.remove('cora')
    # mat_list = config['matrices']
    # impls = df_e2e['Implementation Name'].unique()
    impls = ['TiledFused', 'DGL GraphConv', 'torch_geometric GCNConv']
    br = np.arange(len(mat_list) * 2, step=2)
    bar_width = 0.4
    plt_x = np.arange(len(mat_list))
    plt.rcParams.update(plt.rcParamsDefault)
    times = {}
    colors = ['deepskyblue', 'violet', 'mediumseagreen', 'goldenrod','blueviolet','black']
    bars = {}
    for impl in impls:
        times[impl] = []
        for mat in mat_list:
            times[impl].append(df_e2e[(df_e2e['Graph'] == mat) & (df_e2e['Impl'] == impl)]['Time'].unique()[0])
            # if type(times[impl][-1]) == str:
            #     if(times[impl][-1][-1] == 's'):
            #         times[impl][-1] = times[impl][-1][:-1]
            #     times[impl][-1] = float(times[impl][-1])
        print(times[impl])
    speedups = {}
    for k, impl in enumerate(impls):
        bars[impl] = [x + k * bar_width for x in br]
        speedups[impl]= (np.array(times['torch_geometric GCNConv']) / times[impl])

    color = colors.pop()
    ax.bar(bars['torch_geometric GCNConv'], speedups['torch_geometric GCNConv'], color=color, width=bar_width, label='torch_geometric GCNConv')
    color = colors.pop()
    ax.bar(bars['DGL GraphConv'], speedups['DGL GraphConv'], color=color, width=bar_width, label='DGL GraphConv')
    color = colors.pop()
    ax.bar(bars['TiledFused'], speedups['TiledFused'], color=color, width=bar_width, label='Tile Fused')

    for i, m in enumerate(mat_list):
        ax.text(bars['torch_geometric GCNConv'][i], speedups['torch_geometric GCNConv'][i]+0.01,"{0:0.2f}".format(times['torch_geometric GCNConv'][i]), rotation='vertical', fontsize=7)


    # ax.set_xlabel('matrices', fontweight='bold', fontsize=7.5)
    mat_repr = [mat.split('.')[0] for mat in mat_list]
    ax.set_xticks(bars['DGL GraphConv'], mat_repr, rotation='vertical')
    ax.set_xlabel('matrices', fontweight='bold', fontsize=7.5)
    ax.set_title(str(bcol), fontsize=7)
    # plot_path = os.path.join(log_folder, ''.join(log_file_name.split('.')[:-1]) + '_' + str(bcol))
    ax.spines[['right', 'top']].set_visible(False)
    # ax.legend()


def plot_e2e_from_logs_folder(logs_folder):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(bottom=0.35, left=0.05, right=0.95, top=0.85, wspace=0.1, hspace=0.1)
    file_name = logs_folder.split('/')[-1] + ".png"
    with os.scandir(logs_folder) as entries:
        entry_names = []

        for entry in entries:
            print(entry.name, "-----------------------------------------------")
            if entry.name.endswith(".csv") and entry.is_file():
                entry_names.append(entry.name)
            # if entry is csv file
        entry_names.sort(key=lambda x: -int(x.split('_')[-1].split('.')[0]))
        for i, entry_name in enumerate(entry_names):
            plot_e2e(logs_folder, entry_name, axs[i])
    plot_path = os.path.join(logs_folder, file_name)
    axs[0].set_ylabel('speed-up', fontweight='bold', fontsize=7.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2)
    # fig.suptitle("GCN E2E")
    fig.savefig(plot_path)



plot_e2e_from_logs_folder(sys.argv[1])

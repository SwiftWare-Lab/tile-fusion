import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())

def plot_e2e(log_folder, log_file_name, ax):
    bcol = log_file_name.split('_')[-1].split('.')[0]
    log_file = os.path.join(log_folder, log_file_name)
    df_e2e = pd.read_csv(log_file, sep=',', header=0, index_col=False)
    # print(df_e2e)
    # df_e2e = df_e2e.sort_values(by=['NNZ'])
    # sort df_fusion based on 'NNZ'
    mat_list = list(df_e2e['matrix'].unique())
    mat_list.sort()
    # mat_list.remove('cora_full')
    # mat_list.remove('coauthor_cs')
    # mat_list.remove('ogbn-proteins')
    # mat_list.remove('ogbn-proteins_ordered')
    # mat_list.remove('ogbn-products')
    # mat_list.remove('ogbn-products_ordered')
    # mat_list.remove('reddit2')
    # mat_list.remove('reddit2_ordered')
    # mat_list = config['matrices']
    # impls = df_e2e['Implementation Name'].unique()
    # impls = ['TiledFused', 'DGL GraphConv', 'torch_geometric GCNConv']
    impls = ['FusedGCNConv', 'torch_geometric GCNConv'] #,'DGL GraphConv']
    new_impls = df_e2e['impl'].unique()
    print(new_impls)
    br = np.arange(len(mat_list), step=2)
    bar_width = 0.6
    plt_x = np.arange(len(mat_list))
    plt.rcParams.update(plt.rcParamsDefault)
    times = {}
    colors = ['deepskyblue', 'violet', 'mediumseagreen', 'goldenrod','blueviolet','black']
    bars = {}
    for impl in impls:
        times[impl] = []
        for mat in mat_list:
            if not mat.endswith('ordered'):
                print(mat, impl, bcol)
                # print(df_e2e[(df_e2e['matrix'] == mat) & (df_e2e['impl'] == impl)]['time'].unique())
                # print(mat, bcol, impl)
                times[impl].append(df_e2e[(df_e2e['matrix'] == mat) & (df_e2e['impl'] == impl)]['time'].unique()[0])
            # if type(times[impl][-1]) == str:
            #     if(times[impl][-1][-1] == 's'):
            #         times[impl][-1] = times[impl][-1][:-1]
            #     times[impl][-1] = float(times[impl][-1])
        # print(times[impl])
    fused_gcn_orderd_impl = 'FusedGCNConv RCM Ordered'
    impls.append(fused_gcn_orderd_impl)
    times[fused_gcn_orderd_impl] = []
    new_mat_list = [mat for mat in mat_list if not mat.endswith('ordered')]
    for mat in mat_list:
        # print(mat)
        if mat.endswith('ordered'):
            times[fused_gcn_orderd_impl].append(df_e2e[(df_e2e['matrix'] == mat) & (df_e2e['impl'] == 'FusedGCNConv')]['time'].unique()[0])
    speedups = {}
    for k, impl in enumerate(impls[:-1]):
        bars[impl] = [x + k * bar_width for x in br]
        speedups[impl] = (np.array(times['torch_geometric GCNConv']) / times[impl])
    # print(speedups['FusedGCNConv'])
    # print(speedups[fused_gcn_orderd_impl])
    print(geo_mean_overflow(speedups['torch_geometric GCNConv']))
    color = colors.pop()
    # ax.bar(bars['torch_geometric GCNConv'], speedups['torch_geometric GCNConv'], color=color, width=bar_width, label='torch_geometric GCNConv')
    # print(speedups['torch_geometric GCNConv'])
    print(geo_mean_overflow(speedups['FusedGCNConv']))
    ax.bar(bars['torch_geometric GCNConv'], speedups['torch_geometric GCNConv'], color='green', width=bar_width, label='PyG')
    # ax.bar(bars['DGL GraphConv'], speedups['DGL GraphConv'], color=color, width=bar_width, label='DGL GraphConv')
    color = colors.pop()
    # print(geo_mean_overflow(speedups[fused_gcn_orderd_impl]))
    ax.bar(bars['FusedGCNConv'], speedups['FusedGCNConv'], color='maroon', width=bar_width, label='Tile Fusion')
    # ax.bar(bars[fused_gcn_orderd_impl], speedups[fused_gcn_orderd_impl], color='maroon', width=bar_width, label='Tile Fused')

    for i, m in enumerate(new_mat_list):
        # ax.text(bars['torch_geometric GCNConv'][i], speedups['torch_geometric GCNConv'][i]+0.4,"{0:0.2f}".format(times['torch_geometric GCNConv'][i]), rotation='vertical', fontsize=11)
        # ax.text(bars['DGL GraphConv'][i] - bar_width/2, speedups['DGL GraphConv'][i]+0.1,"{0:0.2f}".format(times['DGL GraphConv'][i]), rotation='vertical', fontsize=7)
        ax.text(bars['torch_geometric GCNConv'][i], speedups['torch_geometric GCNConv'][i]+0.09,"{0:0.2f}".format(times['torch_geometric GCNConv'][i]), rotation='vertical', fontsize=8)

    # ax.set_xlabel('matrices', fontweight='bold', fontsize=7.5)
    mat_repr = [mat.split('.')[0] for mat in mat_list if not mat.endswith('ordered')]
    print(mat_repr)
    # ind = mat_repr.index('reddit2')
    # mat_repr[ind] = 'reddit'
    ind = mat_repr.index('facebook_page_page')
    mat_repr[ind] = 'facebook'
    ind = mat_repr.index('deezer_europe')
    mat_repr[ind] = 'deezer'
    # ax.set_xticks(bars['torch_geometric GCNConv'], mat_repr, rotation=60)
    ax.set_xticks(bars['FusedGCNConv'],range(len(mat_repr)))
    ax.set_xlabel('Graph IDs', fontsize=15)
    # ax.set_title('eDim='+str(bcol), fontsize=15)
    ax.set_ylim(0, 4)
    # plot_path = os.path.join(log_folder, ''.join(log_file_name.split('.')[:-1]) + '_' + str(bcol))
    ax.spines[['right', 'top']].set_visible(False)
    # ax.legend()
    ax.set_ylabel('Speedup over PyG', fontsize=15)


def plot_e2e_from_logs_folder(logs_folder):
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.7))
    # fig.subplots_adjust(bottom=0.35, left=0.05, right=0.95, top=0.85, wspace=0.1, hspace=0.1)
    fig.subplots_adjust(bottom=0.2, left=0.1, right=1, top=0.85, wspace=0.1, hspace=0.1)
    bcols = [64]
    file_name = logs_folder.split('/')[-1] + ".eps"
    with os.scandir(logs_folder) as entries:
        entry_names = []

        for entry in entries:
            print(entry.name, "-----------------------------------------------")
            bcol = entry.name.split('_')[-1].split('.')[0]
            if entry.name.endswith("64.csv") and entry.is_file() and int(bcol) in bcols:

                entry_names.append(entry.name)
            # if entry is csv file
        entry_names.sort(key=lambda x: -int(x.split('_')[-1].split('.')[0]), reverse=True)
        for i, entry_name in enumerate(entry_names):
            plot_e2e(logs_folder, entry_name, ax)
    plot_path = os.path.join(logs_folder, file_name)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2,bbox_to_anchor=(0.54, 1.02),fontsize=15,frameon=False)
    # fig.suptitle("GCN E2E")
    plt.show()
    fig.savefig(plot_path, format='png')



plot_e2e_from_logs_folder(sys.argv[1])

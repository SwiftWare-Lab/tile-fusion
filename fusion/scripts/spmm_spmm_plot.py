import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from copy import deepcopy


def take_median(df, **kwargs):
    num_trial = df['Number of Trials'].unique()[0]
    time_array = []
    # for each row in dataframe df
    for index, row in df.iterrows():
        for i in range(num_trial):
            t1 = row['Trial' + str(i) + ' Subregion0 Executor']
            time_array.append(t1)
    return np.median(time_array)


impls = [
    "GPU_Unfused_SeqReduceRowBalance",
    "GPU_Unfused_SeqReduceRowCoarsened",
    "GPU_Fused_Reordered_HighFusionRatio",
    "GPU_AtomicFused_Reordered_HighFusionRatio",
    "GPU_CSRCSCAtomicFused_Reordered_HighFusionRatio"

]

impl_reprs = {
    "GPU_Unfused_SeqReduceRowBalance": "Unfused Baseline",
    "GPU_Unfused_SeqReduceRowCoarsened": "Unfused Coarsened",
    "GPU_Fused_Reordered_HighFusionRatio": "Tile Fused",
    "GPU_AtomicFused_Reordered_HighFusionRatio": "Atomic Tiled Fused",
    "GPU_CSRCSCAtomicFused_Reordered_HighFusionRatio": "1v1 Atomic Fused"
}

spmm_spmm_impl_row_tiles = {
    "GPU_Unfused_SeqReduceRowCoarsened": [8, 16, 32, 64],
    "GPU_Fused_Reordered_HighFusionRatio": [8, 16, 32, 64],
    "GPU_AtomicFused_Reordered_HighFusionRatio": [8, 16, 32, 64],
    "GPU_CSRCSCAtomicFused_Reordered_HighFusionRatio": [8, 16, 32, 64]
}


def plot_gpu_speedup_tuned_best_of_unfused_coarsened_and_fused(log_folder, log_file_name):
    log_file = os.path.join(log_folder, log_file_name)
    df_benchmark = pd.read_csv(log_file)
    df_benchmark = df_benchmark.sort_values(by=['NNZ'])
    mat_list = df_benchmark['Matrix Name'].unique()
    bcols = df_benchmark['bCols'].unique()
    bcols = np.sort(bcols)
    impl_row_tiles = {impl: [impl + '_' + str(row_tile) for row_tile in spmm_spmm_impl_row_tiles[impl]] for impl in spmm_spmm_impl_row_tiles}
    # print(impls, impl_row_tiles)
    baseline_impls = ['GPU_Unfused_SeqReduceRowBalance']
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(bottom=0.3, left=0.06, right=1, top=0.75, wspace=0.1, hspace=0.1)
    baseline_impl = baseline_impls[0]
    target_impls = deepcopy(impls)
    target_impls.append('best_of_coarsened')
    impl_reprs['best_of_coarsened'] = 'best of coarsened'
    for bi in baseline_impls:
        target_impls.remove(bi)
    for i,bcol in enumerate(bcols):
        times = {}
        df_benchmark_bcol = df_benchmark[df_benchmark['bCols'] == bcol]
        mat_list = df_benchmark_bcol['Matrix Name'].unique()
        mat_list = [m for m in mat_list if not m.endswith('10000.mtx')]
        for impl in impls:
            times[impl] = []
        mat_gflops = []
        for mat in mat_list:
            cur_mat = df_benchmark_bcol[df_benchmark_bcol['Matrix Name'] == mat]
            cur_mat_nnz = cur_mat['NNZ'].unique()[0]
            mat_gflops.append((cur_mat_nnz * bcol + cur_mat_nnz * bcol) / 1e9)
            for impl in impls:
                if impl in impl_row_tiles:
                    all_times = []
                    for row_tile in impl_row_tiles[impl]:
                        cur_impl_row_tile = cur_mat[cur_mat['Implementation Name'] == row_tile]
                        all_times.append(take_median(cur_impl_row_tile))
                    best_time = min(all_times)
                    times[impl].append(best_time)
                else:
                    cur_impl = cur_mat[cur_mat['Implementation Name'] == impl]
                    cur_impl = cur_impl[cur_impl['bCols'] == bcol]
                    times[impl].append(take_median(cur_impl))
        gflops = {}
        for impl in impls:
            gflops[impl] = np.array(mat_gflops) / np.array(times[impl])

        for bi in baseline_impls:
            gflops[bi] = np.array(mat_gflops) / np.array(times[bi])
        speed_ups = {}
        ax = axs[i]
        mixed_impls = ['GPU_Fused_Reordered_HighFusionRatio', 'GPU_AtomicFused_Reordered_HighFusionRatio', 'GPU_Unfused_SeqReduceRowCoarsened']
        times['best_of_coarsened'] = times['GPU_Unfused_SeqReduceRowCoarsened']
        for ii, impl in enumerate(mixed_impls):
            times['best_of_coarsened'] = np.minimum(times['best_of_coarsened'], times[impl])
        for ii, impl in enumerate(target_impls):
            if impl in mixed_impls:
                continue
            if impl != 'best_of_coarsened':
                continue
            speed_ups[impl] = np.array(times[baseline_impl]) / np.array(times[impl])
            print(impl)
            # compute gmean of speed-ups
            gmean = np.exp(np.mean(np.log(speed_ups[impl])))
            print("gmean: ",  gmean, "bcol: ", bcol)
            # max and min speed-up and perentage of matrices higehr than one
            max_speed_up = np.max(speed_ups[impl])
            min_speed_up = np.min(speed_ups[impl])
            percentage = np.sum(speed_ups[impl] > 1) / len(speed_ups[impl])
            print("max speed-up: ", max_speed_up, "min speed-up: ", min_speed_up, "percentage: ", percentage)
            impl_repr = impl_reprs[impl]
            ax.scatter(gflops[baseline_impl], speed_ups[impl], label=impl_repr, s=5, color='#8B0000')
            # ax.scatter(mat_list, speed_ups[impl], label=impl_repr, s=20)
            ax.axhline(y=1, color='black', linestyle='--')
            ax.spines[['right', 'top']].set_visible(False)
            # set x-label
            ax.set_xlabel('Unfused GFLOP/s', fontsize=14)
            # set y-axis to start from zero to 2.5

            # set title

        #ax.set_title('bCol=cCol'+str(bcol))

        #increase font size of labels
        ax.tick_params(axis='both', which='major', labelsize=12)

        #set limit for the y-axis
        ax.set_ylim([0.0, 4.0])
        # set y-axis to go from 0 to 4 with step 1
        ax.set_yticks(np.arange(0, 4.1, 1))


        print(bcol)
        # for i,su in enumerate(speed_ups['GPU_CSRCSCAtomicFused_Reordered_HighFusionRatio']):
        #     if (su < 0.95):
        #         print(mat_list[i], su)
    #legend with one row outside of the plot and in the upper part
    #axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=5)
    # axs[1].legend()

    #fig.suptitle('speed-up over ' + baseline_impl)
    axs[0].set_ylabel('Speedup over \n Unfused Baseline', fontsize=17)
    #plt.show()
    plt.savefig(os.path.join(log_folder, 'A100GPU_spmm_spmm.pdf'), bbox_inches='tight')


def merge_logs(logs_folder):
    plt.close()
    with os.scandir(logs_folder) as entries:
        entry_names = []
        for entry in entries:
            # if entry is csv file
            if entry.name.endswith(".csv") and entry.is_file():
                entry_names.append(entry.name)

        # entry_names = entry_names.sort(key=lambda x: int(x.split("_")[0]))
        print(entry_names[0])
        df = pd.read_csv(os.path.join(logs_folder, entry_names[0]))
        for i in range(1, len(entry_names)):
            print(entry_names[i])
            df = pd.concat([pd.read_csv(os.path.join(logs_folder, entry_names[i])), df], ignore_index=True)
        df.to_csv(os.path.join(logs_folder, "merged.csv"), index=False)


def plot_gcn_from_logs_folder(logs_folder, should_merge="1"):
    # print(should_merge)
    plt.rcParams["font.family"] = "serif"
    if should_merge == "1":
        merge_logs(logs_folder)

    plot_gpu_speedup_tuned_best_of_unfused_coarsened_and_fused(logs_folder, "merged.csv")

plot_gcn_from_logs_folder(sys.argv[1], sys.argv[2])
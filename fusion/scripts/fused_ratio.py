# import the required libraries

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # read csv file as pandas dataframe
    df = pd.read_csv(sys.argv[1])
    best_list = []
    # find unique mmatrix names
    orig_matrix_names = df['Matrix Name'].unique()
    matrix_names = []
    for matrix_name in orig_matrix_names:
        # find the matrix
        matrix = df.loc[df['Matrix Name'] == matrix_name]
        if matrix['NNZ'].iloc[0] < 50000:
            continue
        matrix_names.append(matrix_name)

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
    plt.scatter(best_df['NNZ'], best_df['Number of Fused nnz0']/best_df['NNZ'])
    plt.ylabel('Fused Ratio')
    # make y-axis log scale
    plt.yscale('log')
    plt.xlabel('NNZ')
    # add a horizontal line at 0
    plt.show()
    plt.close()
    # print min and max of fused ratio
    print("min fused ratio:", (best_df['Number of Fused nnz0']/best_df['NNZ']).min())
    print("max fused ratio:", (best_df['Number of Fused nnz0']/best_df['NNZ']).max())
    # print where the min fused ratio is zero
    print(np.where((best_df['Number of Fused nnz0']/best_df['NNZ']) == 0))

    # plot the best 'MTile' per matrix nnz
    plt.scatter(best_df['NNZ'], best_df['MTile'])
    plt.ylabel('MTile')
    plt.xlabel('NNZ')
    plt.show()
    plt.close()

    # plot a pie chart of 'MTile' distribution
    best_df['MTile'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.show()
    plt.close()



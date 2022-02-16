#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    rankings.py
# @Author:      Daniel Puente Ramírez
# @Time:        6/2/22 17:43

import numpy as np
import pandas as pd
from os import walk
from os.path import join
from utils.dir import check_dir
import matplotlib.pyplot as plt


def sort_cols(df):
    keys = df.keys().tolist()
    keys = keys[:2] + keys[3:]
    tmp_vals = []

    for row in df.iterrows():
        arr = row[1][3:]
        assert len(arr) == 3
        if len(np.unique(arr)) + 2 == len(arr):
            ranks = [1, 1, 1]
        elif len(np.unique(arr)) + 1 == len(arr):
            idx = np.where(arr == arr[0])
            if len(idx) == 1 and arr[0] > arr[1] and arr[0] > arr[2]:
                ranks = [1, 2, 2]
            else:
                if arr[0] == arr[2] and arr[1] > arr[0]:
                    ranks = [2, 1, 2]
                else:
                    ranks = [2, 2, 1]
        else:
            order = arr.argsort()
            ranks = [x + 1 for x in order.argsort().tolist()]

        tmp_vals.append([*row[1][:2], *ranks])

    df_fin = pd.DataFrame(tmp_vals, columns=['dataset', 'percent labeled',
                                             'original',
                                             'self training',
                                             'ENN'])\
        .groupby('percent labeled').mean()

    return df_fin


if __name__ == '__main__':
    folder = join('.', 'tests')
    ranked = join('ranks', 'ranked')
    training_method = 'self_training'
    results_found = next(walk(folder), (None, None, []))[2]
    results_found.sort()

    check_dir(folder)
    check_dir(join(folder, ranked))

    working_results = [r for r in results_found if training_method in r]
    if len(working_results) > 1:
        print('Select the date to visualize:')
        for i, result in enumerate(working_results):
            print(f'{i}.', result.split('.')[0].split('_')[-1])
        date = int(input(f'Select [0-{len(working_results) - 1}]'))
    else:
        date = 0

    results_dir = join(folder, ranked, working_results[date].split('.')[0])
    check_dir(results_dir)

    df = pd.read_csv(join(folder, working_results[date]))
    df = df.fillna(0)
    metrics = ['f1-score', 'mean squared error', 'accuracy score']

    for metric in metrics:
        df_f1 = df[['dataset', 'percent labeled', 'fold',
                    f'{metric} SVC', f'{metric} before',
                    f'{metric} after without deletion']].copy()

        df_fin = sort_cols(df_f1)

        df_fin.to_csv(join(results_dir, f'ranks_{metric}.csv'), index=True)

        ax = df_fin.plot(
            title=f"Summary with {metric}",
            ylabel="Average rank",
            xlabel='% labeled',
            grid=True,
            legend='lower right'
        )
        ax.set_xticks(df_fin.index)
        ax.set_xticklabels([str(int(float(x)*100)) for x in df_fin.index])
        plt.savefig(join(results_dir, f'summary_{metric}.png'))

    print(f'Results in {results_dir}')

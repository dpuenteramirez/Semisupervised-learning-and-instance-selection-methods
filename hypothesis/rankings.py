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

if __name__ == '__main__':
    folder = join('.', 'tests', 'ranks')
    ranked = 'ranked'
    training_method = 'self_training'
    results_found = next(walk(folder), (None, None, []))[2]
    results_found.sort()

    check_dir(folder)
    check_dir(join(folder, ranked))

    working_results = [r for r in results_found if training_method in r]
    if len(working_results) > 9:
        print('Select the date to visualize:')
        for i, result in enumerate(working_results):
            print(f'{i}.', result.split('.')[0].split('_')[-1])
        date = int(input(f'Select [0-{len(working_results) - 1}]'))
    else:
        date = 3

    results_dir = join(folder, ranked, working_results[date].split('.')[0])
    check_dir(results_dir)

    df = pd.read_csv(join(folder, working_results[date]))
    df = df.fillna(0)
    df_fin = pd.DataFrame(columns=df.keys())

    for row in df.iterrows():
        arr = row[1][2:]
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
        df_fin = df_fin.append(pd.DataFrame([[*row[1][:2], *ranks]],
                                            columns=df.keys()),
                               ignore_index=True)

    df_fin.to_csv(join(results_dir, 'ranks.csv'), index=False)

    df_means = df[df.keys()[1:]]
    df_means = df_means.groupby('% labeled').mean()
    df_means.to_csv(join(results_dir, 'summary.csv'))
    df_means.plot(
        title="Summary of experiments",
        ylabel="Precision",
        grid=True,
        xticks=df_fin['% labeled']
    )
    plt.savefig(join(results_dir, 'summary.png'))

    # df_1 = df.groupby(['dataset', '% labeled'])
    # df_1 = df[['dataset', '% labeled']].copy().join(df_1.rank(axis=1,
    #                                                           method='dense'))
    # df_1.to_csv(join(folder, ranked, working_results[date]), index=False)
    print(f'Results in {results_dir}')

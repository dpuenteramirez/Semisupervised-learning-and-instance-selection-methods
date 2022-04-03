#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    results.py
# @Author:      Daniel Puente Ramírez
# @Time:        1/4/22 12:02

from os import walk
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    folder = join('.', 'results', '')
    ranks_path = 'ranks'
    plots = 'plots'
    precisions = [0.05, 0.1, 0.15]
    metrics = ['f1-score', 'mean squared error', 'accuracy score']
    results_found = next(walk(folder), (None, None, []))[2]
    if len(results_found) != 3:
        print("This script only works with 3 results in the \'results\' "
              "folder.")
        exit(1)
    dfs = []
    for index, r in enumerate(results_found):
        dfs.append(pd.read_csv(folder + results_found[index]))

    # Fill NaN values?
    df = pd.concat(dfs, ignore_index=True)
    df.drop('fold', axis=1, inplace=True)

    df = df.groupby(['base', 'filter', 'percent labeled']).mean()

    classifiers = dfs[0].base.unique()
    filter_method = dfs[0]['filter'].unique()

    ranks = {}
    for classifier in classifiers:
        values = {}
        for precision in precisions:
            temp = []
            for fm in filter_method:
                temp.append(df.loc[(classifier, fm, precision)].to_numpy())

            for index in range(len(metrics)):
                vals = []
                for index2 in range(len(filter_method)):
                    vals.append(temp[index2][index])

                if len(np.unique(vals)) + 2 == len(vals):
                    r = [1, 1, 1]
                elif len(np.unique(vals)) + 1 == len(vals):
                    idx = np.where(vals == vals[0])
                    if len(idx) == 1 and vals[0] > vals[1] and vals[0] > \
                            vals[2]:
                        r = [1, 2, 2]
                    else:
                        if vals[0] == vals[2] and vals[1] > vals[0]:
                            r = [2, 1, 2]
                        else:
                            r = [2, 2, 1]
                else:
                    vals = np.array(vals)
                    r = vals.argsort()
                    r = [x + 1 for x in r.argsort().tolist()]

                values[precision, metrics[index]] = r

        ranks[classifier] = values

    for classifier in classifiers:
        df_fin = pd.DataFrame(ranks.get(classifier), index=filter_method). \
            transpose()

        for metric in metrics:
            vals = []
            for p in precisions:
                vals.append(df_fin.loc[(p, metric)])
            df_f = pd.concat(vals, ignore_index=True, axis=1).transpose()
            df_f.index = precisions

            df_f.plot(
                title="Summary of experiments",
                ylabel="Precision",
                grid=True,
                xticks=precisions
            )
            plt.savefig(join(plots, f'{classifier}_{metric}.png'))
        df_fin.to_csv(join(ranks_path, f'{classifier}.csv'))

    df.to_csv(join(ranks_path, 'results.csv'))

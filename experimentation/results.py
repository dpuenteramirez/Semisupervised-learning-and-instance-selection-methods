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

    mse = 'mean squared error'
    pl = 'percent labeled'

    folder = join('.', 'results', '')
    ranks_path = 'ranks'
    plots = 'plots'
    # A list of the percentages of the data that is labeled.
    precisions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    percent_precisions = [x * 100 for x in precisions]
    metrics = ['f1-score', mse, 'accuracy score']
    results_found = next(walk(folder), (None, None, []))[2]
    if len(results_found) != len(precisions) + 1:
        print(
            f"This script only works with {len(precisions) + 1} results in the "
            f"\'results\' folder.")
        exit(1)
    dfs = []
    for index, r in enumerate(results_found):
        dfs.append(pd.read_csv(folder + results_found[index]))

    df = pd.concat(dfs, ignore_index=True)
    df.drop(['fold', 'Unnamed: 0'], axis=1, inplace=True)

    classifiers = dfs[0].base.unique()
    filters = np.append(dfs[0]['filter'].unique(), 'base')
    datasets = dfs[0]['dataset'].unique()

    ranks = {}
    vals = ['base', 'filter', pl, 'f1-score',
            'mean squared error', 'accuracy score']

    means = {}
    for classifier in classifiers:
        cl = []
        for dataset in datasets:
            rows = df[df['dataset'] == dataset]
            for precision in precisions:
                temp = pd.DataFrame(index=filters, columns=metrics)
                temp[pl] = precision
                p_rows = rows.loc[
                    (rows['base'] == classifier) & (rows[pl] == precision)]
                vals = p_rows.groupby(['filter']).mean()

                for metric in metrics:
                    dff = vals[metric].to_frame()
                    if metric != mse:
                        p = dff.rank(ascending=False)
                    else:
                        p = dff.rank(ascending=True)
                    temp[metric] = p.to_numpy()
                cl.append(temp)
        means[classifier] = cl

    for classifier in classifiers:
        dff = pd.concat(means[classifier])
        rks = {}
        for metric in metrics:
            for precision in precisions:
                rows = dff.loc[dff[pl] == precision]
                vals = rows[metric].to_frame()
                vals = vals.groupby(level=0).mean()
                rks[(precision, metric)] = np.ravel(vals.to_numpy())
        ranks[classifier] = rks

    fig, axs = \
        plt.subplots(nrows=3, ncols=3, sharex='all', sharey='all', figsize=(
            12, 5))

    for (i, classifier), axss in zip(enumerate(classifiers), axs):
        df_fin = pd.DataFrame(ranks.get(classifier), index=filters). \
            transpose()

        for (j, metric), ax in zip(enumerate(metrics), axss):
            vals = []
            for p in precisions:
                vals.append(df_fin.loc[(p, metric)])
            df_f = pd.concat(vals, ignore_index=True, axis=1).transpose()
            df_f.index = percent_precisions

            ax.plot(df_f)

            if i == 0:
                ax.set_title(str(metric))

            if j == 0:
                classifier = classifier.split('Classifier')[0]
                ax.set_ylabel(str(classifier))

            # ax = df_f.plot(
            #     title=f"Summary of {classifier} with {metric}",
            #     ylabel="Average Rank",
            #     xlabel="Percent Labeled",
            #     grid=True,
            #     xticks=percent_precisions,
            # )
            # plt.savefig(fname=join(plots, f'{classifier}_{metric}.png'),
            #            dpi=300)
        df_fin.to_csv(join(ranks_path, f'{classifier}.csv'))

    fig.legend(
        labels=filters,
        loc="center right",
    )
    plt.subplots_adjust(right=0.9)

    plt.savefig(fname=join(plots, 'General.png'), dpi=300)
    df.to_csv(join(ranks_path, 'results.csv'), index=False)
    print('Plots generated and its CSV')
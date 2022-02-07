#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    results_visualization.py
# @Author:      Daniel Puente Ramírez
# @Time:        25/1/22 16:01

import os
from os import walk
from os.path import join

import numpy as np
import pandas as pd

from utils.custom_plots import plot_bar_line
from utils.reading_tests import DatasetResult

if __name__ == '__main__':
    metric = 'f1'  # 'mse' or 'acc'
    header = ['dataset', '% labeled', 'original', 'self training', 'ENN']
    folder = join('.', 'tests')
    ranks = 'ranks'
    plots = 'plots'
    training_method = 'self_training'
    results_found = next(walk(folder), (None, None, []))[2]
    results_found.sort()

    if not os.path.isdir(join(folder, plots)):
        os.mkdir(join(folder, plots))
        if os.path.isdir(join(folder, plots)):
            print(f'Created main folder for plots. {join(folder, plots)}')
        else:
            print(f'Create manually the folder \'{plots}\' inside {folder} and '
                  f'rerun.')

    if not os.path.isdir(join(folder, ranks)):
        os.mkdir(join(folder, ranks))
        if os.path.isdir(join(folder, ranks)):
            print(f'Created main folder for ranks. {join(folder, ranks)}')
        else:
            print(f'Create manually the folder \'{ranks}\' inside {folder} and '
                  f'rerun.')

    working_results = [r for r in results_found if training_method in r]
    if len(working_results) > 1:
        print('Select the date to visualize:')
        for i, result in enumerate(working_results):
            print(f'{i}.', result.split('.')[0].split('_')[-1])
        date = int(input(f'Select [0-{len(working_results)}]'))
    else:
        date = 0

    if not os.path.isdir(join(folder, plots, working_results[date].split(
            '.')[0])):
        os.mkdir(join(folder, plots, working_results[date].split('.')[0]))
        print('Created folder for plots saving.')

    file_df = pd.read_csv(join(folder, working_results[date]))

    datasets_names = file_df.dataset.unique()
    bar_width = 0.35

    mean_stats = []

    for index, name in enumerate(datasets_names):
        rows = file_df.loc[file_df['dataset'] == name]
        precision = rows['percent labeled'].unique()
        stats = np.array([[name, pre] for pre in precision])
        folds = rows['fold'].max() + 1
        name = name.split('.')[0]

        samples_after_sl = [
            np.nanmean(x) for x in rows['samples after self-training'][::folds]
        ]

        data_svc = DatasetResult(name, precision, folds,
                                 rows['initial samples'],
                                 rows['f1-score SVC'],
                                 rows['mean squared error SVC'],
                                 rows['accuracy score SVC'])

        data_before = DatasetResult(name, precision, folds,
                                    rows['initial samples'],
                                    rows['f1-score before'],
                                    rows['mean squared error before'],
                                    rows['accuracy score before']
                                    )
        data_after_with_deletion = \
            DatasetResult(name, precision, folds,
                          rows['samples after filtering with deletion'],
                          rows['f1-score after with deletion'],
                          rows['mean squared error after with deletion'],
                          rows['accuracy score after with deletion']
                          )
        data_after_without_deletion = \
            DatasetResult(name, precision, folds,
                          rows['samples after filtering without deletion'],
                          rows['f1-score after without deletion'],
                          rows['mean squared error after without deletion'],
                          rows['accuracy score after without deletion']
                          )

        n_groups = len(precision)
        x = np.arange((len(precision)))

        if metric == 'f1':
            metric_name = 'f1-score'
            svc = data_svc.f1
            before_filtering = data_before.f1
            after_with_deletion = data_after_with_deletion.f1
            after_without_deletion = data_after_without_deletion.f1
        elif metric == 'mse':
            metric_name = 'mean squared error'
            svc = data_svc.mse
            before_filtering = data_before.mse
            after_with_deletion = data_after_with_deletion.mse
            after_without_deletion = data_after_without_deletion.mse
        elif metric == 'acc':
            metric_name = 'accuracy score'
            svc = data_svc.acc
            before_filtering = data_before.acc
            after_with_deletion = data_after_with_deletion.acc
            after_without_deletion = data_after_without_deletion.acc
        else:
            raise ValueError(f'{metric} Metric not supported.')

        if index != 0:
            mean_stats = np.concatenate(
                (mean_stats, np.concatenate(
                    (stats, np.array([svc]).T,
                     np.array([before_filtering]).T,
                     np.array([after_without_deletion]).T),
                    axis=1)),
                axis=0
            )
        else:
            mean_stats = np.concatenate(
                (stats, np.array([svc]).T,
                 np.array([before_filtering]).T,
                 np.array([after_without_deletion]).T),
                axis=1)

        data = {'% labeled': x,
                'SVC': svc,
                'Before Filtering': before_filtering,
                'After Filtering': after_with_deletion,
                'Samples Before Filtering': data_before.n_samples,
                'Samples After Self Training': samples_after_sl,
                'Samples After Filtering': data_after_with_deletion.n_samples
                }

        data_df = pd.DataFrame(data)
        plot_bar_line(name, metric_name, precision, data_df,
                      join(folder, plots, working_results[date].split('.')[0],
                           f'Dataset_{name}_{metric}_with_deletion'))

        data_df['Samples After Filtering'] = \
            data_after_without_deletion.n_samples

        plot_bar_line(name, metric_name, precision, data_df,
                      join(folder, plots, working_results[date].split('.')[0],
                           f'Dataset_{name}_{metric}_without_deletion'))

    pd.DataFrame(mean_stats, columns=header).to_csv(
        join(folder, ranks, working_results[date]),
        index=False
    )

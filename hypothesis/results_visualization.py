#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    results_visualization.py
# @Author:      Daniel Puente Ramírez
# @Time:        25/1/22 16:01

import numpy as np
import pandas as pd
from os import walk
import matplotlib.pyplot as plt
from os.path import join
from utils.reading_tests import DatasetResult

if __name__ == '__main__':
    folder = join('.', 'tests')
    training_method = 'self_training'
    results_found = next(walk(folder), (None, None, []))[2]
    results_found.sort()

    working_results = [r for r in results_found if training_method in r]
    if len(working_results) > 2:
        print('Select the date to visualize:')
        i = 0
        for result in working_results:
            print(result)
            if i % 2 == 0:
                print(f'{i}.', result.split('.')[0].split('_')[-1])
                i += 1
            else:
                continue
        date = int(input(f'Select [0-{len(working_results)}]')) * 2
    else:
        date = 0
    pre_post_dict = {'post': pd.read_csv(join(folder, working_results[date])),
                    'pre': pd.read_csv(join(folder, working_results[date + 1]))
                    }

    if len(pre_post_dict['pre']) != len(pre_post_dict['post']):
        print('Files items do not match. First different pair:\n')
        print('\tpre\t\tpost')
        for i in range(len(pre_post_dict['pre'])):
            pre = pre_post_dict['pre'].iloc[[i]].to_numpy()
            post = pre_post_dict['post'].iloc[[i]].to_numpy()

            if np.array_equal(pre[0][:3], post[0][:3]):
                continue
            else:
                print(i, pre[0][:3], post[0][:3])
                break

    datasets_names = pre_post_dict['pre'].dataset.unique()
    results_dict = dict.fromkeys(datasets_names)

    for name in results_dict.keys():
        rows_pre = pre_post_dict['pre'].loc[pre_post_dict['pre']['dataset'] ==
                                            name]
        rows_post = pre_post_dict['post'].loc[pre_post_dict['post'][
                                                  'dataset'] == name]
        precision = rows_pre['percent labeled'].unique()
        n_iterations = rows_pre['iteration'].max()
        results_dict[name] = {
            'pre': DatasetResult(name, precision, n_iterations,
                                 rows_pre['f1-score'].to_list(),
                                 rows_pre['mean squared error'].to_list(),
                                 rows_pre['accuracy score'].to_list()
                                 ),
            'post': DatasetResult(name, precision, n_iterations,
                                  rows_post['f1-score'].to_list(),
                                  rows_post['mean squared error'].to_list(),
                                  rows_post['accuracy score'].to_list()
                                  )
        }

    f1_mean_pre = np.array([x['pre'].f1 for x in results_dict.values()])
    mse_mean_pre = np.array([x['pre'].mse for x in results_dict.values()])
    acc_mean_pre = np.array([x['pre'].acc for x in results_dict.values()])
    f1_mean_post = np.array([x['post'].f1 for x in results_dict.values()])
    mse_mean_post = np.array([x['post'].mse for x in results_dict.values()])
    acc_mean_post = np.array([x['post'].acc for x in results_dict.values()])

    n_groups = len(f1_mean_post[0])
    x = np.arange((len(f1_mean_pre[0])))
    bar_width = 0.35
    for index, (pre, post) in enumerate(zip(f1_mean_pre, f1_mean_post)):
        plt.bar(x, pre, width=bar_width, label='Before Filtering')
        plt.bar(x+bar_width, post, width=bar_width, label='After Filtering')
        plt.legend(loc='best')
        plt.xticks(x+bar_width, precision)
        plt.ylabel('Value')
        plt.xlabel('% labeled')
        plt.title(f'Dataset {datasets_names[index]} - f1-score')
        plt.show()

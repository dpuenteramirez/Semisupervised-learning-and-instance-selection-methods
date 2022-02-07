#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    rankings.py
# @Author:      Daniel Puente Ramírez
# @Time:        6/2/22 17:43

import os
import numpy as np
import pandas as pd
from os import walk
from os.path import join


if __name__ == '__main__':
    folder = join('.', 'tests', 'ranks')
    ranked = 'ranked'
    training_method = 'self_training'
    results_found = next(walk(folder), (None, None, []))[2]
    results_found.sort()

    if not os.path.isdir(join(folder, ranked)):
        os.mkdir(join(folder, ranked))
        if os.path.isdir(join(folder, ranked)):
            print(f'Created main folder for ranks solutions.'
                  f' {join(folder, ranked)}')
        else:
            print(f'Create manually the folder \'{ranked}\' inside {folder} and'
                  f' rerun.')

    working_results = [r for r in results_found if training_method in r]
    if len(working_results) > 1:
        print('Select the date to visualize:')
        for i, result in enumerate(working_results):
            print(f'{i}.', result.split('.')[0].split('_')[-1])
        date = int(input(f'Select [0-{len(working_results)}]'))
    else:
        date = 0

    df = pd.read_csv(join(folder, working_results[date]))
    df = df.fillna(0)
    df_1 = df.groupby(['dataset', '% labeled'])
    df_1 = df[['dataset', '% labeled']].copy().join(df_1.rank(axis=1))
    df_1.to_csv(join(folder, ranked, working_results[date]), index=False)
    print(f'Results in {join(folder, ranked, working_results[date])}')




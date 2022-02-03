#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    csvs_to_plots.py
# @Author:      Daniel Puente Ramírez
# @Time:        8/12/21 10:03
import re
from os import walk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    path = '../test_unlabeled/'
    dataset = 'titanic'
    files = next(walk(path), (None, None, []))[2]

    acc_df = []
    mse_df = []
    order_mse = []
    order_acc = []
    for file in files:
        if '.csv' in file:
            precision = float(re.findall(r"[-+]?\d*\.\d+|\d+", file)[0])
            file = path + file
            if 'mse' in file:
                order_mse.append(precision)
                mse_df.append(pd.read_csv(file))
            else:
                order_acc.append(precision)
                acc_df.append(pd.read_csv(file))
    order_mse = [(i, x) for i, x in enumerate(order_mse)]
    order_acc = [(i, x) for i, x in enumerate(order_acc)]
    order_mse.sort(key=lambda x: x[1])
    order_acc.sort(key=lambda x: x[1])

    datasets = np.array([x for x in mse_df[0]['dataset']])
    index = [idx for idx, elem in enumerate(datasets) if dataset in elem][0]

    acc_algorithms = np.array([[x for x in df.iloc[index][1:]] for df in
                               acc_df])
    acc_algorithms = np.array([acc_algorithms[x] for x, _ in order_acc])
    acc_algorithms = acc_algorithms.astype(float)

    mse_algorithms = np.array([[x for x in df.iloc[index][1:]] for df in
                               mse_df])
    mse_algorithms = np.array([mse_algorithms[x] for x, _ in order_mse])
    mse_algorithms = mse_algorithms.astype(float)

    precision = [x for _, x in order_mse]
    algorithms = ['ENN', 'CNN', 'RNN', 'ICF', 'MSS']
    for i in range(len(acc_algorithms)):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(f'ACC and MSE for {algorithms[i]}')
        ax1.plot(precision, acc_algorithms[:, i])
        ax1.set_title('ACC')
        ax2.plot(precision, mse_algorithms[:, i])
        ax2.set_title('MSE')
        file_name = path + algorithms[i] + f'_{dataset}' + '.png'
        plt.savefig(file_name)
        plt.show()


if __name__ == '__main__':
    main()

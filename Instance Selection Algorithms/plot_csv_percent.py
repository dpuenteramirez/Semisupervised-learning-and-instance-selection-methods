#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    plot_csv_percent.py
# @Author:      Daniel Puente Ramírez
# @Time:        9/12/21 12:35
from os import walk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    dataset = 'page-blocks'
    folder = './test_unlabeled/'

    files = next(walk(folder), (None, None, []))[2]
    
    for file in files:
        if '.csv' in file and ('KNN' in file or 'Tree' in file):
            name = folder + file
            if 'KNN' in file:
                if 'acc' in file:
                    knn_acc_df = pd.read_csv(name)
                else:
                    knn_mse_df = pd.read_csv(name)
            else:
                if 'acc' in file:
                    tree_acc_df = pd.read_csv(name)
                else:
                    tree_mse_df = pd.read_csv(name)

    datasets = np.array([x for x in knn_acc_df['dataset']])
    index = [idx for idx, elem in enumerate(datasets) if dataset in elem][0]

    precision = np.array(knn_acc_df.keys()[1:])
    precision = np.asfarray(precision)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(precision, list(knn_acc_df.iloc[index][1:]))
    axs[0, 1].plot(precision, list(knn_mse_df.iloc[index][1:]))
    axs[1, 0].plot(precision, list(tree_acc_df.iloc[index][1:]))
    axs[1, 1].plot(precision, list(tree_mse_df.iloc[index][1:]))

    axs[0, 0].set_title("ACC")
    axs[0, 1].set_title("MSE")
    axs[0, 0].set_ylabel('KNN')
    axs[1, 0].set_ylabel('Tree')

    axs[1, 0].set_xlabel("Precision")
    axs[1, 1].set_xlabel("Precision")

    plt.tight_layout()
    plt.savefig(folder+dataset)
    plt.show()


if __name__ == '__main__':
    main()

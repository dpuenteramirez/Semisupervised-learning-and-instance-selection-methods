#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    plot_csv_percent.py
# @Author:      Daniel Puente Ramírez
# @Time:        9/12/21 12:35
import argparse
from os import walk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def test_unlabeled(dataset):
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

    axs[1, 0].set_xlabel("So much per one labeled")
    axs[1, 1].set_xlabel("So much per one labeled")

    plt.tight_layout()
    plt.savefig(folder + dataset)
    plt.show()


def plot_self_tarining(dataset):
    df = pd.read_csv(
        '../semisupervised_algorithms/tests/accuracy_self_training.csv')
    datasets = df.keys()[1:]
    print(datasets)


def plot_unlabeled_to_one():
    df_knn_acc = pd.read_csv(
        './test_unlabeled/test_unlabeled_cross_validation_KNN_acc.csv')
    df_knn_mse = pd.read_csv(
        './test_unlabeled/test_unlabeled_cross_validation_KNN_mse.csv')
    df_tree_acc = pd.read_csv(
        './test_unlabeled/test_unlabeled_cross_validation_Tree_acc.csv')
    df_tree_mse = pd.read_csv(
        './test_unlabeled/test_unlabeled_cross_validation_Tree_mse.csv')

    knn_acc = df_knn_acc.to_numpy()
    knn_mse = df_knn_mse.to_numpy()
    tree_acc = df_tree_acc.to_numpy()
    tree_mse = df_tree_mse.to_numpy()
    datasets = [knn_acc, knn_mse, tree_acc, tree_mse]
    str_datasets = ['KNN_acc', 'KNN_mse', 'Tree_acc', 'Tree_mse']

    precision = np.array(df_knn_acc.keys()[1:])
    precision = np.asfarray(precision)

    for index, dataset in enumerate(datasets):

        fig, ax = plt.subplots(figsize=(10, 10))
        for d in dataset:
            plt.plot(precision, d[1:], label=d[0].split('_')[1])
        ax.set_xlabel("So much per one labeled")
        ax.set_ylabel('Correct prediction')
        ax.set_title(str_datasets[index])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'./test_unlabeled/all_in_one_{str_datasets[index]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', nargs=1)
    args = parser.parse_args()

    dataset = 'page-blocks' if args.d is None else str(args.d[0])
    # test_unlabeled(dataset)
    # plot_self_tarining(dataset)
    plot_unlabeled_to_one()

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    graficas.py
# @Author:      Daniel Puente Ramírez
# @Time:        19/11/21 08:15

import matplotlib.pyplot as plt


def grafica_2D(S):
    # https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_scatter.html
    x_index = 0
    y_index = 1
    formatter = plt.FuncFormatter(lambda i, *args: S.target_names[int(i)])
    plt.figure(figsize=(5, 4))
    plt.scatter(S.data[:, x_index], S.data[:, y_index], c=S.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(S.feature_names[x_index])
    plt.ylabel(S.feature_names[y_index])

    plt.tight_layout()
    plt.show()

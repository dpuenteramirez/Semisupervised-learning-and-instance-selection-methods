#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    custom_plots.py
# @Author:      Daniel Puente Ramírez
# @Time:        27/1/22 17:27
import matplotlib.pyplot as plt
import numpy as np


def plot_bar_line(name, metric, precision, data_df, save_path):
    percent_labeled = '% labeled'
    x_ticks = np.arange(len(precision))
    title = f'Dataset {name} - {metric} '
    if 'without' in save_path:
        title += 'without deletion'
    else:
        title += 'with deletion'
    ax = data_df.plot.bar(
        x=percent_labeled, y=['SVC', 'Before Filtering', 'After Filtering'],
        ylabel='Precision', xlabel=percent_labeled, figsize=(12, 6),
        title=title
    )
    ax2 = ax.twinx()

    data_df.plot(x=percent_labeled, y=['Samples Before Filtering',
                                       'Samples After Self Training',
                                       'Samples After Filtering'],
                 marker='o', color=['red', 'cyan', 'black'],
                 ylabel='Nº Samples', ax=ax2)
    plt.xticks(x_ticks, [str(x) for x in precision])

    ax.legend(['SVC', 'Before Filtering', 'After Filtering'], loc="upper left")
    ax2.legend(['Original dataset', 'After Self Training',
                'Samples ENN'],
               loc="lower right")
    plt.savefig(save_path)
    plt.show()

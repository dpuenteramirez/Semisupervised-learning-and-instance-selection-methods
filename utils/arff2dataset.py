#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    arff2dataset.py
# @Author:      Daniel Puente Ramírez
# @Time:        22/12/21 18:05
# @Version:     2.0

import arff
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch


def arff2sk_dataset(dataset_path):
    dataset = arff.load(open(dataset_path, 'r'))
    dat = np.array(dataset['data'])
    tt = np.array(dat[:, -1])
    dat = np.delete(dat, -1, 1)
    dat[dat == ''] = 0.0
    dat = dat.astype(float)

    try:
        tar_names = np.array(dataset['attributes'][-1][1]).astype(int)
        tar = tt.astype(int)
    except ValueError:
        tar_names = np.array([x for x in range(len(dataset['attributes'][-1][
                                                       1]))])
        relation = {}
        for index, target in enumerate(dataset['attributes'][-1][1]):
            relation[target] = index
        tar = np.array([relation[t] for t in tt])

    att_names = np.array([x[0] for x in dataset['attributes'][:-1]])
    dataset = Bunch(data=dat, target=tar, feature_names=att_names,
                    class_names=tar_names)

    return dataset


def arff_data(dataset_path, attr=False):
    file = open(dataset_path, 'r')
    data = []
    attrs = []
    start = False
    while True:
        next_line = file.readline()
        if not next_line:
            break
        if next_line[0] == '%':
            continue
        if '@attribute' in next_line.strip().lower():
            n = next_line.strip().split(' ')
            attrs.append(n[1] if '\t' not in n[1] else n[1].split(sep='\t')[0])

            continue
        if '@DATA' in next_line.strip().upper():
            start = True
            continue
        if start:
            line_data = next_line.strip().split(sep=',')
            data.append(np.array(line_data))
    file.close()
    data = np.array(data)
    labels = data[:, -1]
    data = np.delete(data, -1, 1)
    data = data.astype(float)
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)

    if not attr:
        return Bunch(data=data, target=labels)
    else:
        return Bunch(data=data, target=labels, attr=attrs)

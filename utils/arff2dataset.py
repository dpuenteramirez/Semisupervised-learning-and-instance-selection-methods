#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    arff2dataset.py
# @Author:      Daniel Puente Ramírez
# @Time:        22/12/21 18:05
# @Version:     1.12252021

import arff
import numpy as np
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

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    arff2dataset.py
# @Author:      Daniel Puente Ramírez
# @Time:        22/12/21 18:05

import arff
import numpy as np
from sklearn.utils import Bunch


def arff2sk_dataset(dataset_path):
    dataset = arff.load(open(dataset_path, 'r'))
    dat = np.array(dataset['data'])
    tar = np.array(dat[:, -1])
    dat = np.delete(dat, -1, 1)
    dat[dat == ''] = 0.0
    dat = dat.astype(float)
    tar_names = np.unique(tar)
    tar_names_list = list(tar_names)
    att_names = np.array([x[0] for x in dataset['attributes'][:-1]])
    tar = np.array([tar_names_list.index(x) for x in tar])
    dataset = Bunch(data=dat, target=tar, feature_names=att_names,
                    class_names=tar_names)

    return dataset

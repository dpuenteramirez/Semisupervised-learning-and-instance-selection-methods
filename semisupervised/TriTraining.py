#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    TriTraining.py
# @Author:      Daniel Puente Ramírez
# @Time:        27/12/21 10:25
# @Version:     4.0

from math import floor, ceil

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch

from .utils import split


def measure_error(classifier_j, classifier_k, labeled_data):
    pred_j = classifier_j.predict(labeled_data)
    pred_k = classifier_k.predict(labeled_data)
    same = len([0 for x, y in zip(pred_j, pred_k) if x == y])
    return (len(pred_j) - same) / same


class TriTraining:
    """
    Zhou, Z. H., & Li, M. (2005). Tri-training: Exploiting unlabeled data
    using three classifiers. IEEE Transactions on knowledge and Data
    Engineering, 17(11), 1529-1541.

    Parameters
    ----------
    random_state : int, default=None
        The random seed used to initialize the classifiers

    c1 : base_estimator, default=KNeighborsClassifier
        The first classifier to be used

    c1_params : dict, default=None
        Parameters for the first classifier

    c2 : base_estimator, default=DecisionTreeClassifier
        The second classifier to be used

    c2_params : dict, default=None
        Parameters for the second classifier

    c3 : base_estimator, default=RandomForestClassifier
        The third classifier to be used

    c3_params : dict, default=None
        Parameters for the third classifier

    """

    def __init__(self, random_state=None,
                 c1=None, c1_params=None,
                 c2=None, c2_params=None,
                 c3=None, c3_params=None):
        """Tri-Training."""

        classifiers = [c1, c2, c3]
        classifiers_params = [c1_params, c2_params, c3_params]
        default_classifiers = [KNeighborsClassifier, DecisionTreeClassifier,
                               RandomForestClassifier]
        configs = []
        for index, (c, cp) in enumerate(zip(classifiers, classifiers_params)):
            if c is not None:
                if cp is not None:
                    configs.append(c(**cp))
                else:
                    configs.append(c())
            else:
                configs.append(default_classifiers[index]())

        self.hj, self.hk, self.hi = configs

        self.random_state = random_state if random_state is not None else \
            np.random.randint(low=0, high=10e5, size=1)[0]

    def subsample(self, l_t, s):
        np.random.seed(self.random_state)
        rng = np.random.default_rng()
        data = np.array(l_t['data'])
        target = np.array(l_t['target'])
        samples_index = rng.choice(len(data), size=s, replace=False)
        samples = data[samples_index]
        targets = target[samples_index]
        return Bunch(data=samples, target=targets)

    def fit(self, samples, y):
        try:
            labeled, u, y = split(samples, y)
        except IndexError:
            raise ValueError('Dimensions do not match.')

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        train, _, test, _ = train_test_split(labeled, y, train_size=floor(
            len(labeled) / 3), stratify=y, random_state=self.random_state)
        h_j = self.hj.fit(train, test)
        ep_j = 0.5
        lp_j = 0
        train, _, test, _ = train_test_split(labeled, y, train_size=floor(
            len(labeled) / 3), stratify=y, random_state=self.random_state)
        h_k = self.hk.fit(train, test)
        ep_k = 0.5
        lp_k = 0
        train, _, test, _ = train_test_split(labeled, y, train_size=floor(
            len(labeled) / 3), stratify=y, random_state=self.random_state)
        h_i = self.hi.fit(train, test)
        ep_i = 0.5
        lp_i = 0

        while True:
            hash_i = h_i.__hash__()
            hash_j = h_j.__hash__()
            hash_k = h_k.__hash__()

            update_j = False
            l_j = Bunch(data=[], target=[])
            e_j = measure_error(h_j, h_k, labeled)

            if e_j < ep_j:
                for sample in u:
                    sample_s = sample.reshape(1, -1)
                    if h_j.predict(sample_s) == h_k.predict(sample_s):
                        pred = h_i.predict(sample_s)
                        prev_dat = list(l_j['data'])
                        prev_tar = list(l_j['target'])
                        prev_dat.append(sample)
                        l_j['data'] = np.array(prev_dat)
                        prev_tar.append(pred)
                        l_j['target'] = np.array(prev_tar)

                if lp_j == 0:
                    lp_j = floor(e_j / (ep_j - e_j) + 1)

                if lp_j < len(l_j['data']):
                    if e_j * len(l_j['data']) < ep_j * lp_j:
                        update_j = True
                    elif lp_j > e_j / (ep_j - e_j):
                        l_j = self.subsample(l_j, ceil(((ep_j * lp_j) / e_j)
                                                       - 1))
                        update_j = True

            update_k = False
            l_k = Bunch(data=np.array([]), target=np.array([]))
            e_k = measure_error(h_j, h_k, labeled)

            if e_k < ep_k:
                for sample in u:
                    sample_s = sample.reshape(1, -1)
                    if h_j.predict(sample_s) == h_k.predict(sample_s):
                        pred = h_i.predict(sample_s)
                        prev_dat = list(l_k['data'])
                        prev_tar = list(l_k['target'])
                        prev_dat.append(sample)
                        l_k['data'] = np.array(prev_dat)
                        prev_tar.append(pred)
                        l_k['target'] = np.array(prev_tar)

                if lp_k == 0:
                    lp_k = floor(e_k / (ep_k - e_k) + 1)

                if lp_k < len(l_k['data']):
                    if e_k * len(l_k['data']) < ep_k * lp_k:
                        update_k = True
                    elif lp_k > e_k / (ep_k - e_k):
                        l_k = self.subsample(l_k, ceil(((ep_k * lp_k) / e_k)
                                                       - 1))
                        update_k = True

            update_i = False
            l_i = Bunch(data=np.array([]), target=np.array([]))
            e_i = measure_error(h_j, h_k, labeled)

            if e_i < ep_i:
                for sample in u:
                    sample_s = sample.reshape(1, -1)
                    if h_j.predict(sample_s) == h_k.predict(sample_s):
                        pred = h_i.predict(sample_s)
                        prev_dat = list(l_i['data'])
                        prev_tar = list(l_i['target'])
                        prev_dat.append(sample)
                        l_i['data'] = np.array(prev_dat)
                        prev_tar.append(pred)
                        l_i['target'] = np.array(prev_tar)

                if lp_i == 0:
                    lp_i = floor(e_i / (ep_i - e_i) + 1)

                if lp_i < len(l_i['data']):
                    if e_i * len(l_i['data']) < ep_i * lp_i:
                        update_i = True
                    elif lp_i > e_i / (ep_i - e_i):
                        l_i = self.subsample(l_i, ceil(((ep_i * lp_i) / e_i)
                                                       - 1))
                        update_i = True

            if update_j:
                train = np.concatenate((labeled, l_j['data']), axis=0)
                test = np.concatenate((y, np.ravel(l_j['target'])),
                                      axis=0)
                h_j = self.hj.fit(train, test)
                ep_j = e_j
                lp_j = len(l_j)
            if update_k:
                train = np.concatenate((labeled, l_k['data']), axis=0)
                test = np.concatenate((y, np.ravel(l_k['target'])),
                                      axis=0)
                h_k = self.hk.fit(train, test)
                ep_k = e_k
                lp_k = len(l_k)
            if update_i:
                train = np.concatenate((labeled, l_i['data']), axis=0)
                test = np.concatenate((y, np.ravel(l_i['target'])),
                                      axis=0)
                h_i = self.hi.fit(train, test)
                ep_i = e_i
                lp_i = len(l_i)

            if h_i.__hash__() == hash_i and h_j.__hash__() == hash_j and \
                    h_k.__hash__() == hash_k:
                break

    def predict(self, samples):
        labels = []
        pred1 = self.hi.predict(samples)
        pred2 = self.hj.predict(samples)
        pred3 = self.hk.predict(samples)

        for p in zip(pred1, pred2, pred3):
            count = np.bincount(p)
            labels.append(np.where(count == np.amax(count))[0][0])

        return np.array(labels)

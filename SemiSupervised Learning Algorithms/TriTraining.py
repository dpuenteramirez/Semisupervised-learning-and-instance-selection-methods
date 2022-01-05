#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    Tri-Training.py
# @Author:      Daniel Puente Ramírez
# @Time:        27/12/21 10:25
# @Version:     2.0

from math import floor, ceil

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch


class TriTraining:
    def __init__(self, learn=None, random_state=None):
        if learn is None or learn not in [*range(3)]:
            raise ValueError(
                'The learn algorithm must be passed to the constructor.\n'
                '\t 1 for 3-NN \n\t 2 for Decision Tree \n\t 3 for Random '
                'Forest \n\t 4 for Gaussian NB'
            )
        if learn == 1:
            self.hj = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p=2)
            self.hk = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p=2)
            self.hi = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p=2)
        elif learn == 2:
            self.hj = DecisionTreeClassifier(random_state=random_state)
            self.hk = DecisionTreeClassifier(random_state=random_state)
            self.hi = DecisionTreeClassifier(random_state=random_state)
        elif learn == 3:
            self.hj = RandomForestClassifier(random_state=random_state)
            self.hk = RandomForestClassifier(random_state=random_state)
            self.hi = RandomForestClassifier(random_state=random_state)
        else:
            self.hj = GaussianNB()
            self.hk = GaussianNB()
            self.hi = GaussianNB()

        self.random_state = random_state if random_state is not None else \
            np.random.randint(low=0, high=10e5, size=1)[0]

    def measure_error(self, classifier_j, classifier_k, labeled_data):
        pred_j = classifier_j.predict(labeled_data)
        pred_k = classifier_k.predict(labeled_data)
        same = len([0 for x, y in zip(pred_j, pred_k) if x == y])
        return (len(pred_j) - same) / same

    def subsample(self, l_t, s):
        np.random.seed(self.random_state)
        rng = np.random.default_rng()
        data = np.array(l_t['data'])
        target = np.array(l_t['target'])
        samples_index = rng.choice(len(data), size=s, replace=False)
        samples = data[samples_index]
        targets = target[samples_index]
        return Bunch(data=samples, target=targets)

    def fit(self, L, U, y):
        if len(L) != len(y):
            raise ValueError(
                f'The dimension of the labeled data must be the same as the '
                f'number of labels given. {len(L)} != {len(y)}'
            )

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        train, _, test, _ = train_test_split(L, y, train_size=floor(
            len(L) / 3), stratify=y, random_state=self.random_state)
        h_j = self.hj.fit(train, test)
        ep_j = 0.5
        lp_j = 0
        train, _, test, _ = train_test_split(L, y, train_size=floor(
            len(L) / 3), stratify=y, random_state=self.random_state)
        h_k = self.hk.fit(train, test)
        ep_k = 0.5
        lp_k = 0
        train, _, test, _ = train_test_split(L, y, train_size=floor(
            len(L) / 3), stratify=y, random_state=self.random_state)
        h_i = self.hi.fit(train, test)
        ep_i = 0.5
        lp_i = 0

        while True:
            hash_i = h_i.__hash__()
            hash_j = h_j.__hash__()
            hash_k = h_k.__hash__()

            update_j = False
            L_j = Bunch(data=[], target=[])
            e_j = self.measure_error(h_j, h_k, L)

            if e_j < ep_j:
                for sample in U:
                    sample_s = sample.reshape(1, -1)
                    if h_j.predict(sample_s) == h_k.predict(sample_s):
                        pred = h_i.predict(sample_s)
                        prev_dat = list(L_j['data'])
                        prev_tar = list(L_j['target'])
                        prev_dat.append(sample)
                        L_j['data'] = np.array(prev_dat)
                        prev_tar.append(pred)
                        L_j['target'] = np.array(prev_tar)

                if lp_j == 0:
                    lp_j = floor(e_j / (ep_j - e_j) + 1)

                if lp_j < len(L_j['data']):
                    if e_j * len(L_j['data']) < ep_j * lp_j:
                        update_j = True
                    elif lp_j > e_j / (ep_j - e_j):
                        L_j = self.subsample(L_j, ceil(((ep_j * lp_j) / e_j)
                                                       - 1))
                        update_j = True

            update_k = False
            L_k = Bunch(data=np.array([]), target=np.array([]))
            e_k = self.measure_error(h_j, h_k, L)

            if e_k < ep_k:
                for sample in U:
                    sample_s = sample.reshape(1, -1)
                    if h_j.predict(sample_s) == h_k.predict(sample_s):
                        pred = h_i.predict(sample_s)
                        prev_dat = list(L_k['data'])
                        prev_tar = list(L_k['target'])
                        prev_dat.append(sample)
                        L_k['data'] = np.array(prev_dat)
                        prev_tar.append(pred)
                        L_k['target'] = np.array(prev_tar)

                if lp_k == 0:
                    lp_k = floor(e_k / (ep_k - e_k) + 1)

                if lp_k < len(L_k['data']):
                    if e_k * len(L_k['data']) < ep_k * lp_k:
                        update_k = True
                    elif lp_k > e_k / (ep_k - e_k):
                        L_k = self.subsample(L_k, ceil(((ep_k * lp_k) / e_k)
                                                       - 1))
                        update_k = True

            update_i = False
            L_i = Bunch(data=np.array([]), target=np.array([]))
            e_i = self.measure_error(h_j, h_k, L)

            if e_i < ep_i:
                for sample in U:
                    sample_s = sample.reshape(1, -1)
                    if h_j.predict(sample_s) == h_k.predict(sample_s):
                        pred = h_i.predict(sample_s)
                        prev_dat = list(L_i['data'])
                        prev_tar = list(L_i['target'])
                        prev_dat.append(sample)
                        L_i['data'] = np.array(prev_dat)
                        prev_tar.append(pred)
                        L_i['target'] = np.array(prev_tar)

                if lp_i == 0:
                    lp_i = floor(e_i / (ep_i - e_i) + 1)

                if lp_i < len(L_i['data']):
                    if e_i * len(L_i['data']) < ep_i * lp_i:
                        update_i = True
                    elif lp_i > e_i / (ep_i - e_i):
                        L_i = self.subsample(L_i, ceil(((ep_i * lp_i) / e_i)
                                                       - 1))
                        update_i = True

            if update_j:
                train = np.concatenate((L, L_j['data']), axis=0)
                test = np.concatenate((y, np.ravel(L_j['target'])),
                                      axis=0)
                h_j = self.hj.fit(train, test)
                ep_j = e_j
                lp_j = len(L_j)
            if update_k:
                train = np.concatenate((L, L_k['data']), axis=0)
                test = np.concatenate((y, np.ravel(L_k['target'])),
                                      axis=0)
                h_k = self.hk.fit(train, test)
                ep_k = e_k
                lp_k = len(L_k)
            if update_i:
                train = np.concatenate((L, L_i['data']), axis=0)
                test = np.concatenate((y, np.ravel(L_i['target'])),
                                      axis=0)
                h_i = self.hi.fit(train, test)
                ep_i = e_i
                lp_i = len(L_i)

            if h_i.__hash__() == hash_i and h_j.__hash__() == hash_j and \
                    h_k.__hash__() == hash_k:
                break

    def predict(self, X):
        labels = []
        pred1 = self.hi.predict(X)
        pred2 = self.hj.predict(X)
        pred3 = self.hk.predict(X)

        for p in zip(pred1, pred2, pred3):
            count = np.bincount(p)
            labels.append(np.where(count == np.amax(count))[0][0])

        return np.array(labels)

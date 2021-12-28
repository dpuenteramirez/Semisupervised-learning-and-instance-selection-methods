#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    Tri-Training.py
# @Author:      Daniel Puente Ramírez
# @Time:        27/12/21 10:25
import time
from math import floor, ceil

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch


def measure_error(classifier_j, classifier_k, labeled_data):
    pred_j = classifier_j.predict(labeled_data)
    pred_k = classifier_k.predict(labeled_data)
    same = len([0 for x, y in zip(pred_j, pred_k) if x == y])
    return (len(pred_j) - same) / same


def subsample(l_t, s):
    np.random.seed(42)
    rng = np.random.default_rng()
    data = np.array(l_t['data'])
    target = np.array(l_t['target'])
    samples_index = rng.choice(len(data), size=s, replace=False)
    samples = data[samples_index]
    targets = target[samples_index]
    return Bunch(data=samples, target=targets)


def tri_training(L, U):
    start = time.time()
    # Initialize values
    # BootStrap
    # Classifier
    # e' and l
    labeled = L['data']
    target = L['target']
    unlabeled = U['data']

    train, _, test, _ = train_test_split(labeled, target, train_size=floor(
        len(labeled) / 3), stratify=target, random_state=42)
    h_j = KNeighborsClassifier(n_neighbors=3, n_jobs=-1).fit(train, test)
    ep_j = 0.5
    lp_j = 0
    train, _, test, _ = train_test_split(labeled, target, train_size=floor(
        len(labeled) / 3), stratify=target, random_state=420)
    h_k = DecisionTreeClassifier(random_state=420).fit(train, test)
    ep_k = 0.5
    lp_k = 0
    train, _, test, _ = train_test_split(labeled, target, train_size=floor(
        len(labeled) / 3), stratify=target, random_state=4200)
    h_i = RandomForestClassifier(random_state=4200).fit(train, test)
    ep_i = 0.5
    lp_i = 0

    while True:
        print(1)
        hash_i = h_i.__hash__()
        hash_j = h_j.__hash__()
        hash_k = h_k.__hash__()

        # Classifier h_j
        update_j = False
        L_j = Bunch(data=[], target=[])
        e_j = measure_error(h_j, h_k, L['data'])

        if e_j < ep_j:
            for sample in unlabeled:
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
                    L_j = subsample(L_j, ceil(((ep_j * lp_j) / e_j) - 1))
                    update_j = True

        # Classifier h_k
        update_k = False
        L_k = Bunch(data=np.array([]), target=np.array([]))
        e_k = measure_error(h_j, h_k, L['data'])

        if e_k < ep_k:
            for sample in unlabeled:
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
                    L_k = subsample(L_k, ceil(((ep_k * lp_k) / e_k) - 1))
                    update_k = True

        # Classifier h_i
        update_i = False
        L_i = Bunch(data=np.array([]), target=np.array([]))
        e_i = measure_error(h_j, h_k, L['data'])

        if e_i < ep_i:
            for sample in unlabeled:
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
                    L_i = subsample(L_i, ceil(((ep_i * lp_i) / e_i) - 1))
                    update_i = True

        # for i \in {1..3} do
        if update_j:
            train = np.concatenate((L['data'], L_j['data']), axis=0)
            test = np.concatenate((L['target'], np.ravel(L_j['target'])),
                                  axis=0)
            h_j = KNeighborsClassifier(n_neighbors=3, n_jobs=-1).fit(train,
                                                                     test)
            ep_j = e_j
            lp_j = len(L_j)
        if update_k:
            train = np.concatenate((L['data'], L_k['data']), axis=0)
            test = np.concatenate((L['target'], np.ravel(L_k['target'])),
                                  axis=0)
            h_k = DecisionTreeClassifier(random_state=420).fit(train, test)
            ep_k = e_k
            lp_k = len(L_k)
        if update_i:
            train = np.concatenate((L['data'], L_i['data']), axis=0)
            test = np.concatenate((L['target'], np.ravel(L_i['target'])),
                                  axis=0)
            h_i = RandomForestClassifier(random_state=4200).fit(train, test)
            ep_i = e_i
            lp_i = len(L_i)

        if h_i.__hash__() == hash_i and h_j.__hash__() == hash_j and \
                h_k.__hash__() == hash_k:
            end = time.time()
            print(end - start)
            break

    pred_j = h_j.predict(unlabeled)
    pred_k = h_k.predict(unlabeled)
    pred_i = h_i.predict(unlabeled)

    new_labels = []
    for w, x, y, z in zip(unlabeled, pred_j, pred_k, pred_i):
        if x == y and x == z:
            new_labels.append(x)
        elif x == y or x == z:
            new_labels.append(x)
        elif y == z:
            new_labels.append(y)
        else:
            new_labels.append(z)

    return Bunch(data=unlabeled, target=new_labels)


if __name__ == '__main__':
    dataset = load_iris()
    X = dataset['data']
    y = dataset['target']
    # Odd order of attributes...
    X_train, y_train, X_test, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42)
    pred = tri_training(Bunch(data=X_train, target=X_test), Bunch(data=y_train))
    print(classification_report(y_true=y_test, y_pred=pred['target']))

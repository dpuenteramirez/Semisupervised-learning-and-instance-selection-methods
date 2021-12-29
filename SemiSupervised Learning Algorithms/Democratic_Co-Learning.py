#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    Democratic_Co-Learning.py
# @Author:      Daniel Puente Ramírez
# @Time:        29/12/21 15:39
import copy
from math import sqrt

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch


def check_bounds(wi):
    if wi[0] < 0:
        wi[0] = 0
    if wi[1] > 1:
        wi[1] = 1
    return wi


def democratic(L, U):
    """
    Democratic Co-Learning implementation
    Using 3 classifiers.
    :param L:
    :param U:
    :return:
    """

    const = 1.96  # 95%
    n_classifiers = 3
    unlabeled_data = U['data']

    # Copies of labeled data for each classifier
    L1_data = copy.deepcopy(list(L['data']))
    L1_labels = copy.deepcopy(list(L['target']))
    e_1 = 0
    L2_data = copy.deepcopy(list(L['data']))
    L2_labels = copy.deepcopy(list(L['target']))
    e_2 = 0
    L3_data = copy.deepcopy(list(L['data']))
    L3_labels = copy.deepcopy(list(L['target']))
    e_3 = 0

    while True:
        L1_hash = L1_data.__hash__
        L2_hash = L2_data.__hash__
        L3_hash = L3_data.__hash__

        # Train the n classifiers with all the labeled data
        h_1 = MultinomialNB().fit(L1_data, L1_labels)
        h_2 = KNeighborsClassifier(n_neighbors=3, n_jobs=-1).fit(L2_data,
                                                                 L2_labels)
        h_3 = DecisionTreeClassifier(random_state=420).fit(L3_data, L3_labels)

        new_labels = []
        probas = []
        for sample in unlabeled_data:
            sample_s = [sample]
            c1_t = h_1.predict_proba(sample_s)[0]
            c1_p, c_1 = np.amax(c1_t), np.where(c1_t == np.amax(c1_t))[0][0]
            c2_t = h_2.predict_proba(sample_s)[0]
            c2_p, c_2 = np.amax(c2_t), np.where(c2_t == np.amax(c2_t))[0][0]
            c3_t = h_3.predict_proba(sample_s)[0]
            c3_p, c_3 = np.amax(c3_t), np.where(c3_t == np.amax(c3_t))[0][0]
            proba = np.array([c1_p, c2_p, c3_p])
            labels = np.array([c_1, c_2, c_3])
            new_labels.append(labels[np.where(proba == np.amax(proba))])
            probas.append(np.array([c1_t, c2_t, c3_t]))

        # Choose which exs to propose for labeling
        L1_prime_data = []
        L1_prime_label = []
        L2_prime_data = []
        L2_prime_label = []
        L3_prime_data = []
        L3_prime_label = []

        # Classifier 1
        pred = h_1.predict(L['data'])
        error = len([0 for p, tar in zip(pred, L['target']) if p !=
                     tar]) / len(pred)
        w_1 = [error - const * sqrt((error * (1 - error)) / len(L['data'])),
               error + const * sqrt((error * (1 - error)) / len(L['data']))]
        w_1 = sum(check_bounds(w_1)) / 2

        # For each classifier
        for index, proba in enumerate(probas):
            c_k = new_labels[index][0]
            sum_izq = [0 for _ in range(len(probas[0]))]
            sum_der = 0
            for index2, classifier in enumerate(proba):
                best = np.where(classifier == np.amax(classifier))[0][0]
                if best == c_k:
                    sum_der += w_1
                else:
                    sum_izq[index2] += w_1

            if sum_der > max(sum_izq):
                if np.where(proba[0] == np.amax(proba[0]))[0][0] != c_k:
                    L1_prime_data.append(unlabeled_data[index])
                    L1_prime_label.append(c_k)

        # Classifier 2
        pred = h_2.predict(L['data'])
        error = len([0 for p, tar in zip(pred, L['target']) if p !=
                     tar]) / len(pred)
        w_2 = [error - const * sqrt((error * (1 - error)) / len(L['data'])),
               error + const * sqrt((error * (1 - error)) / len(L['data']))]
        w_2 = sum(check_bounds(w_2)) / 2

        # For each classifier
        for index, proba in enumerate(probas):
            c_k = new_labels[index][0]
            sum_izq = [0 for _ in range(len(probas[0]))]
            sum_der = 0
            for index2, classifier in enumerate(proba):
                best = np.where(classifier == np.amax(classifier))[0][0]
                if best == c_k:
                    sum_der += w_1
                else:
                    sum_izq[index2] += w_1

            if sum_der > max(sum_izq):
                if np.where(proba[1] == np.amax(proba[1]))[0][0] != c_k:
                    L2_prime_data.append(unlabeled_data[index])
                    L2_prime_label.append(c_k)

        # Classifier 3
        pred = h_3.predict(L['data'])
        error = len([0 for p, tar in zip(pred, L['target']) if p !=
                     tar]) / len(pred)
        w_3 = [error - const * sqrt((error * (1 - error)) / len(L['data'])),
               error + const * sqrt((error * (1 - error)) / len(L['data']))]
        w_3 = sum(check_bounds(w_3)) / 2

        # For each classifier
        for index, proba in enumerate(probas):
            c_k = new_labels[index][0]
            sum_izq = [0 for _ in range(len(probas[0]))]
            sum_der = 0
            for index2, classifier in enumerate(proba):
                best = np.where(classifier == np.amax(classifier))[0][0]
                if best == c_k:
                    sum_der += w_1
                else:
                    sum_izq[index2] += w_1

            if sum_der > max(sum_izq):
                if np.where(proba[2] == np.amax(proba[2]))[0][0] != c_k:
                    L3_prime_data.append(unlabeled_data[index])
                    L3_prime_label.append(c_k)

        # Estimate if adding L'i to Li improves accuracy
        # For each classifier
        # Classifier 1
        try:
            pred = h_1.predict(L1_prime_data)
        except ValueError:
            try:
                pred = h_1.predict([L1_prime_data])
                error = len([0 for p, tar in zip(pred, L1_prime_label) if p !=
                             tar]) / len(pred)
                ci_1 = [error - const * sqrt((error * (1 - error)) / len(pred)),
                        error + const * sqrt((error * (1 - error)) / len(pred))]
                ci_1 = check_bounds(ci_1)
                q_1 = len(pred) * pow((1 - 2 * (e_1 / len(pred))), 2)
                e_prime_1 = (1 - (ci_1[0] * len(pred)) / len(pred)) * len(pred)
                q_prime_1 = (len(L1_data) + len(pred)) * pow(
                    1 - (2 * (e_1 + e_prime_1)) / (len(
                        L1_data) + len(pred)), 2)

                if q_prime_1 > q_1:
                    L1_data.append(L1_prime_data)
                    L1_labels.append(L1_prime_label)
                    e_1 += e_prime_1
            except ValueError as e:
                print(repr(e))

        # Classifier 2
        try:
            pred = h_2.predict(L2_prime_data)
        except ValueError:
            try:
                pred = h_2.predict([L2_prime_data])
                error = len([0 for p, tar in zip(pred, L2_prime_label) if p !=
                             tar]) / len(pred)
                ci_2 = [error - const * sqrt((error * (1 - error)) / len(pred)),
                        error + const * sqrt((error * (1 - error)) / len(pred))]
                ci_2 = check_bounds(ci_2)
                q_2 = len(pred) * pow((1 - 2 * (e_2 / len(pred))), 2)
                e_prime_2 = (1 - (ci_2[0] * len(pred)) / len(pred)) * len(pred)
                q_prime_2 = (len(L2_data) + len(pred)) * pow(
                    1 - (2 * (e_2 + e_prime_2)) / (len(
                        L2_data) + len(pred)), 2)

                if q_prime_2 > q_2:
                    L2_data.append(L2_prime_data)
                    L2_labels.append(L2_prime_label)
                    e_2 += e_prime_2
            except ValueError as e:
                print(repr(e))

        # Classifier 3
        try:
            pred = h_3.predict(L3_prime_data)
        except ValueError:
            try:
                pred = h_3.predict([L3_prime_data])
                error = len([0 for p, tar in zip(pred, L3_prime_label) if p !=
                             tar]) / len(pred)
                ci_3 = [error - const * sqrt((error * (1 - error)) / len(pred)),
                        error + const * sqrt((error * (1 - error)) / len(pred))]
                ci_3 = check_bounds(ci_3)
                q_3 = len(pred) * pow((1 - 2 * (e_3 / len(pred))), 2)
                e_prime_3 = (1 - (ci_3[0] * len(pred)) / len(pred)) * len(pred)
                q_prime_3 = (len(L3_data) + len(pred)) * pow(
                    1 - (2 * (e_3 + e_prime_3)) / (len(
                        L3_data) + len(pred)), 2)

                if q_prime_3 > q_3:
                    L3_data.append(L3_prime_data)
                    L3_labels.append(L3_prime_label)
                    e_3 += e_prime_3
            except ValueError as e:
                print(repr(e))

        if L1_data.__hash__ == L1_hash and L2_data.__hash__ == L2_hash and \
                L3_data.__hash__ == L3_hash:
            break

    # Combine
    pred = h_1.predict(L['data'])
    error = len([0 for p, tar in zip(pred, L['target']) if p !=
                 tar]) / len(pred)
    w_1 = [error - const * sqrt((error * (1 - error)) / len(L['data'])),
           error + const * sqrt((error * (1 - error)) / len(L['data']))]
    w_1 = sum(check_bounds(w_1)) / 2
    pred = h_2.predict(L['data'])
    error = len([0 for p, tar in zip(pred, L['target']) if p !=
                 tar]) / len(pred)
    w_2 = [error - const * sqrt((error * (1 - error)) / len(L['data'])),
           error + const * sqrt((error * (1 - error)) / len(L['data']))]
    w_2 = sum(check_bounds(w_2)) / 2
    pred = h_3.predict(L['data'])
    error = len([0 for p, tar in zip(pred, L['target']) if p !=
                 tar]) / len(pred)
    w_3 = [error - const * sqrt((error * (1 - error)) / len(L['data'])),
           error + const * sqrt((error * (1 - error)) / len(L['data']))]
    w_3 = sum(check_bounds(w_3)) / 2

    labeled_data = copy.deepcopy(list(L['data']))
    unlabeled_data = copy.deepcopy((list(U['data'])))
    all_instances = labeled_data + unlabeled_data

    gj = [0 for _ in range(len(probas[0]))]
    gj_h = [[0 for _ in range(len(probas[0]))] for _ in range(n_classifiers)]
    for sample in all_instances:
        sample_s = [sample]
        if w_1 > 0.5:
            p = h_1.predict(sample_s)
            gj[p] += 1
            gj_h[0][p] += 1
        if w_2 > 0.5:
            p = h_2.predict(sample_s)
            gj[p] += 1
            gj_h[1][p] += 1
        if w_3 > 0.5:
            p = h_3.predict(sample_s)
            gj[p] += 1
            gj_h[2][p] += 1

    confidence = [0 for _ in range(len(probas[0]))]
    for j in range(len(gj)):
        izq = (gj[j] + 0.5) / (gj[j] + 1)
        div = True if gj[j] != 0 else False
        if div:
            der = [(gj_h[0][j] * w_1) / gj[j], (gj_h[1][j] * w_2) / gj[j],
                   (gj_h[2][j] * w_3) / gj[j]]
        else:
            der = [1 for _ in range(n_classifiers)]

        confidence[j] = sum([izq * d for d in der]) / len(der)

    classifiers = [h_1, h_2, h_3]
    confidence = np.array(confidence)
    classifier = np.where(confidence == np.amax(confidence))[0][0]
    return classifiers[classifier].predict(unlabeled_data)


if __name__ == '__main__':
    dataset = load_iris()
    X = dataset['data']
    y = dataset['target']

    # Odd order of attributes...
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y)
    pred_y = democratic(Bunch(data=X_train, target=y_train), Bunch(data=X_test))

    print(classification_report(y_true=y_test, y_pred=pred_y))

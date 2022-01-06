#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    DemocraticCoLearning.py
# @Author:      Daniel Puente Ramírez
# @Time:        29/12/21 15:39
# @Version:     2.0

import copy
from math import sqrt

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def check_bounds(wi):
    """Check upper and lower bounds. The left minimum value can be 0, and the
    right minimum value can be 1.

    :param wi: lower and upper mean confidence
    :return: wi fixed
    """
    if wi[0] < 0:
        wi[0] = 0
    if wi[1] > 1:
        wi[1] = 1
    return wi


class DemocraticCoLearning:
    """Democratic Co-Learning Implementation. Based on:
        Zhou, Y., & Goldman, S. (2004, November). Democratic co-learning.
        In 16th IEEE International Conference on Tools with Artificial
        Intelligence (pp. 594-602). IEEE.
    """

    def __init__(self, random_state=None):
        self.const = 1.96  # 95%
        self.random_state = random_state if random_state is not None else \
            np.random.randint(low=0, high=10e5, size=1)[0]
        self.n_classifiers = 3
        self.n_attributes = 0
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0

        self.h1 = MultinomialNB()
        self.h2 = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p=2)
        self.h3 = DecisionTreeClassifier(random_state=self.random_state)

    def fit(self, L, U, y):

        unlabeled_data = U
        self.n_attributes = len(L[0])

        L1_data = copy.deepcopy(list(L))
        L1_labels = copy.deepcopy(list(y))
        e_1 = 0
        L2_data = copy.deepcopy(list(L))
        L2_labels = copy.deepcopy(list(y))
        e_2 = 0
        L3_data = copy.deepcopy(list(L))
        L3_labels = copy.deepcopy(list(y))
        e_3 = 0

        while True:
            L1_hash = L1_data.__hash__
            L2_hash = L2_data.__hash__
            L3_hash = L3_data.__hash__

            self.h1.fit(L1_data, L1_labels)
            self.h2.fit(L2_data, L2_labels)
            self.h3.fit(L3_data, L3_labels)

            new_labels = []
            probas = []
            for sample in unlabeled_data:
                sample_s = [sample]
                c1_t = self.h1.predict_proba(sample_s)[0]
                c1_p, c_1 = np.amax(c1_t), np.where(c1_t == np.amax(c1_t))[0][0]
                c2_t = self.h2.predict_proba(sample_s)[0]
                c2_p, c_2 = np.amax(c2_t), np.where(c2_t == np.amax(c2_t))[0][0]
                c3_t = self.h3.predict_proba(sample_s)[0]
                c3_p, c_3 = np.amax(c3_t), np.where(c3_t == np.amax(c3_t))[0][0]
                proba = np.array([c1_p, c2_p, c3_p])
                labels = np.array([c_1, c_2, c_3])
                new_labels.append(labels[np.where(proba == np.amax(proba))])
                probas.append(np.array([c1_t, c2_t, c3_t]))

            L1_prime_data = []
            L1_prime_label = []
            L2_prime_data = []
            L2_prime_label = []
            L3_prime_data = []
            L3_prime_label = []

            pred = self.h1.predict(L)
            error = len([0 for p, tar in zip(pred, y) if p !=
                         tar]) / len(pred)
            w1 = [error - self.const * sqrt((error * (1 - error)) / len(L)),
                  error + self.const * sqrt((error * (1 - error)) / len(L))]
            w1 = sum(check_bounds(w1)) / 2

            for index, proba in enumerate(probas):
                c_k = new_labels[index][0]
                sum_izq = [0 for _ in range(len(probas[0]))]
                sum_der = 0
                for index2, classifier in enumerate(proba):
                    best = np.where(classifier == np.amax(classifier))[0][0]
                    if best == c_k:
                        sum_der += w1
                    else:
                        sum_izq[index2] += w1

                if sum_der > max(sum_izq) and np.where(proba[0] == np.amax(
                        proba[0]))[0][0] != c_k:
                    L1_prime_data.append(unlabeled_data[index])
                    L1_prime_label.append(c_k)

            pred = self.h2.predict(L)
            error = len([0 for p, tar in zip(pred, y) if p !=
                         tar]) / len(pred)
            w2 = [error - self.const * sqrt((error * (1 - error)) / len(L)),
                  error + self.const * sqrt((error * (1 - error)) / len(L))]
            w2 = sum(check_bounds(w2)) / 2

            for index, proba in enumerate(probas):
                c_k = new_labels[index][0]
                sum_izq = [0 for _ in range(len(probas[0]))]
                sum_der = 0
                for index2, classifier in enumerate(proba):
                    best = np.where(classifier == np.amax(classifier))[0][0]
                    if best == c_k:
                        sum_der += w2
                    else:
                        sum_izq[index2] += w2

                if sum_der > max(sum_izq) and np.where(proba[1] == np.amax(
                        proba[1]))[0][0] != c_k:
                    L2_prime_data.append(unlabeled_data[index])
                    L2_prime_label.append(c_k)

            pred = self.h3.predict(L)
            error = len([0 for p, tar in zip(pred, y) if p !=
                         tar]) / len(pred)
            w3 = [error - self.const * sqrt((error * (1 - error)) / len(L)),
                  error + self.const * sqrt((error * (1 - error)) / len(L))]
            w3 = sum(check_bounds(w3)) / 2

            for index, proba in enumerate(probas):
                c_k = new_labels[index][0]
                sum_izq = [0 for _ in range(len(probas[0]))]
                sum_der = 0
                for index2, classifier in enumerate(proba):
                    best = np.where(classifier == np.amax(classifier))[0][0]
                    if best == c_k:
                        sum_der += w3
                    else:
                        sum_izq[index2] += w3

                if sum_der > max(sum_izq) and np.where(proba[2] == np.amax(
                        proba[2]))[0][0] != c_k:
                    L3_prime_data.append(unlabeled_data[index])
                    L3_prime_label.append(c_k)

            try:
                pred = self.h1.predict(L1_prime_data)
            except ValueError:
                try:
                    pred = self.h1.predict([L1_prime_data])
                except ValueError as e:
                    print(repr(e))
            error = len(
                [0 for p, tar in zip(pred, L1_prime_label) if p !=
                 tar]) / len(pred)
            ci_1 = [
                error - self.const * sqrt((error * (1 - error)) / len(pred)),
                error + self.const * sqrt((error * (1 - error)) / len(pred))]
            ci_1 = check_bounds(ci_1)
            q_1 = len(pred) * pow((1 - 2 * (e_1 / len(pred))), 2)
            e_prime_1 = (1 - (ci_1[0] * len(pred)) / len(pred)) * len(pred)
            q_prime_1 = (len(L1_data) + len(pred)) * pow(
                1 - (2 * (e_1 + e_prime_1)) / (len(L1_data) + len(pred)), 2)

            if q_prime_1 > q_1:
                L1_data.append(L1_prime_data)
                L1_labels.append(L1_prime_label)
                e_1 += e_prime_1

            try:
                pred = self.h2.predict(L2_prime_data)
            except ValueError:
                try:
                    pred = self.h2.predict([L2_prime_data])
                except ValueError as e:
                    print(repr(e))
            error = len(
                [0 for p, tar in zip(pred, L2_prime_label) if p !=
                 tar]) / len(pred)
            ci_2 = [
                error - self.const * sqrt((error * (1 - error)) / len(pred)),
                error + self.const * sqrt((error * (1 - error)) / len(pred))]
            ci_2 = check_bounds(ci_2)
            q_2 = len(pred) * pow((1 - 2 * (e_2 / len(pred))), 2)
            e_prime_2 = (1 - (ci_2[0] * len(pred)) / len(pred)) * len(pred)
            q_prime_2 = (len(L2_data) + len(pred)) * pow(
                1 - (2 * (e_2 + e_prime_2)) / (len(L2_data) + len(pred)), 2)

            if q_prime_2 > q_2:
                L2_data.append(L2_prime_data)
                L2_labels.append(L2_prime_label)
                e_2 += e_prime_2

            try:
                pred = self.h3.predict(L3_prime_data)
            except ValueError:
                try:
                    pred = self.h3.predict([L3_prime_data])
                except ValueError as e:
                    print(repr(e))
            error = len(
                [0 for p, tar in zip(pred, L3_prime_label) if p !=
                 tar]) / len(pred)
            ci_3 = [
                error - self.const * sqrt((error * (1 - error)) / len(pred)),
                error + self.const * sqrt((error * (1 - error)) / len(pred))]
            ci_3 = check_bounds(ci_3)
            q_3 = len(pred) * pow((1 - 2 * (e_3 / len(pred))), 2)
            e_prime_3 = (1 - (ci_3[0] * len(pred)) / len(pred)) * len(pred)
            q_prime_3 = (len(L3_data) + len(pred)) * pow(
                1 - (2 * (e_3 + e_prime_3)) / (len(L3_data) + len(pred)), 2)

            if q_prime_3 > q_3:
                L3_data.append(L3_prime_data)
                L3_labels.append(L3_prime_label)
                e_3 += e_prime_3

            if L1_data.__hash__ == L1_hash and L2_data.__hash__ == L2_hash and \
                    L3_data.__hash__ == L3_hash:
                break

        pred = self.h1.predict(L)
        error = len([0 for p, tar in zip(pred, y) if p != tar]) / len(pred)
        w1 = [error - self.const * sqrt((error * (1 - error)) / len(L)),
              error + self.const * sqrt((error * (1 - error)) / len(L))]
        self.w1 = sum(check_bounds(w1)) / 2
        pred = self.h2.predict(L)
        error = len([0 for p, tar in zip(pred, y) if p != tar]) / len(pred)
        w2 = [error - self.const * sqrt((error * (1 - error)) / len(L)),
              error + self.const * sqrt((error * (1 - error)) / len(L))]
        self.w2 = sum(check_bounds(w2)) / 2
        pred = self.h3.predict(L)
        error = len([0 for p, tar in zip(pred, y) if p != tar]) / len(pred)
        w3 = [error - self.const * sqrt((error * (1 - error)) / len(L)),
              error + self.const * sqrt((error * (1 - error)) / len(L))]
        self.w3 = sum(check_bounds(w3)) / 2

    def predict(self, X):
        all_instances = X

        gj = [0 for _ in range(self.n_attributes)]
        gj_h = [[0 for _ in range(self.n_attributes)] for _ in
                range(self.n_classifiers)]
        for sample in all_instances:
            sample_s = [sample]
            if self.w1 > 0.5:
                p = self.h1.predict(sample_s)
                gj[p] += 1
                gj_h[0][p] += 1
            if self.w2 > 0.5:
                p = self.h2.predict(sample_s)
                gj[p] += 1
                gj_h[1][p] += 1
            if self.w3 > 0.5:
                p = self.h3.predict(sample_s)
                gj[p] += 1
                gj_h[2][p] += 1

        confidence = [0 for _ in range(self.n_attributes)]
        for j in range(len(gj)):
            izq = (gj[j] + 0.5) / (gj[j] + 1)
            div = True if gj[j] != 0 else False
            if div:
                der = [(gj_h[0][j] * self.w1) / gj[j],
                       (gj_h[1][j] * self.w2) / gj[j],
                       (gj_h[2][j] * self.w3) / gj[j]]
            else:
                der = [1 for _ in range(self.n_classifiers)]

            confidence[j] = sum([izq * d for d in der]) / len(der)

        classifiers = [self.h1, self.h2, self.h3]
        confidence = np.array(confidence)
        classifier = np.where(confidence == np.amax(confidence))[0][0]
        return classifiers[classifier].predict(X)

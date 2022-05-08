#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    DemocraticCoLearning.py
# @Author:      Daniel Puente Ramírez
# @Time:        29/12/21 15:39
# @Version:     5.0

import copy
import warnings
from math import sqrt

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from .utils import split


class DemocraticCoLearning:
    """
    Democratic Co-Learning Implementation. Based on:
    Zhou, Y., & Goldman, S. (2004, November). Democratic co-learning.
    In 16th IEEE International Conference on Tools with Artificial
    Intelligence (pp. 594-602). IEEE.

    Parameters
    ----------
    random_state : int, default=None
        The random seed used to initialize the classifiers

    c1 : base_estimator, default=MultinomialNB
        The first classifier to be used

    c1_params : dict, default=None
        Parameters for the first classifier

    c2 : base_estimator, default=KNeighborsClassifier
        The second classifier to be used

    c2_params : dict, default=None
        Parameters for the second classifier

    c3 : base_estimator, default=DecisionTreeClassifier
        The third classifier to be used

    c3_params : dict, default=None
        Parameters for the third classifier

    """

    def __init__(
        self,
        random_state=None,
        c1=None,
        c1_params=None,
        c2=None,
        c2_params=None,
        c3=None,
        c3_params=None,
    ):
        """Democratic Co-Learning."""
        self.const = 1.96  # 95%
        self.random_state = (
            random_state
            if random_state is not None
            else np.random.randint(low=0, high=10e5, size=1)[0]
        )
        self.n_classifiers = 3
        self.n_attributes = 0
        self.n_labels = 0
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0

        classifiers = [c1, c2, c3]
        classifiers_params = [c1_params, c2_params, c3_params]
        default_classifiers = [
            MultinomialNB,
            KNeighborsClassifier,
            DecisionTreeClassifier,
        ]
        configs = []
        for index, (c, cp) in enumerate(zip(classifiers, classifiers_params)):
            if c is not None:
                if cp is not None:
                    configs.append(c(**cp))
                else:
                    configs.append(c())
            else:
                configs.append(default_classifiers[index]())

        try:
            self.h1, self.h2, self.h3 = configs
        except ValueError:
            raise AttributeError(
                "Classifiers and/or params were not correctly passed."
            )

    def fit(self, samples, y):
        """
        The function takes in a set of labeled and unlabeled data, and uses the
        labeled data to train three classifiers. Then, it uses the three
        classifiers to predict the labels of the unlabeled data. If the
        prediction is correct, the data is not added to the training set. If
        the prediction is incorrect, the data is added to the training set.
        The process is repeated until the training set stops changing

        :param samples: the training data
        :param y: the labels of the samples
        """
        try:
            labeled, u, y = split(samples, y)
        except IndexError:
            raise ValueError("Dimensions do not match.")

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        self.n_labels = max(np.unique(y)) + 1

        unlabeled_data = u
        self.n_attributes = len(labeled[0])

        l1_data = copy.deepcopy(list(labeled))
        l1_labels = copy.deepcopy(list(y))
        e_1 = 0
        l2_data = copy.deepcopy(list(labeled))
        l2_labels = copy.deepcopy(list(y))
        e_2 = 0
        l3_data = copy.deepcopy(list(labeled))
        l3_labels = copy.deepcopy(list(y))
        e_3 = 0

        while True:
            l1_hash = l1_data.__hash__
            l2_hash = l2_data.__hash__
            l3_hash = l3_data.__hash__

            self.h1.fit(l1_data, l1_labels)
            self.h2.fit(l2_data, l2_labels)
            self.h3.fit(l3_data, l3_labels)

            new_labels = []
            probas = []
            for sample in unlabeled_data:
                sample_s = [sample]
                c1_t = self.h1.predict_proba(sample_s)[0]
                c1_p, c_1 = np.amax(c1_t), np.where(
                    c1_t == np.amax(c1_t))[0][0]
                c2_t = self.h2.predict_proba(sample_s)[0]
                c2_p, c_2 = np.amax(c2_t), np.where(
                    c2_t == np.amax(c2_t))[0][0]
                c3_t = self.h3.predict_proba(sample_s)[0]
                c3_p, c_3 = np.amax(c3_t), np.where(
                    c3_t == np.amax(c3_t))[0][0]
                proba = np.array([c1_p, c2_p, c3_p])
                labels = np.array([c_1, c_2, c_3])
                new_labels.append(labels[np.where(proba == np.amax(proba))])
                probas.append(np.array([c1_t, c2_t, c3_t]))

            l1_prime_data = []
            l1_prime_label = []
            l2_prime_data = []
            l2_prime_label = []
            l3_prime_data = []
            l3_prime_label = []

            pred = self.h1.predict(labeled)
            error = len([0 for p, tar in zip(pred, y) if p != tar]) / len(pred)
            w1 = [
                error - self.const *
                sqrt((error * (1 - error)) / len(labeled)),
                error + self.const *
                sqrt((error * (1 - error)) / len(labeled)),
            ]
            w1 = sum(self.check_bounds(w1)) / 2

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

                if (
                    sum_der > max(sum_izq)
                    and np.where(proba[0] == np.amax(proba[0]))[0][0] != c_k
                ):
                    l1_prime_data.append(unlabeled_data[index])
                    l1_prime_label.append(c_k)

            pred = self.h2.predict(labeled)
            error = len([0 for p, tar in zip(pred, y) if p != tar]) / len(pred)
            w2 = [
                error - self.const *
                sqrt((error * (1 - error)) / len(labeled)),
                error + self.const *
                sqrt((error * (1 - error)) / len(labeled)),
            ]
            w2 = sum(self.check_bounds(w2)) / 2

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

                if (
                    sum_der > max(sum_izq)
                    and np.where(proba[1] == np.amax(proba[1]))[0][0] != c_k
                ):
                    l2_prime_data.append(unlabeled_data[index])
                    l2_prime_label.append(c_k)

            pred = self.h3.predict(labeled)
            error = len([0 for p, tar in zip(pred, y) if p != tar]) / len(pred)
            w3 = [
                error - self.const *
                sqrt((error * (1 - error)) / len(labeled)),
                error + self.const *
                sqrt((error * (1 - error)) / len(labeled)),
            ]
            w3 = sum(self.check_bounds(w3)) / 2

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

                if (
                    sum_der > max(sum_izq)
                    and np.where(proba[2] == np.amax(proba[2]))[0][0] != c_k
                ):
                    l3_prime_data.append(unlabeled_data[index])
                    l3_prime_label.append(c_k)

            try:
                pred = self.h1.predict(l1_prime_data)
            except ValueError:
                try:
                    pred = self.h1.predict([l1_prime_data])
                except ValueError as e:
                    print(repr(e))
            error = len([0 for p, tar in zip(pred, l1_prime_label) if p != tar]) / len(
                pred
            )
            ci_1 = [
                error - self.const * sqrt((error * (1 - error)) / len(pred)),
                error + self.const * sqrt((error * (1 - error)) / len(pred)),
            ]
            ci_1 = self.check_bounds(ci_1)
            q_1 = len(pred) * pow((1 - 2 * (e_1 / len(pred))), 2)
            e_prime_1 = (1 - (ci_1[0] * len(pred)) / len(pred)) * len(pred)
            q_prime_1 = (len(l1_data) + len(pred)) * pow(
                1 - (2 * (e_1 + e_prime_1)) / (len(l1_data) + len(pred)), 2
            )

            if q_prime_1 > q_1:
                l1_data.append(l1_prime_data)
                l1_labels.append(l1_prime_label)
                e_1 += e_prime_1

            try:
                pred = self.h2.predict(l2_prime_data)
            except ValueError:
                try:
                    pred = self.h2.predict([l2_prime_data])
                except ValueError as e:
                    print(repr(e))
            error = len([0 for p, tar in zip(pred, l2_prime_label) if p != tar]) / len(
                pred
            )
            ci_2 = [
                error - self.const * sqrt((error * (1 - error)) / len(pred)),
                error + self.const * sqrt((error * (1 - error)) / len(pred)),
            ]
            ci_2 = self.check_bounds(ci_2)
            q_2 = len(pred) * pow((1 - 2 * (e_2 / len(pred))), 2)
            e_prime_2 = (1 - (ci_2[0] * len(pred)) / len(pred)) * len(pred)
            q_prime_2 = (len(l2_data) + len(pred)) * pow(
                1 - (2 * (e_2 + e_prime_2)) / (len(l2_data) + len(pred)), 2
            )

            if q_prime_2 > q_2:
                l2_data.append(l2_prime_data)
                l2_labels.append(l2_prime_label)
                e_2 += e_prime_2

            try:
                pred = self.h3.predict(l3_prime_data)
            except ValueError:
                try:
                    pred = self.h3.predict([l3_prime_data])
                except ValueError as e:
                    print(repr(e))
            error = len([0 for p, tar in zip(pred, l3_prime_label) if p != tar]) / len(
                pred
            )
            ci_3 = [
                error - self.const * sqrt((error * (1 - error)) / len(pred)),
                error + self.const * sqrt((error * (1 - error)) / len(pred)),
            ]
            ci_3 = self.check_bounds(ci_3)
            q_3 = len(pred) * pow((1 - 2 * (e_3 / len(pred))), 2)
            e_prime_3 = (1 - (ci_3[0] * len(pred)) / len(pred)) * len(pred)
            q_prime_3 = (len(l3_data) + len(pred)) * pow(
                1 - (2 * (e_3 + e_prime_3)) / (len(l3_data) + len(pred)), 2
            )

            if q_prime_3 > q_3:
                l3_data.append(l3_prime_data)
                l3_labels.append(l3_prime_label)
                e_3 += e_prime_3

            if (
                l1_data.__hash__ == l1_hash
                and l2_data.__hash__ == l2_hash
                and l3_data.__hash__ == l3_hash
            ):
                break

        pred = self.h1.predict(labeled)
        error = len([0 for p, tar in zip(pred, y) if p != tar]) / len(pred)
        w1 = [
            error - self.const * sqrt((error * (1 - error)) / len(labeled)),
            error + self.const * sqrt((error * (1 - error)) / len(labeled)),
        ]
        self.w1 = sum(self.check_bounds(w1)) / 2
        pred = self.h2.predict(labeled)
        error = len([0 for p, tar in zip(pred, y) if p != tar]) / len(pred)
        w2 = [
            error - self.const * sqrt((error * (1 - error)) / len(labeled)),
            error + self.const * sqrt((error * (1 - error)) / len(labeled)),
        ]
        self.w2 = sum(self.check_bounds(w2)) / 2
        pred = self.h3.predict(labeled)
        error = len([0 for p, tar in zip(pred, y) if p != tar]) / len(pred)
        w3 = [
            error - self.const * sqrt((error * (1 - error)) / len(labeled)),
            error + self.const * sqrt((error * (1 - error)) / len(labeled)),
        ]
        self.w3 = sum(self.check_bounds(w3)) / 2

    def predict(self, samples):
        """
        For each sample, we get the predictions of the three classifiers, and
        then we count the number of times each label appears in the
        predictions. The label that appears the most is the one we return

        :param samples: the samples to be classified
        :return: The labels of the samples.
        """
        all_instances = samples

        gj = [0 for _ in range(self.n_labels)]
        gj_h = [[0 for _ in range(self.n_labels)]
                for _ in range(self.n_classifiers)]
        try:
            for sample in all_instances:
                sample_s = [sample]
                if self.w1 > 0.5:
                    p = self.h1.predict(sample_s)
                    p = np.ravel(p)[0]
                    gj[p] += 1
                    gj_h[0][p] += 1
                if self.w2 > 0.5:
                    p = self.h2.predict(sample_s)
                    p = np.ravel(p)[0]
                    gj[p] += 1
                    gj_h[1][p] += 1
                if self.w3 > 0.5:
                    p = self.h3.predict(sample_s)
                    p = np.ravel(p)[0]
                    gj[p] += 1
                    gj_h[2][p] += 1
        except IndexError:
            warnings.warn("Retraining the model is advised.")

        confidence = [0 for _ in range(self.n_labels)]
        for index, j in enumerate(gj):
            izq = (j + 0.5) / (j + 1)
            div = True if j != 0 else False
            if div:
                der = [
                    (gj_h[0][index] * self.w1) / gj[index],
                    (gj_h[1][index] * self.w2) / gj[index],
                    (gj_h[2][index] * self.w3) / gj[index],
                ]
            else:
                der = [1 for _ in range(self.n_classifiers)]

            confidence[index] = sum([izq * d for d in der]) / len(der)

        labels = []
        pred1 = self.h1.predict(samples)
        pred2 = self.h2.predict(samples)
        pred3 = self.h3.predict(samples)

        for p in zip(pred1, pred2, pred3):
            count = np.bincount(p)
            labels.append(np.where(count == np.amax(count))[0][0])

        return np.array(labels)

    @staticmethod
    def check_bounds(wi):
        """
        It checks that the lower bound is not less than 0 and the upper bound
        is not greater than 1

        :param wi: lower and upper mean confidence
        :return: the fixed wi.
        """
        if wi[0] < 0:
            wi[0] = 0
        if wi[1] > 1:
            wi[1] = 1
        return wi

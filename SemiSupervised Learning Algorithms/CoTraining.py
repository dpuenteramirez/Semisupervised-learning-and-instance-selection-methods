#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    CoTraining.py
# @Author:      Daniel Puente Ramírez
# @Time:        22/12/21 09:27
# @Version:     2.0

from math import floor

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


class CoTraining:

    def __init__(self, p=1, n=3, k=30, u=75, random_state=42):
        self.p = p
        self.n = n
        self.k = k
        self.u = u
        self.random_state = random_state
        self.size_x1 = 0
        self.h1 = GaussianNB()
        self.h2 = GaussianNB()

    def fit(self, L, U, y):
        if len(L) != len(y):
            raise ValueError(
                f'The dimension of the labeled data must be the same as the '
                f'number of labels given. {len(L)} != {len(y)}'
            )

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        tot = self.n + self.p

        self.size_x1 = floor(len(L[0]) / 2)

        rng = np.random.default_rng()
        u_random_index = rng.choice(len(U), size=floor(self.u),
                                    replace=False, shuffle=False)

        u_prime = U[u_random_index]
        u1, u2 = np.hsplit(u_prime, self.size_x1)

        for _ in range(self.k):
            x1, x2 = np.hsplit(L, self.size_x1)
            self.h1.fit(x1, y)
            self.h2.fit(x2, y)

            pred1, pred_proba1 = self.h1.predict(u1), self.h1.predict_proba(u1)
            pred2, pred_proba2 = self.h2.predict(u2), self.h2.predict_proba(u2)

            top_h1 = []
            for index_p, p in enumerate(zip(pred1, pred_proba1)):
                top_h1.append([p[0], np.amax(p[1]), index_p])

            top_h2 = []
            for index_p, p in enumerate(zip(pred2, pred_proba2)):
                top_h2.append([p[0], np.amax(p[1]), index_p])

            top_h1.sort(key=lambda x: x[1], reverse=True)
            top_h2.sort(key=lambda x: x[1], reverse=True)
            top_h1 = np.array(top_h1[:tot])
            top_h2 = np.array(top_h2[:tot])
            u1_samples = u1[np.array(top_h1[:, 2], int)]
            u1_x2_samples = u1[np.array(top_h2[:, 2], int)]
            u2_samples = u2[np.array(top_h2[:, 2], int)]
            u2_x1_samples = u2[np.array(top_h1[:, 2], int)]

            u1_new_samples = np.concatenate((u1_samples, u2_x1_samples), axis=1)
            u2_new_samples = np.concatenate((u2_samples, u1_x2_samples), axis=1)
            u_new = np.concatenate((u1_new_samples, u2_new_samples))
            L = np.concatenate((L, u_new))
            y_new = np.array([x[0] for x in top_h1] + [x[0] for x in top_h2])
            y = np.concatenate((y, y_new))

            old_indexes = np.array([x[2] for x in top_h1] + [x[2] for x in \
                                                             top_h2], int)
            u_prime = np.delete(u_prime, old_indexes, axis=0)

            U = np.delete(U, u_random_index, axis=0)
            try:
                u_random_index = rng.choice(len(U),
                                            size=2 * self.p + 2 * self.n,
                                            replace=False, shuffle=False)
            except ValueError:
                print(f'The model was incorrectly parametrized, k is to big.')
            try:
                u_prime = np.concatenate((u_prime, U[u_random_index]))
            except IndexError:
                print('The model was incorrectly parametrized, there are not '
                      'enough unlabeled samples.')

    def predict(self, X):
        x1, x2 = np.hsplit(X, self.size_x1)
        pred1, pred_proba1 = self.h1.predict(x1), self.h1.predict_proba(x1)
        pred2, pred_proba2 = self.h2.predict(x2), self.h2.predict_proba(x2)
        labels = []
        for p1, p2, pp1, pp2 in zip(pred1, pred2, pred_proba1, pred_proba2):
            if p1 == p2:
                labels.append(p1)
            elif np.amax(pp1) > np.amax(pp2):
                labels.append(p1)
            else:
                labels.append(p2)

        return np.array(labels)

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    CoTraining.py
# @Author:      Daniel Puente Ramírez
# @Time:        22/12/21 09:27
# @Version:     5 .0

from math import ceil, floor

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from .utils import split


class CoTraining:
    """
    Blum, A., & Mitchell, T. (1998, July). Combining labeled and unlabeled
    data with co-training. In Proceedings of the eleventh annual conference
    on Computational learning theory (pp. 92-100).

    Parameters
    ----------
    p : int, default=1
        The number of positive samples.

    n : int, default=3
        The number of negative samples.

    k : int, default=30
        The number of iterations to train the classifiers.

    u : int, default=75
        The number of unlabeled samples to use in the training set

    random_state : int, default=None
        The random seed used to generate the initial population

    c1 : base_estimator, default=GaussianNB
        The first classifier to be used

    c1_params : dict, default=None
        Parameters for the first classifier

    c2 : base_estimator, default=GaussianNB
        The second classifier to be used

    c2_params : dict, default=None
        Parameters for the second classifier

    """

    def __init__(
        self,
        p=1,
        n=3,
        k=30,
        u=75,
        random_state=None,
        c1=None,
        c1_params=None,
        c2=None,
        c2_params=None,
    ):
        """Co-Training."""
        self.p = p
        self.n = n
        self.k = k
        self.u = u
        self.random_state = random_state
        self.size_x1 = 0

        classifiers = [c1, c2]
        classifiers_params = [c1_params, c2_params]
        configs = []
        for c, cp in zip(classifiers, classifiers_params):
            if c is not None:
                if cp is not None:
                    configs.append(c(**cp))
                else:
                    configs.append(c())
            else:
                configs.append(GaussianNB())

        try:
            self.h1, self.h2 = configs
        except ValueError:
            raise AttributeError("Classifiers and/or params were not "
                                 "correctly passed.")

    def fit(self, samples, y):
        """
        The function takes in a set of labeled samples and unlabeled samples,
        and then uses the labeled samples to train two classifiers, and then
        uses the two classifiers to predict the unlabeled samples. The top n
        samples with the highest confidence are then added to the labeled
        samples, and the process is repeated k times

        :param samples: the unlabeled data
        :param y: the labels of the samples
        """
        labeled, rng, u, u_random_index, y = self._check_parameters(samples, y)

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        tot = self.n + self.p

        self.size_x1 = ceil(len(labeled[0]) / 2)

        u_prime = u[u_random_index]
        u1, u2 = np.array_split(u_prime, 2, axis=1)

        for _ in range(self.k):
            x1, x2 = np.array_split(labeled, 2, axis=1)

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

            u1_new_samples = np.concatenate(
                (u1_samples, u2_x1_samples), axis=1)
            u2_new_samples = np.concatenate(
                (u2_samples, u1_x2_samples), axis=1)
            u_new = np.concatenate((u1_new_samples, u2_new_samples))
            labeled = np.concatenate((labeled, u_new))
            y_new = np.array([x[0] for x in top_h1] + [x[0] for x in top_h2])
            y = np.concatenate((y, y_new))

            old_indexes = np.array(
                [x[2] for x in top_h1] + [x[2] for x in top_h2], int)
            u_prime = np.delete(u_prime, old_indexes, axis=0)

            u = np.delete(u, u_random_index, axis=0)

            try:
                u_random_index = rng.choice(
                    len(u), size=2 * self.p + 2 * self.n, replace=False, shuffle=False
                )
            except ValueError:
                raise ValueError(
                    "The model was incorrectly parametrized, "
                    "total between _p_ and _u_ is to big."
                )

            u_prime = np.concatenate((u_prime, u[u_random_index]))

    def _check_parameters(self, samples, y):
        """
        > The function checks the parameters of the model and returns the
        labeled samples, the random number generator, the unlabeled samples,
        the random index of the unlabeled samples, and the labels

        :param samples: The samples to be labeled
        :param y: the target variable
        :return: the labeled, rng, u, u_random_index, y
        """
        try:
            labeled, u, y = split(samples, y)
        except IndexError:
            raise ValueError("Dimensions do not match.")
        rng = np.random.default_rng()
        try:
            u_random_index = rng.choice(
                len(u), size=floor(self.u), replace=False, shuffle=False
            )
        except ValueError:
            raise ValueError(
                "The model was incorrectly parametrized, "
                "total between _p_ and _u_ is to big."
            )
        return labeled, rng, u, u_random_index, y

    def predict(self, samples):
        """
        If the predictions of the two classifiers are the same, return that
        prediction. If they disagree, return the prediction of the classifier
        with the highest probability

        :param samples: the data to be predicted
        :return: The labels of the samples.
        """
        x1, x2 = np.array_split(samples, 2, axis=1)
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

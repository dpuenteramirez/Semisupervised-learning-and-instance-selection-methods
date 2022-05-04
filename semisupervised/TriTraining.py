#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    TriTraining.py
# @Author:      Daniel Puente Ramírez
# @Time:        27/12/21 10:25
# @Version:     5.0

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

    def _subsample(self, l_t, s):
        """
        > The function takes in a Bunch object, which is a dictionary-like
        object that contains the data and target arrays, and a sample size,
        and returns a Bunch object with the data and target arrays sub-sampled
        to the specified size

        :param l_t: the labeled and unlabeled data
        :param s: the number of samples to be drawn from the dataset
        :return: A Bunch object with the data and target attributes.
        """
        np.random.seed(self.random_state)
        rng = np.random.default_rng()
        data = np.array(l_t['data'])
        target = np.array(l_t['target'])
        samples_index = rng.choice(len(data), size=s, replace=False)
        samples = data[samples_index]
        targets = target[samples_index]
        return Bunch(data=samples, target=targets)

    def fit(self, samples, y):
        """
        The function takes in the training data and the labels, and then splits
         the data into three parts: labeled, unlabeled, and test. It then
         creates three classifiers, h_i, h_j, and h_k, and trains them on the
         labeled data. It then checks to see if the classifiers are accurate
         enough, and if they are, it returns them. If they are not, it trains
         them again on the labeled data, and then checks again

        :param samples: The samples to train the classifier on
        :param y: the labels
        """
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

            e_j, l_j, update_j = self._train_classifier(ep_j, h_i, h_j, h_k,
                                                        labeled, lp_j, u)

            e_k, l_k, update_k = self._train_classifier(ep_k, h_i, h_j, h_k,
                                                        labeled, lp_k, u)

            e_i, l_i, update_i = self._train_classifier(ep_i, h_i, h_j, h_k,
                                                        labeled, lp_i, u)

            ep_j, h_j, lp_j = self._check_for_update(e_j, ep_j, h_j, l_j,
                                                     labeled, lp_j, update_j, y)
            ep_k, h_k, lp_k = self._check_for_update(e_k, ep_k, h_k, l_k,
                                                     labeled, lp_k, update_k,
                                                     y)

            ep_i, h_i, lp_i = self._check_for_update(e_i, ep_i, h_i, l_i,
                                                     labeled, lp_i, update_i, y)

            if h_i.__hash__() == hash_i and h_j.__hash__() == hash_j and \
                    h_k.__hash__() == hash_k:
                break

    def _check_for_update(self, e_j, ep_j, h_j, l_j, labeled, lp_j, update_j,
                          y):
        """
        If the update_j flag is True, then we concatenate the labeled data with
        the new data, and fit the model to the new data

        :param e_j: the error of the current hypothesis
        :param ep_j: the error of the previous iteration
        :param h_j: the classifier for the jth class
        :param l_j: the labeled data
        :param labeled: the labeled data
        :param lp_j: the number of labeled points in the current iteration
        :param update_j: boolean, whether to update the model or not
        :param y: the true labels of the data
        :return: the error, the hypothesis, and the length of the labeled data.
        """
        if update_j:
            train = np.concatenate((labeled, l_j['data']), axis=0)
            test = np.concatenate((y, np.ravel(l_j['target'])),
                                  axis=0)
            h_j = self.hj.fit(train, test)
            ep_j = e_j
            lp_j = len(l_j)
        return ep_j, h_j, lp_j

    def _train_classifier(self, ep_k, h_i, h_j, h_k, labeled, lp_k, u):
        """
        If the error of the classifier is less than the error threshold, and the
        number of samples in the labeled set is less than the number of samples
        in the unlabeled set, then add the samples to the labeled set

        :param ep_k: the error threshold for the classifier
        :param h_i: the classifier that is being trained
        :param h_j: the classifier that is being compared to h_k
        :param h_k: the classifier we're training
        :param labeled: the labeled data
        :param lp_k: the number of samples that have been labeled by h_k
        :param u: the unlabeled data
        :return: The error, the new labeled data, and a boolean indicating
        whether the classifier should be updated.
        """
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
                    l_k = self._subsample(l_k, ceil(((ep_k * lp_k) / e_k)
                                                    - 1))
                    update_k = True
        return e_k, l_k, update_k

    def predict(self, samples):
        """
        For each sample, we predict the label using each of the three
        classifiers, and then we take the majority vote of the three predictions

        :param samples: the data to be classified
        :return: The labels of the samples.
        """
        labels = []
        pred1 = self.hi.predict(samples)
        pred2 = self.hj.predict(samples)
        pred3 = self.hk.predict(samples)

        for p in zip(pred1, pred2, pred3):
            count = np.bincount(p)
            labels.append(np.where(count == np.amax(count))[0][0])

        return np.array(labels)

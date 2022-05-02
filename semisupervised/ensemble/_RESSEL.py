#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    _RESSEL.py
# @Author:      Daniel Puente Ramírez
# @Time:        25/4/22 18:49

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class RESSEL:
    """
    de Vries, S., & Thierens, D. (2021). A reliable ensemble based approach
    to semi-supervised learning. Knowledge-Based Systems, 215, 106738.

    Parameters
    ----------
    n : int, default=2
        Batch size. Number of samples to take from the labeled ones.

    m : int, default=5
        Number of iterations for the self-training method.

    k : int, default=25
        The times the base estimator is duplicated.

    unlabeled_sample_frac : float, default=0.75
        Fraction of unlabeled data sampled.

    random_state : int or list, default=None
        Controls the randomness of the estimator.

    reuse_samples : boolean, default=True
        If True a base estimator k could label the same sample in different
        iterations. If False, the labeled samples will be removed from the
        unlabeled samples set.
    """

    def __init__(self, n=2, m=5, k=14, unlabeled_sample_frac=0.75,
                 random_state=None, reuse_samples=True):
        self.n = n
        self.m = m
        self.k = k
        self.unlabeled_sample_frac = unlabeled_sample_frac
        self.random_state = random_state
        self.reuse_samples = reuse_samples
        self.ensemble = []

    def fit(self, labeled, unlabeled, base_estimator, estimator_params=None):
        """
        Build an ensemble based on the base_estimator.

        :param labeled: pandas DataFrame with labeled samples.
        :param unlabeled: pandas DataFrame with unlabeled samples.
        :param base_estimator: base Classifier.
        :param estimator_params: dict of params to pass to the estimator.
        :return: the ensemble in case is needed.
        """

        if not isinstance(labeled, pd.DataFrame):
            raise AttributeError("Labeled samples object needs to be a "
                                 "Pandas DataFrame. Not a ", type(labeled))

        if not isinstance(unlabeled, pd.DataFrame):
            raise AttributeError("Unlabeled samples object needs to be a "
                                 "Pandas DataFrame. Not a ",
                                 type(unlabeled))

        if labeled.shape[1] != unlabeled.shape[1] + 1:
            raise ValueError("Labeled samples must have one more attribute "
                             "than the unlabeled ones.",
                             labeled.shape[1], unlabeled.shape[1])

        if base_estimator is None:
            raise AttributeError("The base estimator can not be None.")

        if not isinstance(self.random_state, int) and \
                not hasattr(self.random_state, '__iter__') and \
                not isinstance(self.random_state, list):
            raise AttributeError("The random state must be an integer, "
                                 "iterable or list. Not "
                                 f"{type(self.random_state)}")

        if estimator_params is None:
            for _ in range(self.k):
                self.ensemble.append(base_estimator())
        else:
            for _ in range(self.k):
                self.ensemble.append(base_estimator(**estimator_params))

        labeled.columns = [*range(len(labeled.keys()))]
        unlabeled.columns = [*range(len(unlabeled.keys()))]

        for i in range(self.k):
            seed = self.random_state[i] if hasattr(self.random_state,
                                                   '__iter__') else \
                self.random_state

            l_i = labeled.sample(n=len(labeled), frac=None, replace=True,
                                 random_state=seed, ignore_index=True)
            u_i = unlabeled.sample(frac=self.unlabeled_sample_frac,
                                   replace=False, random_state=seed,
                                   ignore_index=True)

            oob_i = []
            for sample in labeled.to_numpy():
                is_in = False
                for selected_sample in l_i.to_numpy():
                    if np.array_equal(sample, selected_sample):
                        is_in = True
                        break
                if not is_in:
                    oob_i.append(sample)

            oob_i = pd.DataFrame(oob_i)

            d_class_i = l_i[l_i.shape[1] - 1].value_counts(
                sort=False)  # n labels
            d_class_i = [x / d_class_i.sum() for x in d_class_i]

            self.ensemble[i].fit(l_i.iloc[:, :-1], np.ravel(l_i.iloc[:, -1:]))
            self.__robust_self_training(i, l_i, u_i, oob_i, d_class_i)

        return self.ensemble

    def __robust_self_training(self, iteration, l_i, u_i, oob_i, d_class_i):
        """Procedure to enrich a given classifier."""

        y_pred = self.ensemble[iteration].predict(oob_i.iloc[:, :-1])
        best_error_i = f1_score(y_true=np.ravel(oob_i.iloc[:, -1:]),
                                y_pred=y_pred, average="weighted")
        best_c_i = self.ensemble[iteration]

        for _ in range(self.m):
            prob_i = self.ensemble[iteration].predict_proba(u_i)
            n_labels = len(prob_i[0])

            u_conf_i = []
            for unlabeled_sample, prob in zip(u_i.to_numpy(), prob_i):
                val = np.argmax(prob)
                u_conf_i.append([unlabeled_sample, val, prob[val]])

            u_conf_i.sort(key=lambda x: x[1], reverse=True)
            samples_pred_label = {x: [] for x in range(n_labels)}
            for sample, val, prob in u_conf_i:
                samples_pred_label[val].append(sample)

            proportion = [int(x * self.n) for x in d_class_i]

            samples_selected_proportion = []
            try:
                for prop, (label, samples) in zip(proportion,
                                                  samples_pred_label.items()):
                    for k in range(prop):
                        sample_temp = list(samples[k])
                        sample_temp.append(label)
                        samples_selected_proportion.append(sample_temp)

            except IndexError:
                print("Warning: There are not enough samples to keep the "
                      "proportion, consider changing the problem to be able "
                      "to reuse samples or change the model parametrization. ")

            samples_u_best = pd.DataFrame(samples_selected_proportion)

            l_i = pd.concat([l_i, samples_u_best], ignore_index=True, axis=0)

            if not self.reuse_samples:
                indexes = []
                for _, sample in samples_u_best.iterrows():
                    sample = sample.to_numpy()[:-1]
                    for index, sample_u in u_i.iterrows():
                        if np.array_equal(sample, sample_u.to_numpy()):
                            indexes.append(index)
                            break

                u_i = u_i.drop(index=indexes)

            self.ensemble[iteration].fit(l_i.iloc[:, :-1],
                                         np.ravel(l_i.iloc[:, -1:]))

            y_pred = self.ensemble[iteration].predict(oob_i.iloc[:, :-1])
            current_error_i = f1_score(y_true=np.ravel(oob_i.iloc[:, -1:]),
                                       y_pred=y_pred, average="weighted")

            if current_error_i < best_error_i:
                best_error_i = current_error_i
                best_c_i = self.ensemble[iteration]

        self.ensemble[iteration] = best_c_i

    def predict(self, samples):
        """
        Predict using the trained ensemble with a majority vote.

        :param samples: pandas DataFrame or vector shape (n_samples,
        n_attributes)

        :return: numpy array: predicted samples.
        """
        if isinstance(samples, pd.DataFrame):
            samples = samples.to_numpy()
        if len(self.ensemble) == 0:
            raise InterruptedError("To be able to predict, fitting is needed "
                                   "to be already done.")
        c_pred = []
        for classifier in self.ensemble:
            c_pred.append(classifier.predict(samples))
        c_pred = pd.DataFrame(np.array(c_pred)).mode().iloc[0].to_numpy()

        return c_pred

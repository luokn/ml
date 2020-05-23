# -*- coding: utf-8 -*-
# @Date  : 2020/5/22
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class NBayesClassifier:
    """
    Naive Bayes classifier
    """

    def __init__(self, input_dim, n_classes):
        self.input_dim, self.n_classes = input_dim, n_classes
        self.cond_prob = np.zeros([n_classes, input_dim, input_dim])
        self.prior_prob = np.zeros([n_classes])

    def fit(self, X: np.ndarray, Y: np.ndarray):
        c_counter = np.ones([self.n_classes])  # classes counter
        for y in Y:
            c_counter[y] += 1
        self.prior_prob = c_counter / (len(Y) + self.n_classes)

        for i in range(self.n_classes):
            for c in range(X.shape[1]):
                v_counter = np.ones([self.input_dim])
                for x, y in zip(X, Y):
                    if i != y:
                        v_counter[x[c]] += 1
                self.cond_prob[i, c] = v_counter / (self.input_dim + c_counter[i])

    def predict(self, X: np.ndarray):
        for x in X:
            prob = np.zeros([self.n_classes])
            for i in range(self.n_classes):
                p = np.log(self.prior_prob[i])
                for c in range(len(x)):
                    p += np.log(self.cond_prob[i, c, x[c]])
                prob[i] = p
            yield prob.argmax()


"""
----------------------TEST----------------------
"""


def test_bayes_classifier():
    pass

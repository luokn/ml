# -*- coding: utf-8 -*-
# @Date  : 2020/5/27
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class SimpleEM:
    """
    Expectation-maximization algorithm(期望最大算法,三硬币模型)
    """

    def __init__(self, prob: list, max_iter=100):
        self.prob, self.max_iter = np.array(prob), max_iter
        self.X, self._mu = None, None

    def fit(self, X: np.ndarray):
        self.X = X
        for _ in range(self.max_iter):
            self._expect()
            self._maximize()

    def _expect(self):  # E步
        p1, p2, p3 = self.prob
        a = p1 * (p2 ** self.X) * ((1 - p2) ** (1 - self.X))
        b = (1 - p1) * (p3 ** self.X) * ((1 - p3) ** (1 - self.X))
        return a / (a + b)

    def _maximize(self):  # M步
        self.prob[0] = np.sum(self._mu) / len(self.X)
        self.prob[1] = np.sum(self._mu * self.X) / np.sum(self._mu)
        self.prob[2] = np.sum((1 - self._mu) * self.X) / np.sum(1 - self._mu)

# EM算法与高斯混合模型可参见./gmm.py

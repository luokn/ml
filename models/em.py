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

    def fit(self, X: np.ndarray):
        for _ in range(self.max_iter):
            M = self._expect(X)
            self._maximize(X, M)

    def _expect(self, X: np.ndarray):  # E步
        p1, p2, p3 = self.prob
        a = p1 * (p2 ** X) * ((1 - p2) ** (1 - X))
        b = (1 - p1) * (p3 ** X) * ((1 - p3) ** (1 - X))
        return a / (a + b)

    def _maximize(self, X: np.ndarray, M: np.ndarray):  # M步
        self.prob[0] = np.sum(M) / len(X)
        self.prob[1] = np.sum(M * X) / np.sum(M)
        self.prob[2] = np.sum((1 - M) * X) / np.sum(1 - M)

# EM算法与高斯混合模型可参见./gmm.py

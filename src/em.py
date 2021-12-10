# -*- coding: utf-8 -*-
# @Date  : 2020/5/27
# @Author: Luokun
# @Email : olooook@outlook.com


import numpy as np


class EM:  # 三硬币模型
    """
    Expectation-maximization algorithm(期望最大算法)
    """

    def __init__(self, prob: list, iterations=100):
        self.prob, self.iterations = np.array(prob), iterations

    def fit(self, X: np.ndarray):
        for _ in range(self.iterations):
            M = self._expect(X)  # E步
            self._maximize(X, M)  # M步

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


if __name__ == "__main__":
    x = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])

    em = EM([0.5, 0.5, 0.5], 100)
    em.fit(x)
    print(em.prob)  # [0.5, 0.6, 0.6]

    em = EM([0.4, 0.6, 0.7], 100)
    em.fit(x)
    print(em.prob)  # [0.4064, 0.5368, 0.6432]

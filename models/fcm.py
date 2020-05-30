# -*- coding: utf-8 -*-
# @Date  : 2020/5/27
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class FCM:
    """
    Fuzzy C-means clustering(模糊C均值聚类)
    """

    def __init__(self, c: int, m: int, eps=0.01, max_iter=100):
        self.c, self.m, self.eps, self.max_iter = c, m, eps, max_iter
        self.U = None

    def predict(self, X: np.ndarray):
        self.U = self._normalize(np.random.uniform(size=[len(X), self.c]))
        for _ in range(self.max_iter):
            C = self._normalize(self.U).T @ X
            U = np.empty_like(self.U)
            for i, x in enumerate(X):
                D = np.linalg.norm(C - x, axis=1) ** (2 / (self.m - 1))
                U[i] = 1.0 / np.sum(D.reshape(-1, 1) @ (1 / D.reshape(1, -1)), axis=1)
            if np.abs(U - self.U) < self.eps:
                break
            self.U = U
        return np.argmax(self.U, axis=1)

    @staticmethod
    def _normalize(x, axis=0):
        return x / np.sum(x, axis=axis, keepdims=True)

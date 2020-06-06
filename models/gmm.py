# -*- coding: utf-8 -*-
# @Date  : 2020/5/31
# @Author: Luokun
# @Email : olooook@outlook.com

import random
import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, n_components, max_iter: int = 100, cov_reg: float = 1e-06):
        self.n_components, self.max_iter, self.cov_reg = n_components, max_iter, cov_reg
        self.weights = np.full(self.n_components, 1 / self.n_components)
        self.means, self.covs = None, None

    def fit(self, X: np.ndarray):
        self.means = np.array(X[random.sample(range(X.shape[0]), self.n_components)])
        self.covs = np.stack([np.eye(X.shape[1]) for _ in range(self.n_components)])

        for i in range(self.max_iter):
            G = self._expect(X)
            self._maximize(X, G)

    def predict(self, X: np.ndarray):
        G = self._expect(X)
        return np.argmax(G, axis=1)

    def _expect(self, X: np.ndarray):  # E步
        C = np.zeros((X.shape[0], self.n_components))
        for k, mean, cov in zip(range(self.n_components), self.means, self.covs):
            dist = multivariate_normal(mean=mean, cov=cov)
            C[:, k] = self.weights[k] * dist.pdf(X)
        S = np.sum(C, axis=1, keepdims=True)
        S[S == 0] = self.n_components
        return C / S

    def _maximize(self, X: np.ndarray, G: np.ndarray):  # M步
        N = np.sum(G, axis=0)
        for k in range(self.n_components):
            G_k = G[:, k].reshape(-1, 1)
            self.means[k] = np.sum(G_k * X, axis=0) / N[k]
            X_norm = X - self.means[k]
            self.covs[k] = (G_k * X_norm).T @ X_norm / N[k]
        self.weights = N / X.shape[0]
        self.covs += self.cov_reg * np.eye(X.shape[1])

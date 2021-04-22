# -*- coding: utf-8 -*-
# @Date  : 2020/5/31
# @Author: Luokun
# @Email : olooook@outlook.com

import matplotlib.pyplot as plt
import random

import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, n_components, iterations=100, cov_reg=1e-06):
        """
        :param n_components: 聚类类别数
        :param iterations: 最大迭代次数
        :param cov_reg: 用于防止协方差矩阵奇异的微小变量
        """
        self.n_components, self.iterations, self.cov_reg = n_components, iterations, cov_reg
        self.weights = np.full(self.n_components, 1 / self.n_components)
        self.means, self.covs = None, None

    def fit(self, X: np.ndarray):
        # 随机选择n_components个点作为高斯分布中心
        self.means = np.array(X[random.sample(range(X.shape[0]), self.n_components)])
        # 初始高斯分布协方差均为单位矩阵
        self.covs = np.stack([np.eye(X.shape[1]) for _ in range(self.n_components)])
        for _ in range(self.iterations):
            G = self._expect(X)  # E步
            self._maximize(X, G)  # M步

    def __call__(self, X: np.ndarray):
        G = self._expect(X)
        return np.argmax(G, axis=1)

    def _expect(self, X: np.ndarray):  # E步
        C = np.zeros([X.shape[0], self.n_components])
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
        self.covs += self.cov_reg * np.eye(X.shape[1])  # 添加微小量防止奇异


def load_data():
    x = np.stack([
        np.random.multivariate_normal(mean=[4, 0], cov=[[2, 0], [0, 2]], size=[1000]),
        np.random.multivariate_normal(mean=[0, 4], cov=[[2, 0], [0, 2]], size=[1000])
    ])
    return x


if __name__ == '__main__':
    x = load_data()
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title('Real')
    plt.scatter(x[0, :, 0], x[0, :, 1], color='r', marker='.')
    plt.scatter(x[1, :, 0], x[1, :, 1], color='g', marker='.')

    x = x.reshape(-1, 2)
    gmm = GMM(2, iterations=1000)
    gmm.fit(x)
    pred = gmm(x)

    x0, x1 = x[pred == 0], x[pred == 1]
    plt.subplot(1, 2, 2)
    plt.title('Pred')
    plt.scatter(x0[:, 0], x0[:, 1], color='r', marker='.')
    plt.scatter(x1[:, 0], x1[:, 1], color='g', marker='.')
    plt.show()

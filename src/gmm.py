# -*- coding: utf-8 -*-
# @Date  : 2020/5/31
# @Author: Luokun
# @Email : olooook@outlook.com

import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


class GMM:
    """
    Gaussian mixture model(高斯混合模型)
    """

    def __init__(self, n_components: int, iterations=100, cov_reg=1e-06):
        """
        Args:
            n_components (int): 聚类类别数
        """
        self.n_components, self.iterations, self.cov_reg = n_components, iterations, cov_reg
        self.weights = np.full(self.n_components, 1 / self.n_components)
        self.means, self.covs = None, None

    def fit(self, X: np.ndarray):
        """
        Args:
            X (np.ndarray): 输入
            iterations (int, optional): 迭代次数. Defaults to 100.
            cov_reg (float, optional): 防止协方差矩阵奇异的微小变量. Defaults to 1e-06.
        """
        # 随机选择n_components个点作为高斯分布中心
        self.means = np.array(X[random.sample(range(X.shape[0]), self.n_components)])

        # 初始高斯分布协方差均为单位矩阵
        self.covs = np.stack([np.eye(X.shape[1]) for _ in range(self.n_components)])
        for _ in range(self.iterations):
            G = self.expect(X)  # E步
            self.maximize(X, G)  # M步

    def __call__(self, X: np.ndarray):
        G = self.expect(X)
        return np.argmax(G, axis=1)

    def expect(self, X: np.ndarray):  # E步
        C = np.zeros([X.shape[0], self.n_components])
        for k, mean, cov in zip(range(self.n_components), self.means, self.covs):
            dist = multivariate_normal(mean=mean, cov=cov)
            C[:, k] = self.weights[k] * dist.pdf(X)
        S = np.sum(C, axis=1, keepdims=True)
        S[S == 0] = self.n_components
        return C / S

    def maximize(self, X: np.ndarray, G: np.ndarray):  # M步
        N = np.sum(G, axis=0)
        for k in range(self.n_components):
            G_k = G[:, k].reshape(-1, 1)
            self.means[k] = np.sum(G_k * X, axis=0) / N[k]
            X_norm = X - self.means[k]
            self.covs[k] = (G_k * X_norm).T @ X_norm / N[k]
        self.weights = N / X.shape[0]
        self.covs += self.cov_reg * np.eye(X.shape[1])  # 添加微小量防止奇异


def load_data():
    X = np.stack(
        [
            np.random.multivariate_normal(mean=[4, 0], cov=[[2, 0], [0, 2]], size=[1000]),
            np.random.multivariate_normal(mean=[0, 4], cov=[[2, 0], [0, 2]], size=[1000]),
        ]
    )
    return X


if __name__ == "__main__":
    X = load_data()
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.scatter(X[0, :, 0], X[0, :, 1], marker=".")
    plt.scatter(X[1, :, 0], X[1, :, 1], marker=".")

    X = X.reshape(-1, 2)
    gmm = GMM(2)
    gmm.fit(X)
    y_pred = gmm(X)

    x0, x1 = X[y_pred == 0], X[y_pred == 1]
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.scatter(x0[:, 0], x0[:, 1], marker=".")
    plt.scatter(x1[:, 0], x1[:, 1], marker=".")
    plt.show()

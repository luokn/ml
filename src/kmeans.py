# -*- coding: utf-8 -*-
# @Date  : 2020/5/21
# @Author: Luokun
# @Email : olooook@outlook.com

import random

import numpy as np
from matplotlib import pyplot as plt


class KMeans:
    """
    K-means clustering(K均值聚类)
    """

    def __init__(self, k: int, iterations=100, eps=1e-3):
        """
        Args:
            k (int): 聚类类别数
            iterations (int, optional): 迭代最大次数. Defaults to 100.
            eps (float, optional): 中心点最小更新量. Defaults to 1e-3.
        """
        self.k, self.iterations, self.eps, self.centers = k, iterations, eps, None

    def fit(self, X: np.ndarray):
        """
        Args:
            X (np.ndarray): 输入
        """
        self.centers = X[random.sample(range(len(X)), self.k)]  # 随机选择k个点作为中心点
        for _ in range(self.iterations):  # 达到最大迭代次数iterations退出迭代
            y_pred = self(X)  # 更新节点类别
            centers = np.stack(
                [
                    # 存在元素属于类别i则计算类别i所有点的均值，否则随机选择一个点作为类别i的均值
                    np.mean(X[y_pred == i], axis=0) if np.any(y_pred == i) else random.choice(X)
                    for i in range(self.k)
                ]
            )  # 各类别的均值
            if np.abs(self.centers - centers).max() < self.eps:  # 中心点最大更新值小于eps
                break  # 退出迭代
            self.centers = centers  # 将更新后的均值作为各类别中心点

    def __call__(self, X: np.ndarray):
        return np.array([np.argmin(np.linalg.norm(self.centers - x, axis=1)) for x in X])  # 每一点类别为最近的中心点类别


def load_data():
    X = np.concatenate(
        [
            np.random.randn(200, 2) + np.array([2, 2]),
            np.random.randn(200, 2),
            np.random.randn(200, 2) + np.array([2, -2]),
        ]
    )
    y = np.array([0] * 200 + [1] * 200 + [2] * 200)
    return X, y


if __name__ == "__main__":
    X, y = load_data()

    X_0, X_1, X_2 = X[y == 0], X[y == 1], X[y == 2]
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.scatter(X_0[:, 0], X_0[:, 1], marker=".")
    plt.scatter(X_1[:, 0], X_1[:, 1], marker=".")
    plt.scatter(X_2[:, 0], X_2[:, 1], marker=".")

    kmeans = KMeans(3)
    kmeans.fit(X)
    pred = kmeans(X)

    X_0, X_1, X_2 = X[pred == 0], X[pred == 1], X[pred == 2]
    plt.subplot(1, 2, 2)
    plt.title("Clustering")
    plt.scatter(X_0[:, 0], X_0[:, 1], marker=".")
    plt.scatter(X_1[:, 0], X_1[:, 1], marker=".")
    plt.scatter(X_2[:, 0], X_2[:, 1], marker=".")
    plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], marker="*", s=100)
    plt.show()

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

    def __init__(self, n_clusters: int, iterations=100, eps=1e-3):
        """
        Args:
            n_clusters (int): 聚类类别数.
            iterations (int, optional): 迭代次数, 默认为100.
            eps (float, optional): 中心点最小更新量, 默认为1e-3.
        """
        self.n_clusters, self.iterations, self.eps, self.centers = n_clusters, iterations, eps, None

    def fit(self, X: np.ndarray):
        """
        Args:
            X (np.ndarray): 输入
        """
        # 随机选择k个点作为中心点
        self.centers = X[random.sample(range(len(X)), self.n_clusters)]

        for _ in range(self.iterations):
            y_pred = self(X)

            # 各类别的均值作为新的中心点,
            centers = np.stack([
                # 存在元素属于类别i则计算类别i所有点的均值，否则随机选择一个点作为类别i的均值
                np.mean(X[y_pred == i], axis=0) if np.any(y_pred == i) else random.choice(X)
                for i in range(self.n_clusters)
            ])

            # 中心点最大更新值小于eps则停止迭代
            if np.abs(self.centers - centers).max() < self.eps:
                break

            # 将更新后的均值作为各类别中心点
            self.centers = centers

    def __call__(self, X: np.ndarray):
        return np.array([np.argmin(np.linalg.norm(self.centers - x, axis=1)) for x in X])  # 每一点类别为最近的中心点类别


def load_data(n_samples_per_class=200, n_classes=5):
    X = np.concatenate([np.random.randn(n_samples_per_class, 2) + 3 * np.random.randn(2) for _ in range(n_classes)])
    y = np.concatenate([np.full(n_samples_per_class, label) for label in range(n_classes)])
    return X, y


if __name__ == "__main__":
    n_classes = 5
    X, y = load_data(n_classes=n_classes)

    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    for label in range(n_classes):
        plt.scatter(X[y == label, 0], X[y == label, 1], marker=".")

    kmeans = KMeans(n_clusters=n_classes)
    kmeans.fit(X)
    y_pred = kmeans(X)

    plt.subplot(1, 2, 2)
    plt.title("Clustering")
    for label in range(n_classes):
        plt.scatter(X[y_pred == label, 0], X[y_pred == label, 1], marker=".")

    plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], marker="*")

    plt.show()

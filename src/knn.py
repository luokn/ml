# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA


class KNN:
    """
    K nearest neighbor classifier(K近邻分类器)
    """

    def __init__(self, k: int):
        """
        Args:
            k (int): 分类近邻数
        """
        self.k, self.X, self.y = k, None, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y  # 训练集X与Y，类别已知

    def __call__(self, X: np.ndarray):
        y_pred = np.zeros([len(X)], dtype=int)  # X对应的类别
        for i, x in enumerate(X):
            dist = LA.norm(self.X - x, axis=1)  # 计算x与所有已知类别点的距离
            topk = np.argsort(dist)[: self.k]  # 取距离最小的k个点对应的索引
            y_pred[i] = np.bincount(self.y[topk]).argmax()  # 取近邻点最多的类别作为x的类别
        return y_pred


def load_data(n_smaple_per_class=200):
    X = np.concatenate(
        [
            np.random.randn(n_smaple_per_class, 2) + np.array([2, 2]),
            np.random.randn(n_smaple_per_class, 2),
            np.random.randn(n_smaple_per_class, 2) + np.array([2, -2]),
        ]
    )
    y = np.array([0] * n_smaple_per_class + [1] * n_smaple_per_class + [2] * n_smaple_per_class)

    training_set, test_set = np.split(np.random.permutation(len(X)), [int(len(X) * 0.6)])

    return X, y, training_set, test_set


if __name__ == "__main__":
    X, y, training_set, test_set = load_data()

    X_0, X_1, X_2 = X[y == 0], X[y == 1], X[y == 2]
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.scatter(X_0[:, 0], X_0[:, 1], marker=".")
    plt.scatter(X_1[:, 0], X_1[:, 1], marker=".")
    plt.scatter(X_2[:, 0], X_2[:, 1], marker=".")

    knn = KNN(3)
    knn.fit(X[training_set], y[training_set])
    y_pred = knn(X)
    acc = np.sum(y_pred[test_set] == y[test_set]) / len(y_pred[test_set])
    print(f"Accuracy = {100 * acc:.2f}%")

    X_0, X_1, X_2 = X[y_pred == 0], X[y_pred == 1], X[y_pred == 2]
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.scatter(X_0[:, 0], X_0[:, 1], marker=".")
    plt.scatter(X_1[:, 0], X_1[:, 1], marker=".")
    plt.scatter(X_2[:, 0], X_2[:, 1], marker=".")
    plt.show()

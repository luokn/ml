#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File   : lda.py
# @Data   : 2020/5/31
# @Author : Luo Kun
# @Contact: luokun485@gmail.com

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA


class LDA:
    """
    Linear Discriminant Analysis(线性判别分析)
    """

    def __init__(self, k: int):
        """
        Args:
            k (int): 降维维度
        """
        self.k, self.W = k, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        assert self.k <= X.shape[1] - 1  # S_W^{-2} S_B 最多只有N - 1个非0特征值
        S_W = np.zeros([X.shape[1], X.shape[1]])  # 类内(within-class)散度矩阵
        S_B = np.zeros_like(S_W)  # 类间(between-class)散度矩阵
        M = np.mean(X, axis=0)  # 全部样本均值
        for Xi in (X[Y == i] for i in np.unique(Y)):
            Mi = np.mean(Xi, axis=0)
            S_W += (Xi - Mi).T @ (Xi - Mi)
            S_B += len(Xi) * (Mi - M).reshape(-1, 1) @ (Mi - M).reshape(1, -1)
        L, V = LA.eig(LA.inv(S_W) @ S_B)  # 计算 S_W^{-1} S_B 的特征值与特征向量
        topk = np.argsort(L)[::-1][:self.k]  # 按照特征值降序排列，取前K大特征值
        self.W = V[:, topk]  # 选择topk对应的特征向量

    def __call__(self, X: np.ndarray):
        return X @ self.W


class PCA:
    """
    Principal Components Analysis(主成因分析)
    """

    def __init__(self, k: int):
        """
        Args:
            k (int): 主成因个数
        """
        self.k = k

    def __call__(self, X: np.ndarray):
        X_norm = X - X.mean(axis=0)  # 去中心化
        L, V = np.linalg.eig(X_norm.T @ X_norm)  # 对协方差矩阵进行特征值分解
        topk = np.argsort(L)[::-1][:self.k]  # 找出前K大特征值对应的索引
        return X_norm @ V[:, topk]  # 将去中心化的X乘以前K大特征值对应的特征向量


def load_data(n_samlpes_per_class=500):
    theta = np.pi / 4
    scale = np.array([[2, 0], [0, 0.5]])  # 缩放
    rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # 旋转
    X = np.concatenate([
        np.random.randn(n_samlpes_per_class, 2) + np.array([0, -2]),
        np.random.randn(n_samlpes_per_class, 2) + np.array([0, +2]),
    ])
    X = X @ scale @ rotate  # 对数据进行缩放和旋转
    y = np.array([0] * n_samlpes_per_class + [1] * n_samlpes_per_class)
    return X, y


if __name__ == "__main__":
    X, y = load_data()

    plt.figure(figsize=[18, 6])
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker=".")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker=".")

    lda = LDA(1)
    lda.fit(X, y)
    Z = lda(X)

    plt.subplot(1, 3, 2)
    plt.title("LDA")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(Z[y == 0, 0], np.zeros([500]), marker=".")
    plt.scatter(Z[y == 1, 0], np.zeros([500]), marker=".")

    # 和PCA对比
    pca = PCA(1)
    Z = pca(X)
    plt.subplot(1, 3, 3)
    plt.title("PCA")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(Z[y == 0, 0], np.zeros([500]), marker=".")
    plt.scatter(Z[y == 1, 0], np.zeros([500]), marker=".")

    plt.show()

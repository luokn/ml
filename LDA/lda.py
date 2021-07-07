# -*- coding: utf-8 -*-
# @Date  : 2020/5/31
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt


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
        L, U = LA.eig(LA.inv(S_W) @ S_B)  # 计算 S_W^{-1} S_B 的特征值与特征向量
        topk = np.argsort(L)[::-1][:self.k]  # 按照特征值降序排列，取前K大特征值
        self.W = U[:, topk]  # 选择topk对应的特征向量

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


def load_data():
    theta = np.pi / 4
    scale = np.array([[2, 0], [0, .8]])  # 缩放
    rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # 旋转
    x = np.stack([
        np.random.randn(500, 2) + [0, 2],
        np.random.randn(500, 2) - [0, 2]
    ])  @ scale @ rotate
    y = np.stack([np.full([500], 0), np.full([500], 1)])
    return x, y


if __name__ == "__main__":
    x, y = load_data()
    plt.figure(figsize=[18, 6])
    plt.subplot(1, 3, 1)
    plt.title('Real')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(x[0, :, 0], x[0, :, 1], color='r', marker='.')
    plt.scatter(x[1, :, 0], x[1, :, 1], color='g', marker='.')

    x, y = x.reshape(-1, 2), y.flatten()

    lda = LDA(1)
    lda.fit(x, y)
    z = lda(x)
    plt.subplot(1, 3, 2)
    plt.title('LDA')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(z[:500, 0], np.zeros([500]), color='r', marker='.')
    plt.scatter(z[500:, 0], np.zeros([500]), color='g', marker='.')

    # 和PCA对比
    pca = PCA(1)
    z = pca(x)
    plt.subplot(1, 3, 3)
    plt.title('PCA')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(z[:500, 0], np.zeros([500]), color='r', marker='.')
    plt.scatter(z[500:, 0], np.zeros([500]), color='g', marker='.')
    plt.show()

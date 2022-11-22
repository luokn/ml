#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File   : pca.py
# @Data   : 2020/5/20
# @Author : Luo Kun
# @Contact: luokun485@gmail.com

import numpy as np
from matplotlib import pyplot as plt


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


def load_data(n_samples_per_class=200):
    X = np.concatenate([
        np.random.randn(n_samples_per_class, 2) + np.array([2, 0]),
        np.random.randn(n_samples_per_class, 2),
        np.random.randn(n_samples_per_class, 2) + np.array([-2, 0]),
    ])
    theta = np.pi / 4  # 逆时针旋转45°
    scale = np.diag([1.2, 0.5])  # 缩放矩阵
    rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # 旋转矩阵
    return X @ scale @ rotate.T


if __name__ == "__main__":
    X = load_data()

    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    for x in X.reshape(3, -1, 2):
        plt.scatter(x[:, 0], x[:, 1], marker=".")

    # 不降维
    Y = PCA(2)(X)
    plt.subplot(1, 3, 2)
    plt.title("PCA 2D")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    for y in Y.reshape(3, -1, 2):
        plt.scatter(y[:, 0], y[:, 1], marker=".")

    # 降为1维
    Z = PCA(1)(X)
    plt.subplot(1, 3, 3)
    plt.title("PCA 1D")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    for z in Z.reshape(3, -1):
        plt.scatter(z, np.zeros_like(z), marker=".")

    plt.show()

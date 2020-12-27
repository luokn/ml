# -*- coding: utf-8 -*-
# @Date  : 2020/11/3
# @Author: Luokun
# @Email : olooook@outlook.com


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class LLE:
    """
    Locally Linear Embedding(局部线性嵌入)
    """

    def __init__(self, k: int, d: int):
        """
        Args:
            k (int): 选取的k近邻数
            d (int): 降维输出维度
        """
        self.k, self.d = k, d

    def transform(self, X: np.array):
        """
        Args:
            X (np.array): shape为 M × c
        Returns
            Y (np.array): shape为 M × d
        """
        W = np.zeros([len(X), len(X)])  # M × M
        for i, x in enumerate(X):
            topk = np.linalg.norm(x - X, axis=1).argsort()[1:self.k + 1]  # 找出距离最近的k个点的下标，排除自身
            S = x - X[topk]  # M × k
            S = S.T @ S  # S = (x_i - N_i)^T (x_i - N_i) => k × k
            w = np.linalg.inv(S).sum(axis=1)  # 计算w
            W[i, topk] = w / w.sum()  # 归一化并扩展到所有点上
        M = np.eye(len(X)) - W
        M @= M.T  # M = (I - W)(I - W) ^ T
        L, U = np.linalg.eig(M)  # 对M进行特征值分解
        topd = L.argsort()[1:self.d + 1]  # 选取前d个非0特征值
        np.tile()
        return U[topd].T  # 降维后的数据为选取的特征值对应的特征向量


def make_swiss_roll(n_samples=100, noise=0.0, shuffle=True):
    t = 1.5 * np.pi * (1 + 2 * np.linspace(0, 1, n_samples))
    x = np.stack([
        t * np.cos(t),
        t * np.sin(t),
        100 * np.random.rand(n_samples),
    ], axis=-1) + noise * np.random.randn(n_samples, 3)  # N × (x, y, z)
    c = np.stack([
        np.linspace(0.2, 1, n_samples),
        np.linspace(1, 0.2, n_samples),
        np.repeat(.5, n_samples)
    ], axis=-1)  # N × (r, g, b)
    if shuffle:
        index = np.arange(n_samples).astype(np.int)
        np.random.shuffle(index)
        x, c = x[index], c[index]
    return x, c


if __name__ == "__main__":
    n_samples = 10000
    lle = LLE(3, 1)
    x, c = make_swiss_roll(n_samples, 0.1)
    ax = Axes3D(plt.figure(figsize=(8, 8)))
    ax.scatter(*x.T, c=c, marker='.')
    plt.show()
    # Y = lle.transform(X)
    # plt.figure(figsize=(8, 8))
    # plt.scatter(Y[:, 0], np.repeat(0, n_samples), c=C)
    # plt.show()

# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

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
        x_norm = X - X.mean(axis=0)  # 去中心化
        L, U = np.linalg.eig(x_norm.T @ x_norm)  # 对协方差矩阵进行特征值分解
        topk = np.argsort(L)[::-1][:self.k]  # 找出前K大特征值对应的索引
        return x_norm @ U[:, topk]  # 将去中心化的X乘以前K大特征值对应的特征向量


def load_data():
    x = np.concatenate([np.random.randn(200, 2) + np.array([2, 0]),
                        np.random.randn(200, 2),
                        np.random.randn(200, 2) + np.array([-2, 0])])
    theta = np.pi / 4  # 逆时针旋转45°
    scale = np.diag([1.2, .6])  # 缩放矩阵
    rotate = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])  # 旋转矩阵
    return x @ scale @ rotate.T


if __name__ == '__main__':
    x = load_data()
    x0, x1, x2 = x.reshape(3, -1, 2)
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    plt.title('Real')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(x0[:, 0], x0[:, 1], color='r', marker='.')
    plt.scatter(x1[:, 0], x1[:, 1], color='g', marker='.')
    plt.scatter(x2[:, 0], x2[:, 1], color='b', marker='.')

    # 不降维
    y = PCA(2)(x)
    y0, y1, y2 = y.reshape(3, -1, 2)
    plt.subplot(1, 3, 2)
    plt.title('PCA 2D')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(y0[:, 0], y0[:, 1], color='r', marker='.')
    plt.scatter(y1[:, 0], y1[:, 1], color='g', marker='.')
    plt.scatter(y2[:, 0], y2[:, 1], color='b', marker='.')

    # 降为1维
    z = PCA(1)(x)
    z0, z1, z2 = z.reshape(3, -1)
    plt.subplot(1, 3, 3)
    plt.title('PCA 1D')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(z0, np.zeros([200]), color='r', marker='.')
    plt.scatter(z1, np.zeros([200]), color='g', marker='.')
    plt.scatter(z2, np.zeros([200]), color='b', marker='.')
    plt.show()

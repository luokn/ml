# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

import matplotlib.pyplot as plt
import numpy as np


class PCA:
    """
    Principal Components Analysis(主成分分析)
    """

    def __init__(self, k: int):
        """
        :param k: 保留的主成因个数
        """
        self.k = k

    def __call__(self, X: np.ndarray):
        X_norm = X - X.mean(axis=0)  # 去中心化
        L, U = np.linalg.eig(X_norm.T @ X_norm)  # 对协方差矩阵进行特征值分解
        topk = np.argsort(L)[::-1][:self.k]  # 找出特征值中前K大特征对应的索引
        return X_norm @ U[:, topk]  # 将去中心化矩阵乘以前K大特征对应的特征向量


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
    z = x.reshape(3, -1, 2)

    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    plt.title('Real')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(z[0, :, 0], z[0, :, 1], color='r', marker='.')
    plt.scatter(z[1, :, 0], z[1, :, 1], color='g', marker='.')
    plt.scatter(z[2, :, 0], z[2, :, 1], color='b', marker='.')

    # 不降维
    x_2d = PCA(2)(x)
    z = x_2d.reshape(3, -1, 2)
    plt.subplot(1, 3, 2)
    plt.title('PCA 2D')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(z[0, :, 0], z[0, :, 1], color='r', marker='.')
    plt.scatter(z[1, :, 0], z[1, :, 1], color='g', marker='.')
    plt.scatter(z[2, :, 0], z[2, :, 1], color='b', marker='.')

    # 降为1维
    x_1d = PCA(1)(x)
    z = x_1d.reshape(3, -1)
    plt.subplot(1, 3, 3)
    plt.title('PCA 1D')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(z[0], np.zeros([200]), color='r', marker='.')
    plt.scatter(z[1], np.zeros([200]), color='g', marker='.')
    plt.scatter(z[2], np.zeros([200]), color='b', marker='.')
    plt.show()

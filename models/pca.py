# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
import matplotlib.pyplot as plt


def pca(X: np.ndarray, K: int):
    """
    Principal Components Analysis
    """
    Y = X - X.mean(axis=0)
    L, U = np.linalg.eig(Y.T @ Y)
    topk = np.argsort(L)[::-1][:K]
    return Y @ U[:, topk]


"""
----------------------TEST----------------------
"""


def plot_scatter(points, title):
    p = points.reshape(3, -1, 2)
    plt.scatter(p[0, :, 0], p[0, :, 1], color='r', marker='.')
    plt.scatter(p[1, :, 0], p[1, :, 1], color='g', marker='.')
    plt.scatter(p[2, :, 0], p[2, :, 1], color='b', marker='.')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.title(title)
    plt.show()


def test_pca():
    scale = np.diag([1.5, .5])
    theta = np.pi / 4
    rotate = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    x = np.random.randn(3, 100, 2)
    x[1] += np.array([-2, 0])
    x[2] += np.array([2, 0])

    x = x.reshape(300, 2)
    x = x @ scale @ rotate.T

    # plot values before PCA
    plot_scatter(x, 'Before PCA')

    x = pca(x, 2)
    plot_scatter(x, 'PCA 2D')

    x = pca(x, 1)
    plot_scatter(np.concatenate([x, np.zeros([len(x), 1])], axis=1), 'PCA 1D')


if __name__ == '__main__':
    test_pca()

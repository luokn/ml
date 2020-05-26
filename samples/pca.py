# -*- coding: utf-8 -*-
# @Date  : 2020/5/24
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
import matplotlib.pyplot as plt

from models.pca import PCA


def test_pca():
    x = np.random.randn(3, 200, 2)
    x[1] += np.array([-2, 0])
    x[2] += np.array([2, 0])

    scale = np.diag([1.2, .6])  # 缩放矩阵
    theta = np.pi / 4  # 逆时针旋转45°
    rotate = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])  # 旋转矩阵

    x = x.reshape(-1, 2)
    x = x @ scale @ rotate.T

    plot_scatter(x.reshape(3, -1, 2), 'Before PCA')

    # 不降维
    x_2d = PCA(2).transform(x)
    plot_scatter(x_2d.reshape(3, -1, 2), 'PCA 2D')

    # 降为1维
    x_1d = PCA(1).transform(x)
    plot_scatter(np.concatenate([x_1d.reshape(3, -1, 1), np.zeros([3, 200, 1])], axis=-1), 'PCA 1D')


def plot_scatter(xys, title):
    plt.figure(figsize=[8, 8])
    for xy, color in zip(xys, ['r', 'g', 'b']):
        plt.scatter(xy[:, 0], xy[:, 1], color=color, marker='.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    test_pca()

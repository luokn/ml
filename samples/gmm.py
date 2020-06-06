# -*- coding: utf-8 -*-
# @Date  : 2020/6/5
# @Author: Luokun
# @Email : olooook@outlook.com

import sys
from os.path import dirname, abspath

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(dirname(dirname(abspath(__file__))))


def test_gmm():
    from models.gmm import GMM
    x = np.stack([
        np.random.multivariate_normal(mean=[5, 0], cov=[[2, 0], [0, 2]], size=[1000]),
        np.random.multivariate_normal(mean=[0, 5], cov=[[2, 0], [0, 1]], size=[1000])
    ])
    plot_scatter(x, 'Real')
    x = x.reshape(-1, 2)

    gmm = GMM(2, max_iter=1000)
    gmm.fit(x)

    pred = gmm.predict(x)
    plot_scatter([x[pred == i] for i in [0, 1]], 'Pred')


def plot_scatter(xys, title):
    plt.figure(figsize=[8, 8])
    for xy, color in zip(xys, ['r', 'g', 'b']):
        plt.scatter(xy[:, 0], xy[:, 1], color=color, marker='.')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    test_gmm()

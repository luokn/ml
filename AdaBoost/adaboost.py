# -*- coding: utf-8 -*-
# @Date  : 2020/5/27
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
from matplotlib import pyplot as plt


class AdaBoost:
    """
    Adaptive Boosting(自适应提升算法)
    """

    def __init__(self, n_estimators: int, lr=0.01, eps=1e-5):
        """
        Args:
            n_estimators (int): 弱分类器个数
            lr (float, optional): 学习率. Defaults to 0.01.
            eps (float, optional): 误差下限. Defaults to 1e-5.
        """
        self.n_estimators, self.lr, self.eps = n_estimators, lr, eps
        self.estimators = []  # 弱分类器

    def fit(self, X: np.ndarray, Y: np.ndarray):
        weights = np.full([len(X)], 1 / len(X))  # 样本权重
        for _ in range(self.n_estimators):
            estimator = WeakEstimator(lr=self.lr)
            error = estimator.fit(X, Y, weights)  # 带权重训练弱分类器
            if error < self.eps:  # 误差达到下限，提前停止迭代
                break
            alpha = np.log((1 - error) / error) / 2  # 更新弱分类器权重
            weights *= np.exp(-alpha * Y * estimator(X))  # 更新样本权重
            weights /= np.sum(weights)  # 除以规范化因子
            self.estimators += [(alpha, estimator)]  # 添加此弱分类器

    def __call__(self, X: np.ndarray):
        pred = sum((alpha * estimator(X) for alpha, estimator in self.estimators))
        return np.where(pred > 0, 1, -1)


class WeakEstimator:  # 弱分类器, 一阶决策树
    def __init__(self, lr: float):
        self.lr = lr
        # 划分特征、划分阈值，符号{-1，1}
        self.feature, self.threshold, self.sign = None, None, None

    def fit(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray):
        error = float('inf')  # 最小带权误差
        for feature, x in enumerate(X.T):
            for threshold in np.arange(np.min(x) - self.lr, np.max(x) + self.lr, self.lr):
                for sign in [1, -1]:
                    e = np.sum(weights[np.where(x > threshold, sign, -sign) != Y])  # 取分类错误的样本权重求和
                    if e < error:
                        self.feature, self.threshold, self.sign, error = feature, threshold, sign, e
        return error

    def __call__(self, X: np.ndarray):
        return np.where(X[:, self.feature] > self.threshold, self.sign, -self.sign)


def load_data():
    x = np.stack([
        np.random.randn(500, 2) + np.array([2, 0]),
        np.random.randn(500, 2) + np.array([0, 2])
    ])
    y = np.stack([np.full([500], -1), np.full([500], 1)])
    return x, y


if __name__ == '__main__':
    x, y = load_data()
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title('Real')
    plt.scatter(x[0, :, 0], x[0, :, 1], color='r', marker='.')
    plt.scatter(x[1, :, 0], x[1, :, 1], color='g', marker='.')

    x, y = x.reshape(-1, 2), y.flatten()
    adaboost = AdaBoost(50)
    adaboost.fit(x, y)
    pred = adaboost(x)
    acc = np.sum(pred == y) / len(pred)
    print(f'Accuracy = {100 * acc:.2f}%')

    x0, x1 = x[pred == -1], x[pred == 1]
    plt.subplot(1, 2, 2)
    plt.title('Pred')
    plt.scatter(x0[:, 0], x0[:, 1], color='r', marker='.')
    plt.scatter(x1[:, 0], x1[:, 1], color='g', marker='.')
    plt.show()

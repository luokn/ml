# -*- coding: utf-8 -*-
# @Date  : 2020/5/27
# @Author: Luokun
# @Email : olooook@outlook.com

import matplotlib.pyplot as plt
import numpy as np


class AdaBoost:
    """
    Adaptive Boosting(自适应提升算法)
    """

    def __init__(self, n_estimators: int, lr=0.01, eps=1e-5):
        """
        :param n_estimators: 弱分类器个数
        :param eps: 误差阈值
        """
        self.n_estimators, self.lr, self.eps = n_estimators, lr, eps
        self.estimators = []
        self.alpha = np.empty([n_estimators])  # 弱分类器权重

    def fit(self, X: np.ndarray, Y: np.ndarray):
        weights = np.full([len(X)], 1 / len(X))  # 样本权重
        estimators = [WeakEstimator(f, self.lr) for f in range(X.shape[1])]  # 所有特征的弱分类器
        predictions = np.zeros([len(estimators), len(Y)])  # 弱分类器输出
        for i, estimator in enumerate(estimators):
            estimator.fit(X, Y)  # 逐特征训练弱分类器
            predictions[i] = estimator(X)  # 记录所有弱分类器输出
        for k in range(self.n_estimators):
            errors = np.array([
                np.sum(np.where(pred == Y, 0, weights)) for pred in predictions
            ])  # 计算每一个弱分类器的带权重误差
            idx = np.argmin(errors).item()  # 选择最小误差
            if errors[idx] < self.eps:  # 误差达到阈值，停止
                break
            self.estimators.append(estimators[idx])  # 添加弱分类器
            self.alpha[k] = .5 * np.log((1 - errors[idx]) / errors[idx])  # 更新弱分类器权重
            exp_res = np.exp(-self.alpha[k] * Y * predictions[idx])
            weights *= exp_res / (weights @ exp_res)  # 更新样本权重

    def __call__(self, X: np.ndarray):
        pred = np.array([
            alpha * estimator(X) for alpha, estimator in zip(self.alpha, self.estimators)
        ])
        pred = np.sum(pred, axis=0)
        return np.where(pred > 0, 1, -1)


class WeakEstimator:  # 弱分类器, 一阶决策树
    def __init__(self, feature: int, lr: float):
        self.feature, self.lr = feature, lr  # 划分特征、学习率
        self.div, self.sign = None, None  # 划分值、符号

    def fit(self, X: np.ndarray, Y: np.ndarray):
        x, max_corr = X[:, self.feature], 0
        for value in np.arange(x.min(), x.max() + self.lr, self.lr):
            pos_corr = np.sum(np.where(x > value, 1, -1) == Y)
            neg_corr = len(x) - pos_corr
            if pos_corr > max_corr:
                self.div, self.sign, max_corr = value, 1, pos_corr
            elif neg_corr > max_corr:
                self.div, self.sign, max_corr = value, -1, neg_corr

    def __call__(self, X: np.ndarray):
        return np.where(X[:, self.feature] > self.div, self.sign, -self.sign)


def load_data():
    x = np.stack([np.random.randn(200, 2) + np.array([-1, 0]),
                  np.random.randn(200, 2) + np.array([1, -1])])
    y = np.stack([np.full([200], -1), np.full([200], 1)])
    return x, y


if __name__ == '__main__':
    x, y = load_data()

    plt.figure(figsize=[10, 5])
    plt.subplot(1, 2, 1)
    plt.title('Real')
    plt.scatter(x[0, :, 0], x[0, :, 1], color='r', marker='.')
    plt.scatter(x[1, :, 0], x[1, :, 1], color='g', marker='.')

    x, y = x.reshape(-1, 2), y.flatten()
    ada_boost = AdaBoost(5)
    ada_boost.fit(x, y)
    pred = ada_boost(x)
    acc = np.sum(pred == y) / len(pred)
    print(f'Accuracy = {100 * acc:.2f}%')

    x0, x1 = x[pred == -1], x[pred == 1]
    plt.subplot(1, 2, 2)
    plt.title('Pred')
    plt.scatter(x0[:, 0], x0[:, 1], color='r', marker='.')
    plt.scatter(x1[:, 0], x1[:, 1], color='g', marker='.')
    plt.show()

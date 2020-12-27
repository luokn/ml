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
        self._X, self._Y, self._weights = None, None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self._X, self._Y, self._weights = X, Y, np.full([len(X)], 1 / len(X))
        estimators = [self._WeakEstimator(f, self.lr) for f in range(X.shape[1])]  # 所有特征的弱分类器
        predictions = np.zeros([len(estimators), len(Y)])  # 弱分类器输出
        for i, estimator in enumerate(estimators):
            estimator.fit(X, Y)  # 逐特征训练弱分类器
            predictions[i] = estimator(X)  # 记录所有弱分类器输出
        for k in range(self.n_estimators):
            errors = np.array([
                np.sum(np.where(pred == Y, 0, self._weights)) for pred in predictions
            ])  # 计算每一个弱分类器的带权重误差
            idx = np.argmin(errors).item()  # 选择最小误差
            if errors[idx] < self.eps:  # 误差达到阈值，停止
                break
            self.estimators.append(estimators[idx])  # 添加弱分类器
            self._update_alpha(k, errors[idx])  # 更新弱分类器权重
            self._update_weights(k, predictions[idx])  # 更新样本权重
        self._X, self._Y, self._weights = None, None, None

    def predict(self, X: np.ndarray):
        pred = np.array([
            alpha * estimator(X) for alpha, estimator in zip(self.alpha, self.estimators)
        ])
        pred = np.sum(pred, axis=0)
        return np.where(pred > 0, 1, -1)

    def _update_alpha(self, k: int, e: float):  # 更新弱分类器权重
        self.alpha[k] = .5 * np.log((1 - e) / e)

    def _update_weights(self, k: int, pred: np.ndarray):  # 更新样本权重
        exp_res = np.exp(-self.alpha[k] * self._Y * pred)
        self._weights = self._weights * exp_res / (self._weights @ exp_res)

    class _WeakEstimator:  # 弱分类器, 一阶决策树
        def __init__(self, feature: int, lr: float):
            self.feature, self.lr = feature, lr  # 划分特征、学习率
            self.div, self.sign = None, None  # 划分值、符号

        def fit(self, X: np.ndarray, Y: np.ndarray):
            Xf, max_corr = X[:, self.feature], 0
            for value in np.arange(Xf.min(), Xf.max() + self.lr, self.lr):
                pos_corr = np.sum(np.where(Xf > value, 1, -1) == Y)
                neg_corr = len(Xf) - pos_corr
                if pos_corr > max_corr:
                    self.div, self.sign, max_corr = value, 1, pos_corr
                elif neg_corr > max_corr:
                    self.div, self.sign, max_corr = value, -1, neg_corr

        def __call__(self, X: np.ndarray):
            return np.where(X[:, self.feature] > self.div, self.sign, -self.sign)


def load_data():
    x, y = np.random.randn(2, 200, 2), np.zeros([2, 200])
    x[0] += np.array([-1, 0])
    x[1] += np.array([1, -1])
    y[0] = -1
    y[1] = 1
    return x, y


def plot_scatter(xys, title):
    plt.figure(figsize=(10, 10))
    for xy, color in zip(xys, ['r', 'g', 'b']):
        plt.scatter(xy[:, 0], xy[:, 1], color=color)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    x, y = load_data()
    plot_scatter(x, 'Real')

    x = x.reshape(-1, 2)
    y = y.flatten()

    # train
    ab = AdaBoost(5)
    ab.fit(x, y)

    pred = ab.predict(x)
    plot_scatter([x[pred == i] for i in [-1, 1]], 'Pred')

    # print accuracy
    acc = np.sum(pred == y) / len(pred)
    print(f'Acc = {100 * acc:.2f}%')

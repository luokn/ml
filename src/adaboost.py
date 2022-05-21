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
        self.estimators = []  # 弱分类器
        self.n_estimators, self.lr, self.eps = n_estimators, lr, eps

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X (np.ndarray): ...
            Y (np.ndarray): ...
            lr (float, optional): 学习率. Defaults to 0.01.
            eps (float, optional): 误差下限. Defaults to 1e-5.
        """
        weights = np.full(len(X), 1 / len(X))  # 样本权重
        for _ in range(self.n_estimators):
            estimator = WeakEstimator(lr=self.lr)
            error = estimator.fit(X, y, weights=weights)  # 带权重训练弱分类器
            if error < self.eps:  # 误差达到下限，提前停止迭代
                break
            alpha = np.log((1 - error) / error) / 2  # 更新弱分类器权重
            weights *= np.exp(-alpha * y * estimator(X))  # 更新样本权重
            weights /= np.sum(weights)  # 除以规范化因子
            self.estimators += [(alpha, estimator)]  # 添加此弱分类器

    def __call__(self, X: np.ndarray) -> np.ndarray:
        y_pred = sum((alpha * estimator(X) for alpha, estimator in self.estimators))
        return np.where(y_pred > 0, 1, -1)


class WeakEstimator:  # 弱分类器, 一阶决策树
    def __init__(self, lr: float):
        self.lr, self.feature, self.threshold, self.sign = lr, None, None, None  # 划分特征、划分阈值，符号{-1，1}

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        min_error = float("inf")  # 最小带权误差
        for f, x in enumerate(X.T):
            for threshold in np.arange(np.min(x) - self.lr, np.max(x) + self.lr, self.lr):
                for sign in [1, -1]:
                    error = np.sum(weights[np.where(x > threshold, sign, -sign) != y])  # 取分类错误的样本权重求和
                    if error < min_error:
                        self.feature, self.threshold, self.sign, min_error = f, threshold, sign, error
        return min_error

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.where(X[:, self.feature] > self.threshold, self.sign, -self.sign)


def load_data(n_samples_per_class=500):
    X = np.concatenate(
        [
            np.random.randn(n_samples_per_class, 2) + np.array([2, 0]),
            np.random.randn(n_samples_per_class, 2) + np.array([0, 2]),
        ]
    )
    y = np.array([1] * n_samples_per_class + [-1] * n_samples_per_class)

    training_set, test_set = np.split(np.random.permutation(len(X)), [int(len(X) * 0.6)])

    return X, y, training_set, test_set


if __name__ == "__main__":
    X, y, training_set, test_set = load_data()

    X_neg, X_pos = X[y == -1], X[y == 1]
    plt.figure("AdaBoost", figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.scatter(X_neg[:, 0], X_neg[:, 1], marker=".")
    plt.scatter(X_pos[:, 0], X_pos[:, 1], marker=".")

    adaboost = AdaBoost(50)
    adaboost.fit(X[training_set], y[training_set])
    y_pred = adaboost(X)
    acc = np.sum(y_pred[test_set] == y[test_set]) / len(test_set)
    print(f"Accuracy = {100 * acc:.2f}%")

    X_neg, X_pos = X[y_pred == -1], X[y_pred == 1]
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.scatter(X_neg[:, 0], X_neg[:, 1], marker=".")
    plt.scatter(X_pos[:, 0], X_pos[:, 1], marker=".")

    plt.show()

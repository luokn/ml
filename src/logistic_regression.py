# -*- coding: utf-8 -*-
# @Date  : 2020/5/21
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression:
    """
    Logistic regression classifier(逻辑斯蒂回归分类器)
    """

    def __init__(self, input_dim: int, lr=5e-4):
        """
        Args:
            input_dim (int): 特征维度
            lr (float): 学习率, 默认为5e-4
        """
        self.weights = np.random.randn(input_dim + 1)  # 随机初始化参数
        self.lr = lr  # 学习率

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_pad = pad(X)  # 为X填充1作为偏置
        pred = sigmoid(X_pad @ self.weights)  # 计算预测值
        grad = X_pad.T @ (pred - y) / len(pred)  # 计算梯度
        self.weights -= self.lr * grad  # 沿负梯度更新参数

    def __call__(self, X: np.ndarray) -> np.ndarray:
        y_pred = sigmoid(pad(X) @ self.weights)  # 计算预测值
        return np.where(y_pred > 0.5, 1, 0)  # 将(0, 1)之间分布的概率转化为{0, 1}标签


def pad(x):
    return np.concatenate([x, np.ones([len(x), 1])], axis=1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data(n_samples_per_class=500):
    X = np.concatenate(
        [
            np.random.randn(n_samples_per_class, 2) + np.array([1, -1]),
            np.random.randn(n_samples_per_class, 2) + np.array([-1, 1]),
        ]
    )
    y = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

    training_set, test_set = np.split(np.random.permutation(len(X)), [int(len(X) * 0.6)])
    return X, y, training_set, test_set


def train_logistic_regression(model, X, y, epochs=100, batch_size=32):
    indices = np.arange(len(X))
    for _ in range(epochs):
        np.random.shuffle(indices)
        for i in range(batch_size, len(X) + 1, batch_size):
            model.fit(X[indices[i - batch_size : i]], y[indices[i - batch_size : i]])


if __name__ == "__main__":
    X, y, training_set, test_set = load_data()

    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker=".")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker=".")

    logistic_regression = LogisticRegression(2)
    train_logistic_regression(logistic_regression, X, y, epochs=500)
    y_pred = logistic_regression(X)
    acc = np.sum(y_pred[test_set] == y[test_set]) / len(test_set)
    print(f"Accuracy = {100 * acc:.2f}%")

    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], marker=".")
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], marker=".")

    w = logistic_regression.weights
    a, b = -w[0] / w[1], -w[2] / w[1]
    line_x = np.linspace(-4, 4, 400)
    line_y = a * line_x + b
    plt.plot(line_x, line_y, lw=1)

    plt.fill_between(line_x, np.full(400, -4), line_y, alpha=0.1)
    plt.fill_between(line_x, np.full(400, +4), line_y, alpha=0.1)

    plt.show()

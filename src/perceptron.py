# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
from matplotlib import pyplot as plt


class Perceptron:
    """
    Perceptron classifier(感知机分类器)
    """

    def __init__(self, input_dim: int, lr=5e-4):
        """
        Args:
            input_dim (int): 输入特征维度
        """
        self.weights = np.random.randn(input_dim + 1)  # 权重
        self.lr = lr  # 学习率

    def fit(self, X: np.ndarray, y: np.ndarray):
        for x, y in zip(pad(X), y):
            if y * (x @ self.weights) <= 0:  # 分类错误, y 与 wx + b 符号不同
                neg_grad = x * y  # 计算weights的负梯度
                self.weights += self.lr * neg_grad  # 沿负梯度方向更新weights

    def __call__(self, X: np.ndarray) -> np.ndarray:
        y_pred = pad(X) @ self.weights
        return np.where(y_pred > 0, 1, -1)


def pad(x):
    return np.concatenate([x, np.ones([len(x), 1])], axis=1)


def load_data(n_samples_per_class=500):
    X = np.concatenate(
        [
            np.random.randn(n_samples_per_class, 2) + np.array([1, -1]),
            np.random.randn(n_samples_per_class, 2) + np.array([-1, 1]),
        ]
    )
    y = np.array([-1] * n_samples_per_class + [1] * n_samples_per_class)

    training_set, test_set = np.split(np.random.permutation(len(X)), [int(len(X) * 0.6)])

    return X, y, training_set, test_set


def train_perceptron(model, X, y, epochs=100):
    indices = np.arange(len(X))
    for _ in range(epochs):
        np.random.shuffle(indices)
        model.fit(X[indices], y[indices])


if __name__ == "__main__":
    X, y, training_set, test_set = load_data()

    X_neg, X_pos = X[y == -1], X[y == 1]
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], marker=".")
    plt.scatter(X_pos[:, 0], X_pos[:, 1], marker=".")

    perceptron = Perceptron(input_dim=2)
    train_perceptron(perceptron, X[training_set], y[training_set], epochs=100)
    y_pred = perceptron(X)
    acc = np.sum(y_pred[test_set] == y[test_set]) / len(test_set)
    print(f"Accuracy = {100 * acc:.2f}%")

    X_neg, X_pos = X[y_pred == -1], X[y_pred == 1]
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], marker=".")
    plt.scatter(X_pos[:, 0], X_pos[:, 1], marker=".")

    w = perceptron.weights
    a, b = -w[0] / w[1], -w[2] / w[1]
    line_x = np.linspace(-5, 5, 100)
    line_y = a * line_x + b
    plt.plot(line_x, line_y, c="b", lw=1)
    plt.show()

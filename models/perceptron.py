import numpy as np


class PerceptronClassifier:
    """
    Perceptron classifier(感知机分类器, 原始形式)
    """

    def __init__(self, input_dim: int, lr=0.01):
        self.input_dim, self.lr = input_dim, lr
        self.weights = np.random.randn(input_dim + 1)  # 权重

    def fit(self, X: np.ndarray, Y: np.ndarray):
        x_pad = pad(X)
        err = Y * (x_pad @ self.weights) <= 0  # err表示是否分类错误
        if not np.any(err):  # 没有任何错误
            return  # 直接返回
        x_err, y_err = x_pad[err], Y[err]  # 选择分类错误的X与Y
        grad = np.mean(y_err.reshape(-1, 1) * x_err, axis=0)  # 计算weights梯度
        self.weights -= self.lr * grad  # 更新weights梯度

    def predict(self, X: np.ndarray):
        return sign(pad(X) @ self.weights)


def pad(x):
    return np.concatenate([x, np.ones([len(x), 1])], axis=1)


def sign(x):
    y = np.ones_like(x)
    y[x < 0] = -1
    return y

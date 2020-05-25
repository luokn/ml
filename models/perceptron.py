import numpy as np


class PerceptronClassifier:
    """
    Perceptron classifier(感知机分类器, 原始形式)
    """

    def __init__(self, input_dim: int, lr=0.001):
        self.input_dim, self.lr = input_dim, lr
        self.weights = np.random.randn(input_dim)  # 权重向量
        self.bias = 0  # 偏置

    def fit(self, X: np.ndarray, Y: np.ndarray):
        err = Y * (X @ self.weights + self.bias) <= 0  # err表示是否分类错误
        if not np.any(err):  # 没有任何错误
            return  # 不再计算梯度调整参数
        x_err, y_err = X[err], Y[err]  # 选择分类错误的X与Y
        grad_w = np.mean(y_err @ x_err, axis=1)  # 计算weights梯度
        grad_b = np.mean(y_err)  # 计算bias梯度
        self.weights -= self.lr * grad_w  # 更新weights梯度
        self.bias -= self.lr * grad_b  # 更新bias梯度

    def predict(self, X: np.ndarray):
        return sign(X @ self.weights + self.bias)


def sign(x):
    y = np.zeros_like(x)
    y[x >= 0] = 1
    return y

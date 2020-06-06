import numpy as np


class Perceptron:
    """
    Perceptron classifier(感知机分类器)
    """

    def __init__(self, input_dim: int, lr: float):
        """
        :param input_dim: 输入特征长度
        :param lr: 学习率
        """
        self.weights = np.random.randn(input_dim + 1)  # 权重
        self.input_dim, self.lr = input_dim, lr

    def fit(self, X: np.ndarray, Y: np.ndarray):
        for x, y in zip(pad(X), Y):
            if y * (x @ self.weights) <= 0:  # 分类错误, y 与 wx + b 符号不同
                neg_grad = x * y  # 计算weights的负梯度
                self.weights += self.lr * neg_grad  # 沿负梯度方向更新weights

    def predict(self, X: np.ndarray):
        pred = self._pad(X) @ self.weights
        return np.where(pred > 0, 1, -1)

    @staticmethod
    def _pad(x):
        return np.concatenate([x, np.ones([len(x), 1])], axis=1)

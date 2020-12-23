import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    """
    Perceptron classifier(感知机分类器)
    """

    def __init__(self, input_dim: int, lr: float):
        """
        :param input_dim: 输入特征维度
        :param lr: 学习率
        """
        self.weights = np.random.randn(input_dim + 1)  # 权重
        self.input_dim, self.lr = input_dim, lr

    def fit(self, X: np.ndarray, Y: np.ndarray):
        for x, y in zip(self._pad(X), Y):
            if y * (x @ self.weights) <= 0:  # 分类错误, y 与 wx + b 符号不同
                neg_grad = x * y  # 计算weights的负梯度
                self.weights += self.lr * neg_grad  # 沿负梯度方向更新weights

    def predict(self, X: np.ndarray):
        pred = self._pad(X) @ self.weights
        return np.where(pred > 0, 1, -1)

    @staticmethod
    def _pad(x):
        return np.concatenate([x, np.ones([len(x), 1])], axis=1)


def test_perceptron():
    x, y = np.random.randn(2, 500, 2), np.zeros([2, 500], dtype=int)
    x[0] += np.array([1, -1])
    x[1] += np.array([-1, 1])
    y[0] = -1
    y[1] = 1
    plot_scatter(x[0], x[1], 'Real')

    x = x.reshape(-1, 2)
    y = y.flatten()

    perceptron = Perceptron(input_dim=2, lr=1e-4)
    train_perceptron(perceptron, x, y, epochs=100)

    pred = perceptron.predict(x)
    plot_scatter_with_line(x[pred == -1], x[pred == 1], perceptron.weights, 'Pred')

    acc = np.sum(pred == y) / len(pred)
    print(f'Acc = {100 * acc:.2f}%')


def train_perceptron(model, x, y, epochs):
    indices = np.arange(len(x))
    for epoch in range(epochs):
        np.random.shuffle(indices)
        shf_x, shf_y = x[indices], y[indices]
        model.fit(shf_x, shf_y)


def plot_scatter(xy0, xy1, title):
    plt.figure(figsize=[8, 8])
    plt.scatter(xy0[:, 0], xy0[:, 1], color='r', marker='.')
    plt.scatter(xy1[:, 0], xy1[:, 1], color='g', marker='.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(title)
    plt.show()


def plot_scatter_with_line(xy0, xy1, weights, title):
    plt.figure(figsize=[8, 8])
    plt.scatter(xy0[:, 0], xy0[:, 1], color='r', marker='.')
    plt.scatter(xy1[:, 0], xy1[:, 1], color='g', marker='.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(title)

    # plot the dividing line
    ln_x = np.linspace(-5, 5, 100)
    ln_a = - weights[0] / weights[1]
    ln_b = - weights[2] / weights[1]
    ln_y = ln_a * ln_x + ln_b
    plt.plot(ln_x, ln_y, color='b', linewidth=1)
    plt.show()


if __name__ == '__main__':
    test_perceptron()

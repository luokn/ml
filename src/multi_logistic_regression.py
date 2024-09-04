import numpy as np
import random
from sklearn.datasets import load_iris


def create_data(batchsize: int = 149, train_rate: int = 0.8):
    batchsize = min(batchsize, 149)
    train_batch = int(train_rate * batchsize)
    iris = load_iris()
    X = iris.data
    y = np.reshape(iris.target, (-1, 1))
    shuffle_idxs = np.arange(batchsize)
    random.shuffle(shuffle_idxs)
    X = X[shuffle_idxs]
    y = y[shuffle_idxs]
    X_train = X[:train_batch]
    y_train = y[:train_batch]
    X_test = X[train_batch:]
    y_test = y[train_batch:]
    return X_train, y_train, X_test, y_test


class MultiLogisticRegression:
    def __init__(self, input_dim: int, output_dim: int, lr: float) -> None:
        self.i_d = input_dim
        self.o_d = output_dim
        self.lr = lr
        self.weight = np.zeros((input_dim + 1, output_dim))
        pass
    def fit(self, X_pad: np.ndarray, Y_unfold: np.ndarray):
        pred = self.softmax(X_pad @ self.weight)
        err = pred - Y_unfold
        grad = X_pad.T @ err
        self.weight -= self.lr * grad

    def train(self, X: np.ndarray, Y: np.ndarray, iter: int, batch_size: int):
        B = len(X)
        shuffle_idxs = np.arange(B)
        X_pad = self.pad(X)
        Y_unfold = np.eye(self.o_d)[np.round(np.reshape(Y, (-1,)).astype(np.int32))]
        for _ in range(iter):
            random.shuffle(shuffle_idxs)
            X_pad_shuffle = X_pad[shuffle_idxs]
            Y_unfold_shuffle = Y_unfold[shuffle_idxs]
            for i in range(B // batch_size):
                x_batch = X_pad_shuffle[batch_size * i : batch_size * (i + 1)]
                y_batch = Y_unfold_shuffle[batch_size * i : batch_size * (i + 1)]
                self.fit(x_batch, y_batch)
            if batch_size * (B // batch_size) < B:
                x_batch = X_pad_shuffle[batch_size * (i + 1) :]
                y_batch = Y_unfold_shuffle[batch_size * (i + 1) :]
                self.fit(x_batch, y_batch)

    def __call__(self, X: np.ndarray):
        pred = self.softmax(self.pad(X) @ self.weight)
        return np.reshape(np.argmax(pred, axis=1), (-1, 1))

    @staticmethod
    def pad(X: np.ndarray):
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    @staticmethod
    def softmax(Y_unfold: np.ndarray):
        Y_exp = np.exp(Y_unfold)
        return Y_exp / np.reshape(np.sum(Y_exp, axis=1), (-1, 1))  # numpy 广播用法


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = create_data()
    model = MultiLogisticRegression(4, 3, 1e-2)
    model.train(X_train, y_train, iter=100, batch_size=32)
    pred = model(X_test)
    mistakes = np.sum(np.where(np.abs(pred - y_test) > 0.5, 1, 0))
    print("Accuracy:{}%".format(100 * (1 - mistakes / len(pred))))

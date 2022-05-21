# -*- coding: utf-8 -*-
# @Date  : 2020/5/23
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np
from matplotlib import pyplot as plt


class LinearKernel:  # 线性核函数
    def __call__(self, a: np.ndarray, b: np.ndarray):
        return np.sum(a * b, axis=-1)


class PolyKernel:  # 多项式核函数
    def __call__(self, a: np.ndarray, b: np.ndarray):
        return (np.sum(a * b, axis=-1) + 1) ** 2


class RBFKernel:  # 高斯核函数
    def __init__(self, sigma):
        self.divisor = 2 * sigma**2

    def __call__(self, a: np.ndarray, b: np.ndarray):
        return np.exp(-np.sum((a - b) ** 2, axis=-1) / self.divisor)


class SVM:
    """
    Support Vector Machines(支持向量机)
    """

    def __init__(self, kernel="linear", C=1.0, iterations=100, tol=1e-3, sigma=1.0):
        """
        Args:
            kernel (str, optional): 核函数. Defaults to 'linear'.
            C (float, optional): 惩罚因子. Defaults to 1.0.
            iterations (int, optional): 最大迭代次数. Defaults to 100.
            tol (float, optional): 绝对误差限. Defaults to 1e-3.
            sigma (float, optional): 高斯核函数的sigma. Defaults to 1.0.
        """
        assert kernel in ["linear", "poly", "rbf"]

        if kernel == "linear":
            self.K = LinearKernel()  # 线性核函数
        if kernel == "poly":
            self.K = PolyKernel()  # 多项式核函数
        if kernel == "rbf":
            self.K = RBFKernel(sigma)  # 径向基核函数

        self.C, self.iterations, self.tol, self.alpha, self.b = C, iterations, tol, None, 0.0

        self.X, self.y = None, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y

        # 拉格朗日乘子
        self.alpha = np.ones([len(X)])
        for _ in range(self.iterations):
            # 此次迭代缓存的误差
            E = np.array([self.__E(i) for i in range(len(X))])

            # 外层循环，寻找第一个alpha
            for i1 in range(len(X)):

                # 计算误差(不使用E缓存)
                E1 = self.__E(i1)
                if E1 == 0 or self._satisfy_kkt(i1):
                    # 误差为0或满足KKT条件
                    continue

                # 大于0则选择最小,小于0选择最大的
                i2 = np.argmin(E) if E1 > 0 else np.argmax(E)  # 内层循环，寻找第二个alpha
                if i1 == i2:
                    continue
                E2 = self.__E(i2)
                x1, x2, y1, y2 = X[i1], X[i2], y[i1], y[i2]
                alpha1, alpha2 = self.alpha[i1], self.alpha[i2]
                k11, k22, k12 = self.K(x1, x1), self.K(x2, x2), self.K(x1, x2)

                # 计算剪切范围
                if y1 * y2 < 0:
                    L = max(0, alpha2 - alpha1)
                    H = min(self.C, self.C + alpha2 - alpha1)
                else:
                    L = max(0, alpha1 + alpha2 - self.C)
                    H = min(self.C, alpha1 + alpha2)
                if L == H:
                    continue
                eta = k11 + k22 - 2 * k12
                if eta <= 0:
                    continue

                # 计算新alpha
                alpha2_new = np.clip(alpha2 + y2 * (E1 - E2) / eta, L, H)
                alpha1_new = alpha1 + y1 * y2 * (alpha2 - alpha2_new)

                # 计算新b
                alpha2_delta, alpha1_delta = alpha2_new - alpha2, alpha1_new - alpha1
                b1_new = -E1 - y1 * k11 * alpha1_delta - y2 * k12 * alpha2_delta + self.b
                b2_new = -E2 - y1 * k12 * alpha1_delta - y2 * k22 * alpha2_delta + self.b

                # 更新参数
                self.alpha[i1] = alpha1_new
                self.alpha[i2] = alpha2_new
                if 0 < alpha1_new < self.C:
                    self.b = b1_new
                elif 0 < alpha2_new < self.C:
                    self.b = b2_new
                else:
                    self.b = (b1_new + b2_new) / 2

                # 更新误差缓存
                E[i1] = self.__E(i1)
                E[i2] = self.__E(i2)

    def __call__(self, X: np.ndarray):
        y_pred = np.array([self.__g(x) for x in X])
        return np.where(y_pred > 0, 1, -1)  # 将(-\infinity, \infinity)之间的分布转为{-1, +1}标签

    @property
    def support_vectors(self):  # 支持向量
        return self.X[self.alpha > 0]

    def __g(self, x):  # g(x) =\sum_{i=0}^N alpha_i y_i \kappa(x_i, x)
        return np.sum(self.alpha * self.y * self.K(self.X, x)) + self.b

    def __E(self, i):  # E_i = g(x_i) - y_i
        return self.__g(self.X[i]) - self.y[i]

    def _satisfy_kkt(self, i):  # 是否满足KKT条件
        g_i, y_i = self.__g(self.X[i]), self.y[i]
        if np.abs(self.alpha[i]) < self.tol:
            return g_i * y_i >= 1
        if np.abs(self.alpha[i]) > self.C - self.tol:
            return g_i * y_i <= 1
        return np.abs(g_i * y_i - 1) < self.tol


def load_data(n_samples_per_class=200):
    assert n_samples_per_class % 10 == 0, "n_samples_per_class must be divisible by 10"

    X_neg = np.random.randn(n_samples_per_class // 10, 10, 2)
    X_pos = np.random.randn(n_samples_per_class, 2)

    # 将负样本放置到圆环区域
    for i, theta in enumerate(np.linspace(0, 2 * np.pi, len(X_neg))):
        X_neg[i] += 5 * np.array([np.cos(theta), np.sin(theta)])

    X = np.concatenate([X_neg.reshape(-1, 2), X_pos])
    y = np.array([-1] * n_samples_per_class + [1] * n_samples_per_class)

    # 打乱索引，拆分训练集和测试集
    training_set, test_set = np.split(np.random.permutation(len(X)), [int(len(X) * 0.6)])

    return X, y, training_set, test_set


if __name__ == "__main__":
    X, y, training_set, test_set = load_data()

    X_neg, X_pos = X[y == -1], X[y == 1]
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], color="r", marker=".")
    plt.scatter(X_pos[:, 0], X_pos[:, 1], color="g", marker=".")

    svm = SVM(kernel="rbf", C=100, iterations=100, sigma=5)
    svm.fit(X[training_set], y[training_set])
    y_pred = svm(X)
    acc = np.sum(y_pred[test_set] == y[test_set]) / len(test_set)
    print(f"Accuracy = {100 * acc:.2f}%")

    X_neg, X_pos = X[y_pred == -1], X[y_pred == 1]
    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], color="r", marker=".")
    plt.scatter(X_pos[:, 0], X_pos[:, 1], color="g", marker=".")

    vectors = svm.support_vectors
    plt.subplot(1, 3, 3)
    plt.title("Support vectors")
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], color="r", marker=".")
    plt.scatter(X_pos[:, 0], X_pos[:, 1], color="g", marker=".")
    plt.scatter(vectors[:, 0], vectors[:, 1], color="b", marker=".")

    plt.show()

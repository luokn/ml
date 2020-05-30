# -*- coding: utf-8 -*-
# @Date  : 2020/5/23
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class LinearKernel:
    def __call__(self, a: np.ndarray, b: np.ndarray):
        return np.sum(a * b, axis=-1)


class PolyKernel:
    def __call__(self, a: np.ndarray, b: np.ndarray):
        return (np.sum(a * b, axis=-1) + 1) ** 2


class RBFKernel:
    def __init__(self, sigma):
        self.divisor = 2 * sigma ** 2

    def __call__(self, a: np.ndarray, b: np.ndarray):
        return np.exp(-np.sum((a - b) ** 2, axis=-1) / self.divisor)


class SVM:
    """
    Support Vector Machines(支持向量机)
    """

    def __init__(self, C=1.0, tol=1e-3, kernel='linear', max_iter=100, **kwargs):
        self.C = C  # 惩罚因子
        self.tol = tol  # 绝对误差限
        if kernel == 'linear':
            self.K = LinearKernel()  # 线性核函数
        if kernel == 'poly':
            self.K = PolyKernel()  # 多项式核函数
        if kernel == 'rbf':
            self.K = RBFKernel(kwargs['sigma'])  # 径向基核函数
        self.iterations = max_iter  # 最大迭代次数
        self.alpha, self.b = None, .0
        self.X, self.Y = None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X, self.Y = X, Y
        self.alpha = np.ones([len(X)], dtype=np.float)  # 拉格朗日乘子
        for _ in range(self.iterations):
            E = np.array([self._e(i) for i in range(len(X))])  # 此次迭代缓存的误差
            for i1 in range(len(X)):
                E1 = self._e(i1)
                if self._kkt(i1) or E1 == 0:  # 满足KKT条件或者误差为0
                    continue
                # 大于0则选择最小,小于0选择最大的
                i2 = np.argmin(E) if E1 > 0 else np.argmax(E)
                if i1 == i2:
                    continue
                E2 = self._e(i2)

                x1, x2 = X[i1], X[i2]
                y1, y2 = Y[i1], Y[i2]
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

                # 计算新参数
                alpha2_new = np.clip(alpha2 + y2 * (E1 - E2) / eta, L, H)
                alpha1_new = alpha1 + y1 * y2 * (alpha2 - alpha2_new)
                alpha2_delta = alpha2_new - alpha2
                alpha1_delta = alpha1_new - alpha1
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
                E[i1] = self._e(i1)
                E[i2] = self._e(i2)

    def predict(self, X: np.ndarray):
        Y = np.array([self._g(x) for x in X])
        return np.where(Y > 0, 1, -1)

    @property
    def support_vectors(self):  # 支持向量
        return self.X[self.alpha > 0]

    def _kkt(self, i):  # 是否满足KKT条件
        gi, yi = self._g(self.X[i]), self.Y[i]
        if np.abs(self.alpha[i]) < self.tol:
            return gi * yi >= 1
        if np.abs(self.alpha[i]) > self.C - self.tol:
            return gi * yi <= 1
        return np.abs(gi * yi - 1) < self.tol

    def _g(self, x):
        return np.sum(self.alpha * self.Y * self.K(self.X, x)) + self.b

    def _e(self, i):
        return self._g(self.X[i]) - self.Y[i]

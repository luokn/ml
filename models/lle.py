# -*- coding: utf-8 -*-
# @Date  : 2020/11/3
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class LLE:
    """
    Locally Linear Embedding(局部线性嵌入)
    """

    def __init__(self, k: int, d: int):
        """
        Args:
            k (int): 选取的k近邻数
            d (int): 降维输出维度
        """
        self.k, self.d = k, d

    def transform(self, X: np.array):
        """
        Args:
            X (np.array): shape为 M × c
        Returns
            Y (np.array): shape为 M × d
        """
        W = np.zeros([len(X), len(X)])  # M × M
        for i, x in enumerate(X):
            topk = np.linalg.norm(x - X, axis=1).argsort()[1:self.k + 1]  # 找出距离最近的k个点的下标，排除自身
            S = x - X[topk]  # M × k
            S = S.T @ S  # S = (x_i - N_i)^T (x_i - N_i) => k × k
            w = np.linalg.inv(S).sum(axis=1)  # 计算w
            W[i, topk] = w / w.sum()  # 归一化并扩展到所有点上
        M = np.eye(len(X)) - W
        M @= M.T  # M = (I - W)(I - W) ^ T
        L, U = np.linalg.eig(M)  # 对M进行特征值分解
        topd = L.argsort()[1:self.d + 1]  # 选取前d个非0特征值
        np.tile()
        return U[topd].T  # 降维后的数据为选取的特征值对应的特征向量

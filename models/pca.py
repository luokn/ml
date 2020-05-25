# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class PCA:
    """
    Principal Components Analysis(主成分分析)
    """

    def __init__(self, k: int):
        """
        :param k: 保留的主成因个数
        """
        self.k = k

    def transform(self, X: np.ndarray):
        Y = X - X.mean(axis=0)  # 去中心化
        L, U = np.linalg.eig(Y.T @ Y)  # 对协方差矩阵进行特征值分解
        topk = np.argsort(L)[::-1][:self.k]  # 找出特征值中前K大特征对应的索引
        return Y @ U[:, topk]  # 将去中心化矩阵乘以前K大特征对应的特征向量

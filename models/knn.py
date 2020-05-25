# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


class KNN:
    """
    K nearest neighbor classifier
    """

    def __init__(self, k: int):
        """
        :param k: 使用的近邻数
        """
        self.k, self.X, self.Y = k, None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X, self.Y = X, Y

    def predict(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)  # 类别输出
        for i, x in enumerate(X):
            dist = np.linalg.norm(self.X - x, axis=1)  # 计算x与所有已知类别点的距离
            topk = np.argsort(dist)[:self.k]  # 得到距离最近的k各节点对应的索引
            counter = Counter(self.Y[topk])  # 统计k各节点的类别数量
            Y[i] = counter.most_common(1)[0][0]  # 数量最多的类别将作为x的类别
        return Y

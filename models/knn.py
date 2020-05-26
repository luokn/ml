# -*- coding: utf-8 -*-
# @Date  : 2020/5/20
# @Author: Luokun
# @Email : olooook@outlook.com

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


class KNN:
    """
    K nearest neighbor classifier(K近邻分类器)
    """

    def __init__(self, k: int):
        """
        :param k: 分类近邻数
        """
        self.k, self.x_train, self.y_train = k, None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.x_train, self.y_train = X, Y  # 训练集X与Y，类别已知

    def predict(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)  # X对应的类别输出变量
        for i, x in enumerate(X):
            dist = np.linalg.norm(self.x_train - x, axis=1)  # 计算x与所有已知类别点的距离
            topk = np.argsort(dist)[:self.k]  # 取距离最近的k个点对应的索引
            counter = Counter(self.y_train[topk])  # 统计k近邻点的类别数量
            Y[i] = counter.most_common(1)[0][0]  # k近邻次数最多的类别将作为x的类别
        return Y

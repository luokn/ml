# -*- coding: utf-8 -*-
# @Date  : 2020/5/23
# @Author: Luokun
# @Email : olooook@outlook.com

import numpy as np


class DecisionTree:
    """
    Decision tree classifier(决策树分类器，ID3生成算法)
    """

    def __init__(self, rate: float = 0.95):
        self.rate = rate
        self.tree = None
        self.x_train, self.y_train = None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        indices = np.arange(X.shape[0])  # 所有行索引
        features = np.arange(X.shape[1])  # 所有列索引
        self.x_train, self.y_train = X, Y
        self.tree = self._create_tree(indices, features)

    def predict(self, X: np.ndarray):
        Y = np.zeros([len(X)], dtype=int)  # 输出变量
        for i, x in enumerate(X):
            Y[i] = self._predict(self.tree, x)
        return Y

    def _predict(self, node, x):
        if isinstance(node, dict):  # 如果节点是树(字典)类型
            val = x[node['feature']]  # 获取划分特征的值
            return self._predict(node['trees'][val], x)  # 根据值进行下一次递归
        return node  # 如果节点是叶子类型则直接返回该值

    def _create_tree(self, indices, features):
        best_class, rate = self._select_class(indices)  # 获得数量最多的类别及其频率
        if len(features) == 0 or rate > self.rate:  # 无特征可分或者满足一定的单一性
            return best_class  # 返回最单一的类别
        best_feature = self._select_feature(indices, features)  # 选择香农熵最小的特征
        trees = {}
        rest_features = features[features != best_feature]  # 除去选择的特征
        for val in np.unique(self.x_train[indices, best_feature]):  # 为该特征的每一个取值都建立子树
            sub_indices = self._query_indices(indices, best_feature, val)
            trees[int(val)] = self._create_tree(sub_indices, rest_features)  # 递归构建子决策树
        return {'feature': best_feature, 'trees': trees}

    def _query_indices(self, indices, feature, value):
        return indices[self.x_train[indices, feature] == value]

    def _calc_info_gain(self, indices, feature):  # 计算信息增益
        ent = self._calc_entropy(indices)  # 经验熵
        cond_ent = self._calc_cond_entropy(indices, feature)  # 经验条件熵
        return ent - cond_ent  # 信息增益

    def _calc_entropy(self, indices):  # 计算经验熵
        prob = np.bincount(self.y_train[indices]) / len(indices)  # 采用二进制计数法，x必须为正整数向量
        prob = prob[prob != 0]  # 除去0概率
        return np.sum(prob * -np.log(prob))  # 经验熵

    def _calc_cond_entropy(self, indices, feature):  # 计算条件熵
        cond_ent = 0  # 经验条件熵
        for val in np.unique(self.x_train[indices, feature]):
            sub_indices = self._query_indices(indices, feature, val)
            cond_ent += len(sub_indices) / len(indices) * self._calc_entropy(sub_indices)
        return cond_ent  # 条件熵

    def _select_class(self, indices):
        prob = np.bincount(self.y_train[indices]) / len(indices)  # 计算类别频率
        c = np.argmax(prob)
        return c, prob[c]  # 返回出现次数最多的类别，以及其频率

    def _select_feature(self, indices, features):
        gains = np.array([
            self._calc_info_gain(indices, feature) for feature in features
        ])  # 计算features中所有特征的信息增益
        f = np.argmax(gains)
        return features[f]  # 返回信息增益最大的特征

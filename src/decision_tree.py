# -*- coding: utf-8 -*-
# @Date  : 2020/5/23
# @Author: Luokun
# @Email : olooook@outlook.com


import numpy as np


class DecisionTree:
    """
    Decision tree classifier(决策树分类器，ID3生成算法)
    """

    def __init__(self):
        self.rate, self.root = None, None

    def fit(self, X: np.ndarray, Y: np.ndarray, rate: float = 0.95):
        self.rate = rate
        self.root = self._create_tree(X, Y, np.arange(X.shape[0]), np.arange(X.shape[1]))

    def __call__(self, X: np.ndarray):
        return np.array([self._predict(self.root, x) for x in X])

    def _predict(self, node, x: np.ndarray):
        if isinstance(node, dict):  # 如果节点是树(字典)类型
            feature, trees = node["feature"], node["trees"]
            return self._predict(trees[x[feature]], x)  # 根据值进行下一次递归
        return node  # 如果节点是叶子类型则直接返回该值

    def _create_tree(self, X: np.ndarray, Y: np.ndarray, indices: np.ndarray, features: np.ndarray):
        cat_counter = np.bincount(Y[indices])  # 计算类别个数
        if len(features) == 0 or np.max(cat_counter) / len(indices) > self.rate:  # 无特征可分或者满足一定的单一性
            return np.argmax(cat_counter)  # 返回最单一的类别
        k = np.argmax([self._calc_info_gain(X, Y, indices, f) for f in features])  # 最大信息增益特征
        feature = features[k]
        features = np.delete(features, k)  # 除去选择的特征
        trees = {
            value: self._create_tree(X, Y, indices[X[indices, feature] == value], features)
            for value in np.unique(X[indices, feature]).tolist()  # 为该特征的每一个取值都建立子树
        }
        return {"feature": feature, "trees": trees}

    @staticmethod
    def _calc_exp_ent(Y: np.ndarray, indices: np.ndarray):  # 计算经验熵
        prob = np.bincount(Y[indices]) / len(indices)
        prob = prob[prob.nonzero()]  # 除去0概率
        return np.sum(-prob * np.log(prob))  # 经验熵

    @classmethod
    def _calc_cnd_ent(cls, X: np.ndarray, Y: np.ndarray, indices: np.ndarray, feature: int):  # 计算条件熵
        ent = 0  # 经验条件熵
        for value in np.unique(X[indices, feature]):
            indices_ = indices[X[indices, feature] == value]
            ent += len(indices_) / len(indices) * cls._calc_exp_ent(Y, indices_)
        return ent  # 条件熵

    @classmethod
    def _calc_info_gain(cls, X: np.ndarray, Y: np.ndarray, indices: np.ndarray, feature: int):  # 计算信息增益
        exp_ent = cls._calc_exp_ent(Y, indices)  # 经验熵
        cnd_ent = cls._calc_cnd_ent(X, Y, indices, feature)  # 经验条件熵
        return exp_ent - cnd_ent  # 信息增益


def load_data():
    x = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
        ]
    )
    y = np.where(x.sum(axis=1) >= 2, 1, 0)
    return x, y


if __name__ == "__main__":
    x, y = load_data()
    decision_tree = DecisionTree()
    decision_tree.fit(x, y, rate=0.95)
    pred = decision_tree(x)

    print(decision_tree.root)
    print(y)
    print(pred)

    acc = np.sum(pred == y) / len(pred)
    print(f"Accuracy = {100 * acc:.2f}%")

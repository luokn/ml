# -*- coding: utf-8 -*-
# @Date  : 2020/5/23
# @Author: Luokun
# @Email : olooook@outlook.com


import numpy as np


class DecisionTree:
    """
    Decision tree classifier(决策树分类器, ID3生成算法)
    """

    def __init__(self):
        self.rate, self.root = None, None

    def fit(self, X: np.ndarray, y: np.ndarray, rate: float = 0.95):
        self.rate = rate
        self.root = self.build_tree(X, y, np.arange(X.shape[0]), np.arange(X.shape[1]))

    def __call__(self, X: np.ndarray):
        return np.array([self.predict(self.root, x) for x in X])

    def predict(self, node, x: np.ndarray):
        if isinstance(node, dict):  # 如果节点是树(字典)类型
            col, trees = node["col"], node["trees"]
            # 根据值进行下一次递归
            return self.predict(trees[x[col]], x)
        return node  # 如果节点是叶子类型则直接返回该值

    def build_tree(self, X: np.ndarray, y: np.ndarray, rows: np.ndarray, cols: np.ndarray):
        cats = np.bincount(y[rows])

        # 无特征可分或者满足一定的单一性
        if len(cols) == 0 or np.max(cats) / len(rows) > self.rate:
            return np.argmax(cats)  # 返回最单一的类别

        # 最大信息增益特征
        k = np.argmax([self.calc_info_gain(X, y, rows, f) for f in cols])
        col = cols[k]
        cols = np.delete(cols, k)  # 除去选择的特征

        # 为选择的特征创建子树
        trees = {
            value: self.build_tree(X, y, rows[X[rows, col] == value], cols)
            for value in np.unique(X[rows, col]).tolist()  # 为该特征的每一个取值都建立子树
        }
        return {"col": col, "trees": trees}

    @staticmethod
    def calc_exp_ent(y: np.ndarray, rows: np.ndarray):  # 计算经验熵
        prob = np.bincount(y[rows]) / len(rows)
        prob = prob[prob.nonzero()]  # 除去0概率
        return np.sum(-prob * np.log(prob))  # 经验熵

    @classmethod
    def calc_cnd_ent(cls, X: np.ndarray, y: np.ndarray, rows: np.ndarray, col: int):  # 计算条件熵
        ent = 0  # 经验条件熵
        for value in np.unique(X[rows, col]):
            indices_ = rows[X[rows, col] == value]
            ent += len(indices_) / len(rows) * cls.calc_exp_ent(y, indices_)
        return ent  # 条件熵

    @classmethod
    def calc_info_gain(cls, X: np.ndarray, y: np.ndarray, rows: np.ndarray, col: int):  # 计算信息增益
        exp_ent = cls.calc_exp_ent(y, rows)  # 经验熵
        cnd_ent = cls.calc_cnd_ent(X, y, rows, col)  # 经验条件熵
        return exp_ent - cnd_ent  # 信息增益


def load_data():
    X = np.array(
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
    y = np.where(X.sum(axis=1) >= 2, 1, 0)
    return X, y


if __name__ == "__main__":
    X, y = load_data()
    decision_tree = DecisionTree()
    decision_tree.fit(X, y, rate=0.95)
    y_pred = decision_tree(X)

    print(decision_tree.root)
    print(y)
    print(y_pred)

    acc = np.sum(y_pred == y) / len(y_pred)
    print(f"Accuracy = {100 * acc:.2f}%")

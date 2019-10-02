"""
决策树--分类树
"""
from collections import Counter
from math import log
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons,load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from linear_model.My_tools import plot_decision_boundary, tree_print
from linear_model.My_tools import tree_print
class DecsTree_Node:
    def __init__(self, d=None, v=None, l=None, r=None, label=None,e = None,trueRate=None):
        self._d = d
        self._v = v
        self._l = l
        self._r = r
        self.entropy = e
        self.label = label
        self.trueRate = trueRate
    def set_dv(self, d, v):
        self._d = d
        self._v = v
        return self
    def set_label(self, label):
        self.label = label

class DecisionTreeCls:
    def __init__(self,max_depth=10,min_leaf=3):
        self.root = None
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_leaf_node = None
    ## 信息熵的函数
    def fit(self, X, y):
        def entropy(y):
            count = Counter(y)
            res = 0.0
            for num in count.values():
                res += (num / len(y)) * log(num / len(y))
            return -res
        def gini(y):
            count = Counter(y)
            res = 0.0
            for num in count.values():
                res += (num/len(y))**2
            return 1-res
        ## 划分函数，输入一个 d（维度），和值，切分成左右两个
        def split(X, y, d, v):
            X_l, X_r = X[X[:, d]<=v],X[X[:, d]>v]
            y_l, y_r = y[X[:, d]<=v],y[X[:, d]>v]
            return X_l, X_r, y_l, y_r
        ## 用递归去划分
        def try_split(X, y):
            ## 从第一个维度开始找
            best_gini = np.inf
            best_d, best_v = -1, -1
            for d in range(X.shape[1]):
                gini_rcd = []
                vs = np.sort(X[:,d])
                for i in range(len(vs)-1):
                    if vs[i] != vs[i+1]:
                        v = (vs[i]+vs[i+1])/2
                        X_l, X_r, y_l, y_r = split(X, y, d, v)
                        l_fraction = len(y_l)/len(y)
                        e = l_fraction*gini(y_l) + (1-l_fraction)*gini(y_r)
                        gini_rcd.append(e)
                        if e < best_gini:
                            best_v = v
                            best_d = d
                            best_gini = e
                # print('split dim:',d)
                # plt.plot(vs[1:-1], gini_rcd)
                # plt.show()
            return best_d, best_v, best_gini
        def build_tree(X, y,depth):
            depth +=1 ## 控制决策数的深度
            ## 不需要继续划分左子树
            ## 这里最大深度和最小叶子节点
            if len(y) <= self.min_leaf or depth >= self.max_depth:  ## 这个值可以选择更小，但容易过拟合,当作叶子节点
                c = Counter(y)
                label = c.most_common(1)[0][0]
                ##print('bulid:',label)
                node = DecsTree_Node(d=-1,v=-1,e=0)
                node.set_label(label=label)
                node.trueRate = c.most_common(1)[0][1]/len(y)
                self.n_leaf_node +=1
                return node
            ## 已经足够纯度了
            elif gini(y)<0.1 :
                c = Counter(y)
                label = c.most_common(1)[0][0]
                ##print('bulid:',label)
                node = DecsTree_Node(d=-1, v=-1, e=0)
                node.set_label(label=label)
                node.trueRate = c.most_common(1)[0][1] / len(y)
                self.n_leaf_node += 1
                return node
            d, v, e = try_split(X, y)
            X_l, X_r, y_l, y_r = split(X, y, d, v)
            node = DecsTree_Node(d=d, v=v, e=e)
            node._l = build_tree(X_l, y_l, depth)
            node._r = build_tree(X_r, y_r, depth)
            ##print('left:',node._l,'right:',node._r)
            return node
        self.n_leaf_node = 0
        self.root = build_tree(X,y,0)
        return self
    def _predict(self, tree, x):
        if tree._l == None and tree._r == None:
            return tree.label
        if x[tree._d] <= tree._v:
            return self._predict(tree._l, x)
        else:
            return self._predict(tree._r, x)
    def predict(self, X):
        y_pre = np.array([self._predict(self.root,x) for x in X])
        return y_pre
    def _predict_proba(self,tree,x):
        if tree._l == None and tree._r == None:
            return tree.trueRate
        if x[tree._d] <= tree._v:
            return self._predict(tree._l, x)
        else:
            return self._predict(tree._r, x)
    def predict_proba(self,X):
        proba = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            proba[i] = self._predict_proba(self.root,X[i])
        return proba
    def score(self, X, y):
        y_pre = np.zeros(len(y))
        for i in range(X.shape[0]):
            y_pre[i] = self._predict(self.root,X[i])
        return (y_pre == y).astype(int).sum() / len(y)


if __name__ == '__main__':
    seed=66
    # # 使用鸢尾花的数据集测试
    # data = load_iris()
    # X = data.data[:,2:]
    # y = data.target
    # 使用的是月牙数据两个特征
    data = make_moons(n_samples=1000,noise=0.2,random_state=seed)
    X = data[0]
    y = data[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
    cls = DecisionTreeCls(min_leaf=2,max_depth=20)
    cls.fit(X_train, y_train)
    print('mycls: ',cls.score(X_test, y_test))
    print('n_leaves: ', cls.n_leaf_node)
    tree_print(cls.root,0)
    # tree_print(cls.root,0)
    skcls = DecisionTreeClassifier()
    skcls.fit(X_train,y_train)
    print('scikit: ',skcls.score(X_test,y_test))
    # print('sklearn 参数：\nskcls.feature_importances_:{}\nget_depth():{}\n'.format(skcls.max_depth,skcls.feature_importances_,skcls.get_depth()))
    print('n_leaves: ',skcls.get_n_leaves())
    # print('sklearn: ',skcls.score(X_test,y_test))
    plot_decision_boundary(lambda x: cls._predict(cls.root,x), X_train, y_train)
    plot_decision_boundary(lambda x: skcls.predict(x),X, y, X_ndim=2)



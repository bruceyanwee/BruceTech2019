"""
随机森林的实现
采用了行采样+列采样，预测采用投票的原则。
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from python_tricks.decrator import log_runing_time
from linear_model.My_tools import plot_decision_boundary,compare_models
from linear_model.DecisionTreeClassifier import DecisionTreeCls
from linear_model.AdaBoost import AdaBoost
from linear_model.LogisticRegression import LogisticRegressionClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from linear_model.GBDTClassifier import GBoostingTreeClassifier
class Base_Tree_Node:
    def __init__(self,l=None,r=None,d=None,v=None):
        self.l = l
        self.r = r
        self.d = d
        self.v = v
        self.label = None
class Base_model:
    def __init__(self,max_depth=10,min_leaf = 3):
        self.root = None
        self.max_depth = max_depth
        self.min_leaf = min_leaf
    def fit(self,X,y,learning_rate = 0.01):
        ## gini系数
        if X.ndim ==1:
            X = X.reshape(-1,1)
        def gini(y):
            count = Counter(y)
            res =0.0
            for num in count.values():
                res += (num/len(y))**2
            return 1-res
        def split(X,y,d,v):
            X_l,y_l = X[X[:,d]<v],y[X[:,d]<v]
            X_r,y_r = X[X[:,d]>=v],y[X[:,d]>=v]
            return X_l,X_r,y_l,y_r
        def try_split(X,y):
            best_crition = float('inf')
            best_d, best_v = -1 ,-1
            rd_feature = np.random.choice(X.shape[1], int(np.sqrt(X.shape[1])), replace=False)
            for d in rd_feature:
                v_space = np.sort(X[:,d])
                # v_space = np.linspace(X[:,d].min(),X[:,d].max(),int(1/learning_rate))
                for i in range(0,len(v_space)-1):
                    if v_space[i]!=v_space[i+1]:
                        v = (v_space[i]+v_space[i+1])/2
                        X_l, X_r, y_l, y_r = split(X,y,d,v)
                        fraction_l = len(y_l)/len(y)
                        g = fraction_l*gini(y_l)+(1-fraction_l)*gini(y_r)
                        if g < best_crition:
                            best_v = v
                            best_d = d
                            best_crition = g
            return best_d,best_v

        def build(X,y,depth):
            ## 特征子集的随机性体现在每一个节点的建立过程中
            depth += 1
            d,v = try_split(X,y)
            node = Base_Tree_Node(d=d,v=v)
            ## 到达叶子节点
            if gini(y) < 0.01 or len(y) <=self.min_leaf or depth >=self.max_depth:
                count = Counter(y)
                node.label = count.most_common(1)[0][0]
            else :## 需要继续建立左子树
                X_l, X_r, y_l, y_r = split(X, y, d, v)
                node.l = build(X_l,y_l,depth)
                node.r = build(X_r,y_r,depth)
            return node
        ## 建立子决策树
        self.root = build(X,y,0)
        return self
    def _predict(self,node,x):
        if node.l == None and node.r == None:
            return node.label
        if x[node.d] < node.v:
            return self._predict(node.l,x)
        else:
            return self._predict(node.r,x)
    def predict(self,X):
        y_pre = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pre[i] = self._predict(self.root,X[i])
        return y_pre

    def score(self,X,y):
        y_pred = self.predict(X)
        return (y==y_pred).astype('int').sum()/len(y)

class RandomForest:
    def __init__(self,n_estimater=20,max_depth=10,min_leaf = 3):
        self.n_estimater = n_estimater
        self.estimaters = []
        self.max_depth = max_depth
        self.min_leaf = min_leaf
    @log_runing_time
    def fit(self,X,y):
        for i in range(self.n_estimater):
            rd_x_sample = np.random.choice(X.shape[0],int(X.shape[0]/3))##可以选重复的，放回抽取
            cls = Base_model(max_depth=10,min_leaf = 3)
            X_i,y_i = X[rd_x_sample],y[rd_x_sample]
            ##X_i = X_i[:,rd_dim]
            cls.fit(X_i,y_i,learning_rate=0.01)
            self.estimaters.append(cls)
        return self

    def _predict(self,x):
        ## 分类树用投票决定
        y_k = [estimater._predict(estimater.root,x) for estimater in self.estimaters]
        c = Counter(y_k)
        return c.most_common(1)[0][0]

    def predict(self,X):
        y_pre = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pre[i] = self._predict(X[i])
        return y_pre
    def _predict_proba(self,x):
        ## 分类树用投票决定
        y_k = [estimater._predict(estimater.root,x) for estimater in self.estimaters]
        c = Counter(y_k)
        return c[1]/self.n_estimater
    def predict_proba(self,X):
        proba = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            proba[i] = self._predict_proba(X[i])
        return proba
    def score(self,X,y):
        y_pred = self.predict(X)
        return (y == y_pred).astype('int').sum() / len(y)



if __name__ == '__main__':
    seed = 66
    data = make_moons(n_samples=1000,noise=0.2,random_state=seed)
    X,y = data[0],data[1]
    # data = load_iris()
    # X,y = data.data[:,2:],data.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=seed)
    classfiers = [
        LogisticRegression(random_state=seed),
        LogisticRegressionClassifier(c=0.001,eta=0.2),
        RandomForest(n_estimater=20,max_depth=10,min_leaf=3),
        DecisionTreeCls(max_depth=10,min_leaf=3),
        GBoostingTreeClassifier(n_estimaters=30,max_depth=3,min_leaf=6,step=0.6),
        SVC(gamma=2, C=1, random_state=seed),
        GradientBoostingClassifier(
            n_estimators=20, learning_rate=1.0, max_depth=3, random_state=seed),
        RandomForestClassifier(
            max_depth=5, n_estimators=10, random_state=seed),
        DecisionTreeClassifier(random_state=seed),

        # AdaBoost()
    ]
    classifiers_names = ['LR','myLR','myRandomForest','myDecisionTreeCls','myGBDT',
                         'sklSVC','GBoostingClassifier','SklearnRFClassifier',
                         'SklearnDTree',
                         ]
    compare_models(classfiers,classifiers_names,X_train,y_train,X_test,y_test)





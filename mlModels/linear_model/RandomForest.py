"""
随机森林的实现
这个版本的随机森林用的是行抽样，也就是随机样本，效果没有达到很好

"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from python_tricks.decrator import log_runing_time
from linear_model.My_tools import plot_decision_boundary
class Base_Tree_Node:
    def __init__(self,l=None,r=None,d=None,v=None):
        self.l = l
        self.r = r
        self.d = d
        self.v = v
        self.label = None
class Base_model:
    def __init__(self,min_leaf=3,max_depth = 10):
        self.root = None
        self.min_leaf = min_leaf
        self.max_depth = max_depth
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
            best_d, best_v = -1,0
            for d in range(X.shape[1]):
                # v_space = np.(X[:,d].min(),X[:,d].max(),int(1/learning_rate))
                v_space = np.sort(X[:,d])
                for i in range(len(v_space)-1):
                    if v_space[i] != v_space[i+1]:
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
            depth += 1
            d,v = try_split(X,y)
            node = Base_Tree_Node(d=d,v=v)
            ## 到达叶子节点
            if gini(y) < 0.01 or len(y) < self.min_leaf or depth > self.max_depth:
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
            rd_x_sample = np.random.choice(X.shape[0],int(X.shape[0]/3))
            ## 注意这里是决策树的特征子集虽然是随机的，但是每颗都是一样的，特征子集的随机是指的是每个节点都是随机的
            rd_dim = np.random.choice(X.shape[1],int(X.shape[1]/2),replace=False)
            cls = Base_model(min_leaf=self.min_leaf,max_depth=self.max_depth)
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

    def score(self,X,y):
        y_pred = self.predict(X)
        return (y == y_pred).astype('int').sum() / len(y)

if __name__ == '__main__':
    # data = make_moons(n_samples=1000,noise=0.2,random_state=66)
    # X,y = data[0],data[1]
    data = load_iris()
    X,y = data.data[:,2:],data.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=66)
    cls = RandomForest(n_estimater=20,max_depth=10,min_leaf=3)
    cls.fit(X_train,y_train)
    print('randomForest: ',cls.score(X_test,y_test))
    cls2 = Base_model(max_depth=10,min_leaf=3)
    cls2.fit(X_train,y_train)
    print('basemodel:',cls2.score(X_test,y_test))
    # plot_decision_boundary(lambda x:cls2._predict(cls2.root,x),X,y)
    plot_decision_boundary(lambda x: cls._predict(x), X, y)





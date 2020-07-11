"""
GBDT分类模型的总结：
    1、y的onehot编码有利于进行计算损失loss和梯度函数，一个矩阵乘法即可解决
    2、每一轮m，训练的是k棵树，每棵树对样本属于类别k（k=0,1,2...k-1）的概率进行回归
    3、进行样本概率预测，或者说模型输出的时候，每个样本又两层循环
        1、第一层是模型m的求和（m次）
        2、第二层是对样本的每个类别都要预测（k次）
    4、这里的梯度不完全像感知机、逻辑回归、神经网络、
        1、对模型的参数w、theta求梯度，然后更新模型（策略：更新模型参数）
        2、而是对模型整个求梯度，相当于链式求导，只求了第一层，然后更新模型（策略：加入新模型）
        但是目的都一样，最小化loss
    5、回到多分类问题，始终离不开 ovo 和 ovr，这里GBDT用到的类似softmax，其实也就是一种ovr的方法
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from linear_model.My_tools import plot_decision_boundary
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
class TreeNode:
    def __init__(self,l=None,r=None,d=None,v=None):
        self.l = l
        self.r = r
        self.d = d
        self.v = v
        self.c = None
class BaseRGTree:
    def __init__(self,max_depth = 2,min_leaf = 4):
        self.root = None
        self.m = None       ## 表示第几轮的提升，方便加入shrinkge
        self.max_depth = max_depth
        self.min_leaf = min_leaf
    def fit(self,X,y):
        learning_rate = 0.01 ## 节点划分的步长 0.01--划分成100份
        if X.ndim ==1:
            X = X.reshape(-1,1)
        def min_squar(y1, y2):
            c1, c2 = np.mean(y1), np.mean(y2)
            return np.sum((y1 - c1) ** 2) + np.sum((y2 - c2) ** 2)
        def split(X,y,d,v):
            X_l,X_r = X[X[:,d]<v],X[X[:,d]>=v]
            y_l,y_r = y[X[:,d]<v],y[X[:,d]>=v]
            return X_l,X_r,y_l,y_r
        def try_split(X,y):
            if X.ndim == 1:     ## 保证是二维数组
                X = X.reshape(-1,1)
            best_d,best_v = -1,-1
            best_crition = np.inf
            for d in range(X.shape[1]):
                # v_space = np.linspace(X[:,d].min(),X[:,d].max(),int(1/learning_rate))
                v_space = np.sort(X[:,d])
                for i in range(len(v_space)-1):
                    if v_space[i]!=v_space[i+1]:
                        v = (v_space[i]+v_space[i+1])/2
                        X_l, X_r, y_l, y_r = split(X,y,d,v)
                        crition = min_squar(y_l,y_r)
                        if crition < best_crition:
                            best_d, best_v = d,v
                            best_crition = crition
            return best_d,best_v
        def build_tree(X,y,depth):
            ## 开始建立决策树
            depth +=1
            node = TreeNode()
            if depth >=self.max_depth or len(y) < self.min_leaf: ## 递归终止,到达叶子节点
                node.c = np.mean(y)
                return node
            ## 继续建立左右子树
            d,v = try_split(X,y)
            node.d = d
            node.v = v
            X_l, X_r, y_l, y_r = split(X,y,d,v)
            node.l = build_tree(X_l,y_l,depth)
            node.r = build_tree(X_r,y_r,depth)
            return node
        self.root = build_tree(X,y,0)
        return self
    def _predict(self,node,x):
        ## 到达叶子节点
        if x.ndim ==1:
            x = x.reshape(-1,1)
        if node.l == None and node.r == None:
            return node.c
        ## 继续判断左右子树
        if x[node.d]<node.v :
            return self._predict(node.l,x)
        else:
            return self._predict(node.r,x)
    def predict(self,X):
        if X.ndim ==1:
            X = X.reshape(-1,1)
        y_pre = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pre[i] = self._predict(self.root,X[i])
        return y_pre
    def score(self,X,y):
        y_pre = self.predict(X)
        return np.sum((y_pre==y).astype(int))/len(y)

class BaseLR:
    def __init__(self):
        pass

class GBoostingTreeClassifier:
    def __init__(self,n_estimaters =20,max_depth=3,min_leaf=6,step = 0.5):
        self.models = [] ## 模型的数量是 M x K，每一轮训练K颗树
        self.M = n_estimaters
        self.C = None    ## y分布的频次，概率，1 x K
        self.step = step
        self.class_k = None
        self.max_depth=max_depth
        self.min_leaf = min_leaf
    def init_paramaters(self,X,y):
        self.class_k = len(Counter(y).keys())
        self.C = np.zeros(self.class_k)
    def fit(self,X,y):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        ## 求预测偏差，作为下一轮拟合的目标
        self.init_paramaters(X,y)
        def loss_fm(X,y):## y是独热码的形式
            ## 1、计算预测值 pi --难点在于要综合前面所有模型的值、
            ## 2、计算损失值 sum（yi * log pi）
            loss = 0.0
            for i in range(X.shape[0]): ## 每一个样本
                y_proba = np.copy(self.C) ## 样本i的k个类别对应的logp
                for im in range(len(self.models)):
                    y_pre = np.array([model._predict(model.root,X[i]) for model in self.models[im]])
                    ##y_pre = y_pre - np.max(y_pre) ## 防止overflow，指数
                    ##y_proba += np.exp(y_pre)/np.sum(np.exp(y_pre))
                    y_proba += y_pre
                y_proba = y_proba/np.sum(y_proba)
                y_proba[y_proba>1] =1
                y_proba[y_proba<=0] = 1e-3
                y_logproba = np.log(y_proba)
                loss += np.dot(y[i],y_logproba) ## y[i](1 x k) .dot y_logproba(k x 1)  = 1 x 1
            return -loss/len(y)

        def grad_fm(X,y):
            ## 和求损失函数很相似，损失函数是返回 的一个sum()，这里要返回一个向量，表示前面模型的梯度
            proba_pre = np.zeros((X.shape[0],self.class_k))
            proba_mean = self.C
            for i in range(X.shape[0]):  ## 每一个样本
                y_proba = np.copy(proba_mean)  ## 样本i的k个类别对应的logp
                for im in range(len(self.models)):
                    y_pre = np.array([model._predict(model.root,X[i]) for model in self.models[im]])
                    ##y_pre = y_pre - np.max(y_pre)  ## 防止overflow，指数
                    ##y_probaim = np.exp(y_pre) / np.sum(np.exp(y_pre))
                    y_proba += y_pre
                proba_pre[i] = y_proba ## grad (m x 1)
            return proba_pre - y ## m x k 其实这个不算用w的梯度去类比，就想成是需要拟合的概率差

        ## f0为均值
        c = Counter(y)
        self.C = np.array([v for k,v in c.most_common()])/len(y)
        ## 初始化第一个模型
        ## 把y转换成独热码
        oneHot = OneHotEncoder()
        y = oneHot.fit_transform(y.reshape(-1,1)).toarray()
        # print(loss_fm(X,y))
        ## 添加第一轮个树模型，方便计算
        # trees_1 = []
        # grad1 = grad_fm(X,y)
        # for i in range(self.class_k): ## 每个类别的分类器，弄清楚要拟合的是什么，某个类的真实概率和预测概率的差值,fit(X,（deltaP）)
        #     tree1i = BaseRGTree()
        #     tree1i.fit(X,-self.step*grad1[:,i])
        #     trees_1.append(tree1i)
        # self.models.append(trees_1)
        loss_rcd = []
        for m in range(self.M):
            ## 接下来每一轮都会训练k颗数
            trees_m = []
            gradm = grad_fm(X,y)
            for i in range(self.class_k):
                treemi = BaseRGTree(max_depth = self.max_depth,min_leaf = self.min_leaf)
                treemi.fit(X,-self.step*gradm[:,i])
                trees_m.append(treemi)
            self.models.append(trees_m)
            ##print('loss m:',loss_fm(X,y))
            loss_rcd.append(loss_fm(X,y))
        return self,loss_rcd
    def _predict(self,x):
        y_proba = np.copy(self.C)
        for im in range(len(self.models)):
            y_pre = np.array([model._predict(model.root, x) for model in self.models[im]])
            # y_pre = y_pre - np.max(y_pre)  ## 防止overflow，指数
            # y_probaim = np.exp(y_pre) / np.sum(np.exp(y_pre))
            y_proba += y_pre
        return np.argmax(y_proba)
    def _predict_proba(self,x):
        y_proba = np.copy(self.C)
        for im in range(len(self.models)):
            y_pre = np.array([model._predict(model.root,x) for model in self.models[im]])
            # y_pre = y_pre - np.max(y_pre)  ## 防止overflow，指数
            # y_probaim = np.exp(y_pre) / np.sum(np.exp(y_pre))
            y_proba += y_pre
        y_proba[y_proba > 1] = 1
        y_proba[y_proba <= 0] = 0
        # print('样本i： ', y_proba)
        return y_proba
    def predict(self,X):
        y_pre = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pre[i] = self._predict(X[i])
        return y_pre
    def predict_proba(self,X):
        y_preproba = np.zeros((X.shape[0],self.class_k))
        for i in range(X.shape[0]):
            y_preproba[i] = self._predict_proba(X[i])
            # print('proba{}:{}'.format(i,y_preproba[i]))
        return y_preproba
    def score(self,X,y):
        y_pre = self.predict(X)
        return np.sum((y==y_pre).astype(int))/len(y)

if __name__ == '__main__':
    # data = load_iris()
    # X = data.data
    # y = data.target
    seed = 666
    moons = make_moons(n_samples=1000,noise=0.2,random_state=seed)
    X, y = moons[0],moons[1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=seed)
    cls = GBoostingTreeClassifier(n_estimaters=30,max_depth=3,min_leaf=6,step=0.6)
    _,loss_rcd = cls.fit(X_train,y_train)
    print('GBDT score:',cls.score(X_test, y_test))
    # print(cls.predict(X_test))
    # cls.predict_proba(X)
    # print(cls.predict_proba(X))
    # plt.scatter(X[:,0],X[:,1],c=y)
    # plt.show()
    plt.plot(np.linspace(0,len(loss_rcd),len(loss_rcd)),loss_rcd)
    plt.show()
    base = BaseRGTree(max_depth=5,min_leaf=6)
    base.fit(X_train,y_train)
    print('base score: ',base.score(X_test,y_test))
    # print(base.predict(X_test))
    # plot_decision_boundary(lambda x:cls._predict(x),X,y)
    # plot_decision_boundary(lambda x: base._predict(base.root, x), X, y)


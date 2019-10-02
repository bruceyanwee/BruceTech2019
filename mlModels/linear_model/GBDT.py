"""
GBDT梯度提升树
结合boosting和向前分布迭代的思想，拟合残差
1、同样的，作为集成模型，需要基础模型
2、这里的基础模型是决策回归树，注意是回归树，连续值才能出现残差
3、显然GBDT实现回归树简单，如果要实现分类功能
总结存在的问题是：
    1、很容易过拟合，如果步长设置为1的化，
    2、如果设置为 0.5 又不会收敛
    3、GBDT，不用担心偏差，更关心的是方差
    4、可以说boosting的模型的偏差不会大，主要关注是方差，怎么能够减小方差。
    5、对比随机森林是一种并行的bagging，方差不是问题，偏差是重点。
    6、目前选择的回归问题是用的square做损失函数，对于离群点敏感，也就是对于噪声敏感，可以改进成HuberLoss
    7、关于M的选择，可以让模型自己选择，依据是当 改变的grad / grad < threshold 就停止训练
    8、GBDT的解决分类问题是这样：
        （1）这里的思路是借鉴的 线性回归 --> 逻辑回归
        （2）而不是用的决策树分类的判别方式进行的概率计算
        （3）所以一直强调说GBDT的决策树是回归树
    9、所以要把GBDT分类弄清楚，还是要把逻辑回归领悟透
"""
import numpy as np
import matplotlib.pyplot as plt
class TreeNode:
    def __init__(self,l=None,r=None,d=None,v=None):
        self.l = l
        self.r = r
        self.d = d
        self.v = v
        self.c = None
class BaseRGTree:
    def __init__(self,max_length = 6,min_leaf = 3):
        self.root = None
        self.m = None       ## 表示第几轮的提升，方便加入shrinkge
        self.max_length = max_length
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
                v_space = np.linspace(X[:,d].min(),X[:,d].max(),int(1/learning_rate))
                for v in v_space:
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
            if depth >=self.max_length or len(y) < self.min_leaf: ## 递归终止,到达叶子节点
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
class GBoostingTree:
    def __init__(self,M =20,step = 1.0):
        self.models = []
        self.M = M
        self.C = None
        self.step = step
    def fit(self,X,y):
        self.C = np.mean(y)
        tree1 = BaseRGTree(max_length=3,min_leaf=10)
        tree1.fit(X,y-self.C)
        self.models.append(tree1)
        for m in range(self.M):
            ## 计算y（i-1）的输出值
            y_pred = self.C
            for i in range(len(self.models)):
                y_pred += self.step*self.models[i].predict(X)
            ## 这里的损失函数用的是均方误差，求梯度刚好就是 y - y_pred
            grad = y_pred-y
            print('the grad of tree({}) is {} '.format(m,np.sum(grad**2)))
            tree_m = BaseRGTree(max_length=3,min_leaf=10)
            tree_m.fit(X,-grad)
            self.models.append(tree_m)
        return self

    def predict(self,X):
        y_pre = self.C
        for i in range(len(self.models)):
            y_pre += self.step*self.models[i].predict(X)
        return y_pre


if __name__ == '__main__':
    x = np.linspace(0,10,100)
    y = x**2 + np.random.randn(len(x))*5
    y_real = x**2
    model = GBoostingTree(M=40,step=0.3)
    base = BaseRGTree(max_length=3)
    model.fit(x,y)
    base.fit(x,y)
    y_pre = model.predict(x)
    y_base = base.predict(x)
    plt.plot(x,y_real,label = 'y=x**2' )
    plt.plot(x,y,label = 'y=x**2+noise',alpha = 0.3)
    plt.plot(x,y_pre,'*',label = 'GBDT')
    plt.plot(x,y_base,'.',label = 'base_model')
    plt.legend(loc='best')
    plt.show()

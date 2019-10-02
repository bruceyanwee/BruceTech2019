"""
决策树--回归树
"""
from math import log
import numpy as np
from sklearn.model_selection import train_test_split
from linear_model.My_tools import plot_decision_boundary, tree_print
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeRegressor
class TreeNode:
    """
    回归树的节点
    dim = 代表分割的维度
    val = 代表分割的值，
    y_pred,该节点的预测值，用节点下的均值代替
    l,r ，代表左右子数，每个维度进行左右划分
    """
    def __init__(self,dim=None,val=None):
        self.dim = dim
        self.val = val
        self.l = None
        self.r = None
        self.y_pred = None
class RegressionTree:
    def __init__(self):
        self.root = None
    ## 拟合回归树
    def fit(self,X,y,min_leaf=5,max_depth = 3):
        """
        :param X: 样本特征向量，ndim可以是多维的，这里写了两组不同的函数处理
        :param y: 样本的y值
        :param min_leaf: 每个叶子节点的最小样本数，防止过拟合的，默认值是5
        :param max_depth: 数的最大深度，同样是为了防止过拟合
        :return: self，拟合完成后，返回self.root,回归树建立
        """
        def MSE(y):
            """
            评价集合纯度，类似信息熵,这里选取的是均方差，注意与回归树不一样的是，最佳分割点的MSE最大。分类是熵最小
            :param y:
            :return:
            """
            if len(y) > 0:
                return np.sqrt(np.sum((y-y.mean())**2))/len(y)
            return 0
        def split(X,y,d,v):
            """
            给定一组样本，和指定维度，指定值进行划分，返回两个样本集

            """
            X_l,y_l = X[X[:,d]<=v],y[X[:,d]<=v]
            X_r,y_r = X[X[:,d]>v],y[X[:,d]>v]
            return X_l,y_l,X_r,y_r

        def try_split(X,y):
            """
            通过迭代的方式去寻找该树（样本）下的最佳划分，返回最佳划分的（d，v）
            """
            best_mse = -1
            best_d,best_v = -1,-1
            for d in range(X.shape[1]):
                try:
                    x_lin = np.linspace(X[:,d].min(),X[:,d].max(),len(X[:,d]))
                    for v in x_lin[1:-1]:
                        X_l, y_l, X_r, y_r = split(X,y,d,v)
                        mse = (MSE(y_l)+MSE(y_r))/2
                        if(mse > best_mse):
                            best_mse = mse
                            best_d = d
                            best_v = v
                except:
                    print('X 为空 : ',X)
            return best_d,best_v
        def build(X,y,depth):
            """
            递归构建回归树的函数
            :param X:
            :param y:
            :param depth: 控制树的深度
            :return: 一颗树的节点，注意是递归调用
            """
            d, v = try_split(X,y)
            depth +=1
            X_l, y_l, X_r, y_r = split(X,y,d,v)
            node = TreeNode(dim=d,val=v)
            if len(y) <= min_leaf or MSE(y) < 0.1 or depth>=max_depth: ## 到了叶子结点
                node.y_pred = y.mean()
                return node
            elif len(y_l) or len(y_r): ## 还需要继续划分，建立节点,防止出现空的节点
                if len(y_l):
                    node.l = build(X_l,y_l,depth)
                else:
                    node.l = None
                if len(y_r):
                    node.r = build(X_r,y_r,depth)
                else:
                    node.r = None
            return node
        ## 针对只有一个维度的数据 统一加 "_"
        def _split(x,y,v):
            x_l,y_l = x[x<=v],y[x<=v]
            x_r,y_r = x[x>v],y[x>v]
            return x_l,y_l,x_r,y_r
        def _try_split(x,y):
            best_mse = -1
            best_v = float('inf')
            ##print(x)
            if len(x) > 0:
                x_lin = np.linspace(x.min(), x.max(), len(x))
                for v in x_lin[1:-1]:
                    x_l, y_l, x_r, y_r = _split(x, y, v)
                    mse = (MSE(y_l) + MSE(y_r)) / 2
                    if (mse > best_mse):
                        best_mse = mse
                        best_v = v
            return best_v
        def _build(x,y,depth):
            depth += 1
            v = _try_split(x, y)
            x_l, y_l, x_r, y_r = _split(x,y,v)
            node = TreeNode(val=v)
            if len(y) <= min_leaf or MSE(y) < 0.1 or depth >= max_depth:  ## 到了叶子结点
                node.y_pred = y.mean()
                return node
            else:  ## 还需要继续划分，建立节点
                node.l = _build(x_l, y_l,depth)
                node.r = _build(x_r, y_r,depth)
            return node
        ## fit过程:
        if X.ndim == 1:
            self.root = _build(X,y,0)
        else:
            self.root = build(X,y,0)
        return self
    ##预测单个样本
    def _predict(self,x):
        def digui(x,node):
            if node.l == None and node.r == None: ## 到了叶子节点
                return node.y_pred
            if x[node.dim] >= node.val and node.r:
                return digui(x,node.r)
            else:
                return digui(x,node.l)
        ## 单个样本的
        def _digui(x,node):
            if node.l == None and node.r == None: ## 到了叶子节点
                return node.y_pred
            if x >= node.val and node.r:
                return _digui(x,node.r)
            else:
                return _digui(x,node.l)

        if type(x) == np.float64:
            return _digui(x,self.root)
        else:
            return digui(x,self.root)
    ## 预测一批样本
    def predict(self,X):
        if X.ndim == 1:
            size = len(X)
            y = np.zeros(size)
        else:
            size = X.shape[0]
            y = np.zeros(size)
        for i in range(size):
            y[i] = self._predict(X[i])
        return y
    def score1(self,X,y):
        y_pred = self.predict(X)
        return np.sqrt(((y-y_pred)**2).sum())/len(y)
    def score2(self,X,y):
        y_pred = self.predict(X)
        return 1 - np.sqrt(((y-y_pred)**2).sum())/np.sqrt(((y-y.mean())**2).sum())


if __name__ == '__main__':
    x = np.linspace(0,10,100)
    y = (x-5)**2 + (x-5)**2 + np.random.randn(100)
    xx1,xx2 = np.meshgrid(x, x)
    X = np.c_[xx1,xx2]
    y_plot = (xx1-5)**2 + (xx2-5)**2 + np.random.randn(100,100)
    tree = RegressionTree()
    tree.fit(X,y,min_leaf=4,max_depth=10)
    print('score(MSE): ',tree.score2(X,y))
    y_pred = tree.predict(X)
    sktree = DecisionTreeRegressor()
    sktree.fit(X,y)
    print('sktree',sktree.score(X,y),'max_depth: ',sktree.max_depth)
    ##y_pred = y_pred.reshape(100,100)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx1, xx2, y_plot,cmap='rainbow',alpha=0.2)
    ax.plot(x,x,y,color = 'r',label='real')
    ax.scatter(x,x,y_pred,color='g',label = 'predict')
    plt.legend(loc = 'best')
    plt.show()

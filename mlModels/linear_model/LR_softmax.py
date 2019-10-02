"""
逻辑回归的soft Max是GBDT的基础，也是神经网络的基础
    1、每个样本输入，模型对应了K个类别的概率值的输出
    2、每次模型迭代对应的是负梯度，注意在softmax中有k组w更新，在GBDT中k颗新树的生成
"""
from collections import Counter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from linear_model.My_tools import plot_decision_boundary
from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder
from warnings import simplefilter
from sklearn.pipeline import Pipeline
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class LR_SoftMax:
    def __init__(self,c=0.1,learning_rate=0.5):
        self.class_k = None
        self.W = None
        self.b = None
        self.c = c ## 正则化系数
        self.learning_rate=learning_rate
    def init_parameters(self,X,y):
        c = Counter(y)
        self.class_k = len(c.keys())
        self.W = np.random.randn(X.shape[1],self.class_k)
        self.b = np.random.rand(self.class_k)
    def fit(self,X,y,learning_rate=0.5):
        def LossW(X, y,W,b):
            Z = np.dot(X, W) + b
            Z = Z - np.max(Z, axis=1)[:,None]
            y_exp = np.exp(Z)
            y_prolog = np.log(y_exp / np.sum(y_exp, axis=1)[:, None])
            loss = 0.0
            for i in range(X.shape[0]):
                loss += np.dot(y[i],y_prolog[i])
                # for j in range(self.class_k):  ## 类别这里的编码，默认为 0 - k-1
                #     if y[i] == j:  ## y[i] 只会是一个值，0-k-1，找到对应的那个概率
                #         loss += y_prolog[i][j]
                #         ##print('样本{}的偏差: {:.2f} '.format(i,-y_prolog[i][j]))
            regularization = (self.c/2)*np.sum(W**2)
            return -loss / X.shape[0] +regularization/X.shape[0]

        def grad_W(X, y, W, b):
            Z = np.dot(X, W) + b
            # Z = Z - np.max(Z, axis=1)[:,None]
            y_exp = np.exp(Z)
            y_proba = y_exp / np.sum(y_exp, axis=1)[:, None]
            y_prolog = np.log(y_proba)
            loss = np.zeros((X.shape[0], self.class_k))
            for i in range(X.shape[0]):
                loss[i] = y_proba[i] - y[i]
            grad_w = (np.dot(X.T,loss)) / X.shape[0] + self.c/X.shape[0] * W
            grad_b = np.dot(np.ones((1,X.shape[0])),loss)/X.shape[0] +self.c/X.shape[0] * b
            return grad_w,grad_b
        ## n x k
        def SGD(X, y, W, b,n_iters=100):
            n = 1
            loss_rcd = []
            while (n < n_iters):
                n += 1
                gradw,gradb = grad_W(X, y, W, b)
                W = W - self.learning_rate * gradw
                b = b - self.learning_rate * gradb
                loss_rcd.append(LossW(X, y, W,b))
            return W,b,loss_rcd
        ## 保证数据为2维，用到了ndarray的.shape属性
        if X.ndim ==1:
            X = X.reshape(-1,1)
        self.init_parameters(X,y)
        ## 把y转换成onehot编码，易于1{yi = 1}logp的实现
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
        init_W = np.random.randn(X.shape[1],self.class_k)
        init_b = np.random.randn(self.class_k)
        W,b,loss_rcd = SGD(X,y,init_W,init_b,n_iters=1000)
        self.W = W
        self.b = b
        return loss_rcd
    def _predict(self,x):
        if x.ndim ==1:
            x = x.reshape(1,-1)
        Z = np.dot(x, self.W) + self.b
        Z = Z - np.max(Z, axis=1)[:, None]
        y_exp = np.exp(Z)
        y_proba = y_exp / np.sum(y_exp, axis=1)[:, None]
        return np.argmax(y_proba)
    def predict(self,X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = self._predict(X[i])
        return y_pred
    def score(self,X,y):
        y_pre = self.predict(X)
        return np.sum((y==y_pre).astype(int))/len(y)
if __name__ == '__main__':
    """
    X = np.random.randn(10, 4)
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    cls = LR_SoftMax()
    cls.fit(X,y)
    y_pre = cls.predict(X)
    print('y_true: ',y)
    print('predict:',y_pre.astype(int))
    """
    ## 鸢尾花测试
    ## 归一化,
    data = load_iris()
    X = data.data
    y = data.target
    """
    data = make_moons(n_samples=200,noise=0.2)
    X = data[0]
    y = data[1]
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=66)
    pipe = Pipeline([
        ('std',StandardScaler()),
        ('model',LR_SoftMax(c=0.01,learning_rate=0.1)),
    ])
    pipe.fit(X_train,y_train)
    print('score: ',pipe.score(X_test, y_test))
    # plt.plot(np.linspace(0, 10, len(lrc)), lrc)
    # plt.show()
    # plot_decision_boundary(lambda x:cls._predict(x),X,y)



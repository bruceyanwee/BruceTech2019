
import numpy as np
import matplotlib.pyplot as  plt
from math import log
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from linear_model.My_tools import plot_decision_boundary
from python_tricks.decrator import log_runing_time
class LogisticRegressionClassifier:
    def __init__(self,c=0.1,eta=0.5):
        self._theta = None
        self._inception = None
        self._coef = None
        self.c =c
        self.eta = eta
    ## 概率转换函数
    def sigmiod(self,z):
        return 1.0/(1+np.exp(-z))
    ## 把bias转换成Xb
    def toXb(self,X):
        return np.c_[np.ones(X.shape[0]),X]
    ## 拟合函数
    ## 包括了 损失
    ## 函数计算函数 J_theta()，梯度计算函数Grad_theta()，优化函数SGD()
    @log_runing_time
    def fit(self,X,y):
        ## 损失函数，这里没有除以m，样本数，只是一个常数
        def J_theta(X,y,theta):
            p = self.sigmiod(X.dot(theta))
            regularization = self.c/2 * np.sum(theta[1:]**2)
            epson = 1e-5
            return -np.sum((y * np.log(p+epson) + (1-y)*np.log(1-p+epson))) + regularization/X.shape[0]
        ## 梯度的计算是关键，这里直接搬用公式了，推导过程省略
        def Grad_theta(X,y,theta):
            return X.T.dot(self.sigmiod(X.dot(theta))-y)/len(y) + np.insert(self.c/len(y) * theta[1:],0,theta[0])  ## theta0 不进行正则化
        ## 梯度下降
        def SGD(X,y,init_theta,n_iters = 1000):
            n = 1
            theta = init_theta
            learn_rcd = []
            while(n < n_iters):
                learn_rcd.append(J_theta(X,y,theta))
                det_theta = Grad_theta(X,y,theta)
                theta = theta - self.eta * det_theta
                n +=1
                ## 用来记录一下训练过程，用作后面的可视化
            return theta,learn_rcd
        ## 开始fit过程
        X = self.toXb(X)
        init_theta = np.random.randn(X.shape[1])
        self._theta,learn_rcd = SGD(X,y,init_theta)
        # 可视化模型学习曲线
        # plt.plot(np.linspace(0, len(learn_rcd), len(learn_rcd)), learn_rcd)
        # plt.show()
        self._inception = self._theta[0]
        self._coef = self._theta[1:]
        return self,learn_rcd

    ## 预测样本的概率
    def _predict_proba(self,x):
        x = np.insert(x, 0, 1)
        return self.sigmiod(x.dot(self._theta))
    def predict_proba(self,X):
        X = self.toXb(X)
        return self.sigmiod(X.dot(self._theta))
    ## 预测单个样本
    def _predict(self,x):
        x = np.insert(x, 0, 1)
        return (self.sigmiod(x.dot(self._theta))>=0.5).astype(int)
    def predict(self,X):
        X = self.toXb(X)
        return (self.sigmiod(X.dot(self._theta))>=0.5).astype(int)
    def score(self,X,y):
        y_pred = self.predict(X)
        return np.sum((y==y_pred).astype(int))/len(y)
    def __repr__(self):
        return "LogisticRegressionClassifier()"

if __name__ == '__main__':
    seed = 666
    data = make_moons(200,noise=0.1,random_state=seed)
    X,y = data[0],data[1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=seed)
    cls = LogisticRegressionClassifier(c=0.001,eta=0.2)
    _,learn_rcd = cls.fit(X_train,y_train)
    print('score: ',cls.score(X_test,y_test))
    # plt.scatter(X[:,0],X[:,1],c=y)
    # plt.show()
    plt.plot(np.linspace(0,len(learn_rcd),len(learn_rcd)),learn_rcd)
    plt.show()
    plot_decision_boundary(lambda x:cls._predict(x),X,y)




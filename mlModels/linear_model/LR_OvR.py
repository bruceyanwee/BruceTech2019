"""
逻辑回归多分类的另一种实现方式
OvR，K个分类，训练K个分类器
"""
import numpy as np
from collections import Counter
from linear_model.LogisticRegression import LogisticRegressionClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_model.My_tools import plot_decision_boundary
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons,make_blobs
class LR_OvR:
    def __init__(self,c=0.01, learning_rate=0.1):
        self.models = []
        self.class_k = None
        self.c = c
        self.learning_rate = learning_rate
    def init_parameters(self,X,y):
        c = Counter(y)
        self.class_k = len(c.keys())
        for i in range(self.class_k):
            model = LogisticRegressionClassifier(c=self.c,eta=self.learning_rate)
            self.models.append(model)
        return self

    def fit(self,X,y):
        self.init_parameters(X,y)
        ## 训练k个二分类器，分别预测每一个类别的概率，最后输出最大的概率对应的类别
        for i in range(self.class_k): ## 每一个类别 训练y==i的类别
            y_train = np.copy(y)
            index = (y_train==i)
            y_train[index] = 1
            y_train[~index] = 0
            self.models[i].fit(X,y_train)
        return self
    def _predict(self,x):
        y_proba = np.zeros(self.class_k)
        for i in range(self.class_k):
            y_proba[i] = self.models[i]._predict_proba(x)
        return np.argmax(y_proba)
    def predict(self,X):
        y_pre = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pre[i] = self._predict(X[i])
        return y_pre
    def score(self,X,y):
        y_pre = self.predict(X)
        return np.sum((y==y_pre).astype(int))/len(y)

class LR_OvO:
    def __init__(self,c=0.01, learning_rate=0.1):
        self.models = []
        self.class_k = None
        self.models_num = None
        self.c = c
        self.learning_rate = learning_rate
    def init_parameters(self,X,y):
        c = Counter(y)
        self.class_k = len(c.keys())
        self.models_num = int(self.class_k * (self.class_k - 1) / 2)
        for i in range(self.class_k-1): ## 每一个类别 训练y==i的类别
            self.models.append([])
            for j in range(i+1,self.class_k): ## i 代表一个类别，j 代表一个类别
                model = LogisticRegressionClassifier(c=self.c,eta=self.learning_rate)
                self.models[i].append(((i,j),model)) ## 每个模型复杂预测（i，j）类别， 预测 0--i，1--j
        return self
    def fit(self,X,y):
        self.init_parameters(X,y)
        ## 训练k个二分类器，分别预测每一个类别的概率，最后输出最大的概率对应的类别
        for i in range(self.class_k-1): ## 每一个类别 训练y==i的类别
            for j in range(self.class_k-1-i):
                indexi = (y==self.models[i][j][0][0])
                indexj = (y==self.models[i][j][0][1])
                y_train = np.copy(y[indexi | indexj])
                X_train = np.copy(X[indexi | indexj])
                traini = (y_train == self.models[i][j][0][0])
                trainj = (y_train == self.models[i][j][0][1])
                y_train[traini] = 0
                y_train[trainj] = 1
                self.models[i][j][1].fit(X_train,y_train)
        return self
    def _predict(self,x):
        y_count = np.zeros(self.class_k) ## 投票决定类别,i j 对应的是类别
        for i in range(self.class_k-1):
            for j in range(self.class_k-1-i):
                yij_pre = self.models[i][j][1]._predict(x)
                if yij_pre==0:
                    label = self.models[i][j][0][0]
                    y_count[label] +=1
                else:
                    label = self.models[i][j][0][1]
                    y_count[label] +=1
        # print(y_count)
        return np.argmax(y_count)
    def predict(self,X):
        y_pre = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pre[i] = self._predict(X[i])
        return y_pre
    def score(self,X,y):
        y_pre = self.predict(X)
        return np.sum((y==y_pre).astype(int))/len(y)

if __name__ == '__main__':
    ## 鸢尾花测试
    ## 归一化,
    data = load_iris()
    X = data.data
    y = data.target
    # data = make_blobs(n_samples=1000,n_features=2,centers=5,cluster_std=1,random_state=666,)
    # X = data[0]
    # y = data[1]
    # plt.scatter(X[:,0],X[:,1],c=y)
    # plt.show()
    normalizer = MinMaxScaler()
    X = normalizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=666)
    cls = LR_OvO(c=0.01, learning_rate=0.1)
    lrc = cls.fit(X_train, y_train)
    print('score: ', cls.score(X_test, y_test))
    # plt.plot(np.linspace(0, 10, len(lrc)), lrc)
    # plt.show()
    # plot_decision_boundary(lambda x:cls._predict(x),X,y)
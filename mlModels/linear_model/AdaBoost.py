import numpy as np
from math import log
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from linear_model.My_tools import plot_decision_boundary
class AdaBoost:
    def __init__(self,n_estimater=20):
        """
        :param n_estimater: 基模型的数量
        :param learning_rate: 基模型的学习率，默认基模型是树桩模型（只有一个（best_d,best_v,left_negative））
        """
        self.base_models = []
        self.models_num = n_estimater
        self.alpha = []
        self.weights = []
    def init_para(self,X,y):
        self.M,self.N = X.shape[0],X.shape[1]
        init_weights = np.ones(X.shape[0])/X.shape[0]
        self.weights.append(init_weights)
        return self

    ## 训练单个基模型
    def _Gm(self,X,y,weights_m,learning_rate = 0.1):
        """
        :param X:
        :param y:
        :param weights_m: 第m次迭代，也就是第m个基模型的样本权重
        :param learning_rate: 基模型的学习率，默认基模型是树桩模型（只有一个（best_d,best_v,direct = 0））
        :return:
        """
        ## 找到最佳的（best_d,best_v）
        def split(X,y,d,v):
            index_l,index_r = X[:,d] < v,X[:,d] >= v
            X_l,X_r,y_l,y_r = X[index_l], X[index_r], y[index_l], y[index_r]
            return index_l,index_r,y_l,y_r
        def error_with_weights(index_l,index_r,y_l,y_r,weights):
            ## 误分类率的样本*该样本的权值
            ## 这里的定义很关键
            ## direct = 0,表示左边为负类
            error_direct0 = 0.0
            error_direct1 = 0.0
            weights_l,weights_r = weights[index_l],weights[index_r]
            l = (y_l==-1).astype(int)
            r = (y_r== 1).astype(int)
            error_direct0 = np.sum(weights_l*l) + np.sum(weights_r*r)
            l = (y_l == 1).astype(int)
            r = (y_r == -1).astype(int)
            error_direct1 = np.sum(weights_l * l) + np.sum(weights_r * r)
            if error_direct0<error_direct1:
                return 0,error_direct0
            else:
                return 1,error_direct1

        ## 一些工具函数，计算中间值的
        def _alpha_m(error_m):
            ## print(error_m)
            return (1/2) * log((1 - error_m) / error_m)
        ## 基模型的训练
        ## try_split(X,y,weights):
        best_error_withW = float('inf')
        best_d, best_v, best_direct = -1, -1, -1
        for d in range(self.N):
            ## 最佳划分点的待选值
            n_step = int(1/learning_rate)
            v_sapce = np.linspace(X[:,d].min(),X[:,d].max(),n_step)
            for v in v_sapce[1:-1]:
                index_l, index_r, y_l, y_r = split(X,y,d,v)
                ##direct,error = error_with_weights(index_l,index_r,y_l,y_r,weights)
                ##print(weights_m)
                weights_l, weights_r = weights_m[index_l], weights_m[index_r]
                l = (y_l != -1).astype(int)
                r = (y_r != 1).astype(int)
                error_direct0 = np.sum(weights_l * l) + np.sum(weights_r * r)
                l = (y_l != 1).astype(int)
                r = (y_r != -1).astype(int)
                error_direct1 = np.sum(weights_l * l) + np.sum(weights_r * r)
                if error_direct0 < error_direct1:
                    direct, error = 0,error_direct0
                else:
                    direct, error = 1, error_direct1
                ## 是否需要进行更新
                if error < best_error_withW:## 当前次：最佳的基模型
                    best_error_withW = error
                    best_d, best_v, best_direct = d,v,direct
        ## 添加新模型
        self.base_models.append((best_d, best_v, best_direct))
        ## 更新参数
        alpha_m = _alpha_m(best_error_withW)
        self.alpha.append(alpha_m)
        ## 计算 Zm
        y_pred = np.zeros(len(y))
        if best_direct == 0:
            y_pred[X[:,best_d]<best_v] = -1
            y_pred[X[:,best_d] >= best_v] = 1
        else:
            y_pred[X[:,best_d] < best_v] = 1
            y_pred[X[:,best_d] >= best_v] = -1
        Z_m = np.sum(weights_m * np.exp(-alpha_m*y_pred*y))
        ## 更新 weight
        new_weights = weights_m * np.exp(-alpha_m*y_pred*y)/Z_m
        self.weights.append(new_weights)
        return self
    ## 训练模型
    def fit(self,X,y,learning_rate = 0.1):
        self.init_para(X, y)
        for epoch in range(self.models_num):
            ## 1 训练基模型，返回GM在训练集的分类误差率
            ## 2 更新Gm的系数
            ## 3 更新数据集上的权值分布
            self._Gm(X, y, self.weights[epoch], learning_rate)
        ## 归一化
        self.alpha = self.alpha/np.sum(self.alpha)
        return self
    def predict(self,X):
        ## print(X.shape)
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            ## 每个样本综合子模型的结果进行预测
            res = 0.0
            for j in range(len(self.base_models)):
                d, v, direct = self.base_models[j]
                if direct == 0: ## direct == 0,左为-1
                    p = 1 if X[i,d]< v else -1
                    res += self.alpha[j]* p
                else:
                    p = 1 if X[i,d]>=v else -1
                    res += self.alpha[j]* p
            y_pred[i] = 1 if res<0 else -1
        return y_pred
    def _predict(self,x):
        res = 0.0
        for j in range(len(self.base_models)):
            d, v, direct = self.base_models[j]
            if direct == 0:  ## direct == 0,左为-1
                p = 1 if x[d] < v else -1
                res += self.alpha[j] * p
            else:
                p = 1 if x[d] >= v else -1
                res += self.alpha[j] * p
        y_pred = 1 if res < 0 else -1
        return y_pred
    def score(self,X,y):
        y_pred = self.predict(X)
        return np.sum((y==y_pred).astype(int))/len(y)

if __name__ == '__main__':
    data = make_moons(noise=0.1)
    X = data[0]
    y = data[1]
    y[y==0] = -1
    ada = AdaBoost(n_estimater=50)
    ada.fit(X,y)
    print(ada.score(X,y))
    print(ada.predict(X))
    print(y)
    plt.scatter(X[:,0],X[:,1],c=y)
    plot_decision_boundary(lambda x:ada._predict(x),X,y)
    ##plt.title('AdaBoost model,n_estimater={}'.format(ada.models_num))
    plt.show()






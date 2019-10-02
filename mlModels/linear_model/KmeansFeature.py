"""
模型stacking的一个例子，通过K-means的输出作为下一个模型的特征输入
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.datasets import make_moons
import scipy
from linear_model.LogisticRegression import LogisticRegressionClassifier
class KmeansFeaturizer:
    """
    将数字型数据输入k-均值聚类
    并导出输出为
    """
    def __init__(self,k=100):
        self._k = k
        self.k_center = None
        self.label = None
        self.encoder = OneHotEncoder().fit(np.arange(k).reshape(-1,1))
    def fit(self,X,y=None):
        def distance(x1,x2):
            return np.sqrt(np.sum(x1-x2)**2)
        ## 迭代终止条件
        epson = 0.1
        iters,n_iters =1,1000
        k_cts = X[np.random.choice(X.shape[0], self._k, replace=False)]
        y = np.ones(X.shape[0])
        while(iters < n_iters):
            iters +=1
            #1.随机挑选k个样本，作为初始均值向量
            for i in range(X.shape[0]):
                ## 对每个样本计算距离
                dis_k = [distance(X[i],center) for center in k_cts]
                ## 找到最近的center，标记类别
                y[i] = np.argmin(dis_k)
            ## 更新均值向量
            change = 0.0
            for j in range(self._k):
                new_center = np.mean(X[y==j],axis=0)
                change += (new_center-k_cts[j]).sum()
                k_cts[j] = new_center
            if change < 1:
                break
        self.label = y
        self.k_center = k_cts
        return self
    ## 给原始数据加上 k—means的特征项，使用独热码。
    def predict(self,X):
        y_pre = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            disk = [np.sqrt(np.sum(X[i]-center)**2) for center in self.k_center]
            y_pre[i] = np.argmin(disk)
        return y_pre.astype('int')

    def transform(self,X):
        cluster = self.predict(X)
        return self.encoder.transform(cluster.reshape(-1,1))

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)



if __name__ == '__main__':
    data = make_moons(n_samples=1000,noise=0.1)
    X,y = data[0],data[1]
    kmeans = KmeansFeaturizer(k=10)
    kmeans.fit(X)
    # 使用k-均值特技器生成簇特征
    training_cluster_features = kmeans.transform(X)
    print(training_cluster_features)
    # 将新的输入特征和聚类特征整合
    training_with_cluster = scipy.sparse.hstack((X,training_cluster_features))
    print(training_cluster_features[:10])
    lg_cls = LogisticRegressionClassifier()
    lg_cls.fit(X,y)
    print('origin score: ',lg_cls.score(X,y))
    ##lg_cls.fit(training_with_cluster, y)
    ##print('origin score: ', lg_cls.score(training_with_cluster, y))
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()
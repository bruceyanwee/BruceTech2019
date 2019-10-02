import matplotlib.pyplot as plt
import numpy as np
import sklearn
## 可视化决策边界
def plot_decision_boundary(pred_func,X,y,X_ndim =1):
    # 设定最大最小值，附加一点点边缘填充
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    ##X_mesh = np.c_[np.ones(X_mesh.shape[0]),X_mesh]
    ##X_mesh = polynomial(X_mesh,3)
    # 用预测函数预测一下
    Z = np.zeros(X_mesh.shape[0])
    if X_ndim == 1:
        for i in range(len(Z)):
            Z[i] = pred_func(X_mesh[i])
        ## Z = np.map(X_mesh,func)
        Z = Z.reshape(xx.shape)
    else:
        Z = pred_func(X_mesh)
        Z = Z.reshape(xx.shape)
    # 然后画出图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

## 把树的结果打印出来（递归的方式）：
def tree_print(tree,depth):
    if tree == None:
        return
    if tree.label==None and tree._d==0:
        print('depth:{}  dim:{} value:{:.2f} label:{}'.format(depth, tree._d, tree._v, tree.label))
    depth +=1
    tree_print(tree._l,depth)
    tree_print(tree._r,depth)
## 比较多个模型的性能函数，输入一个classifiers的列表

def test_roc(model,data,labels):
    if hasattr(model,'decision_function'):
        predictions = model.decision_function(data)
    else:
        predictions = model.predict_proba(data)
        if predictions.ndim>1:
            predictions = predictions[:,1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)
    return fpr, tpr

def compare_models(classifiers,classifier_names,X_train,y_train,X_test,y_test):
    for model in classifiers:
        model.fit(X_train,y_train)
    plt.figure()
    for i, model in enumerate(classifiers):
        fpr, tpr = test_roc(model, X_test,y_test)
        plt.plot(fpr, tpr, label=classifier_names[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.show()

def test_GBDT_Loss(y_real=0,delta = 0.5):
    huberloss = []
    ys = np.linspace(-2,2,100)
    for y in ys:
        if abs(y-y_real)<=delta:
            loss = (y-y_real)**2 * 0.5
        else:
            loss = delta*(abs(y-y_real)-0.5*delta)
        huberloss.append(loss)
    squrloss = (ys-y_real)**2 *delta
    absloss = np.abs(ys-y_real)*delta
    plt.plot(ys,huberloss,label='huberloss')
    plt.plot(ys,squrloss,label = 'squarloss')
    plt.plot(ys,absloss,label = 'absloss')
    plt.legend(loc='best')
    plt.show()
if __name__ == '__main__':
    test_GBDT_Loss()
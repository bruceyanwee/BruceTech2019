import matplotlib.pyplot as plt
import numpy as np

def jiecheng(k):
    res = 1
    for i in range(1,k+1):
        res = res * i
    return res

def posion(p,k):
    return np.exp(-p)*(p**k)/jiecheng(k)

if __name__ == '__main__':
    print(jiecheng(5))
    p_20 = np.array([[posion(p,k) for k in range(1,20)] for p in range(1,10)])
    print(p_20.shape)
    fig = plt.figure(figsize=(6,6))
    for p,ks in zip(range(1,10),p_20):
        print(p,ks)
        plt.plot(range(1,20),ks,'--',label='p:{}'.format(str(p)))
    ##plt.set_xlabel('k')
    ##plt.set_ylabel('P(X = k)')
    plt.legend(loc = 'best')
    plt.show()

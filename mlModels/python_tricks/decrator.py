"""
学习一些装饰器的用途
"""
import time
def log_runing_time(func):
    def wrapper(*args):
        t1 = time.time()
        res = func(*args)
        t2 = time.time()
        print('func:{} 运行时间：{:.2} s'.format(func.__name__,t2-t1))
        return res
    return wrapper

@log_runing_time
def sum(num):
    res = 0.0
    for i in range(num):
        res +=i
    return res

if __name__ == '__main__':
    a =1
    print(sum.__name__,sum.__annotations__)
    func = sum(1000)
    print(func)
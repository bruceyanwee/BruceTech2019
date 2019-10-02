import numpy as np
def split(n,arr):
    split_point = -1
    for i in range(1,n):
        left,right = arr[:i] ,arr[i:]
        if np.max(left)<=np.min(right):
            print(left,right)
            split_point = i
            break
    if split_point == -1:## 说明没找到这样的点，那么该子数组的可分数是1
        return 1
    left,right = arr[:split_point],arr[split_point:]
    return split(len(left),left)+ split(len(right),right)

if __name__ == '__main__':
    n = 10
    arr = [1,10,3,99,6,7,8,11,10]
    print(split(n,arr))

import numpy as np
def dp_clicks(s):
    N = len(s)
    dp = np.zeros((N+1,2)).astype(int)
    dp[0][0] = 1
    dp[0][1] = 0
    for i in range(1,N+1):
        c = s[i]
        if c >= 'A' and c <= 'Z':
            dp[i,0] = np.min(dp[i-1,0]+1,dp[i-1,1]+2)
            dp[i,1] = np.min(dp[i-1,0]+2,dp[i-1,1]+2)
        if c>='a' and c <='z':
            dp[i,0] = np.min(dp[i-1,0]+2,dp[i-1,1]+2)
            dp[i,1] = np.min(dp[i-1,0]+2,dp[i-1,1]+1)
    return np.min(dp[N][0],dp[N][1])

if __name__ == '__main__':
    s = 'AaAAAA'
    print(dp_clicks(s))
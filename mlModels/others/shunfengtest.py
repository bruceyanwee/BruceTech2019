def lengthOfLIS(n, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    size = len(nums)
    if size <= 1:
        return size
    dp = [1] * size
    for i in range(1, size):
        for j in range(i):
            if nums[i] >= nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
if __name__ == '__main__':
    n = 5
    nums = [1,2,1,3,4]
    print('n: ',n)
    print('nums:',nums)
    print(lengthOfLIS(5,nums))
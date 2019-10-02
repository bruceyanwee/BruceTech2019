
if __name__ == '__main__':
    s = 'HG[3|B[2|CA[2|MN[2|LKJ]]]]F[4|QWER]'
    while(']' in s):
        index_end = s.find(']')
        temp = s[:index_end]
        index_start = temp[::-1].find('[')
        sub_s = s[index_end-index_start-1:index_end+1]
        i = sub_s.find('|')
        dupl_s = int(sub_s[1:i])*sub_s[i+1:-1]
        s = s.replace(sub_s,dupl_s)
        print(s)
    print('complete: ',s)
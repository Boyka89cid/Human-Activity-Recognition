def longPalSub(s):
    
    n = len(s)

    dpal = [[False for j in range(n)] for i in range(n)]

    for i in range(n):
        dpal[i][i] = True

    for i in range(n-1):
        if s[i] == s[i+1]:
            dpal[i][i+1] = True

    k = 3

    while k < n:
        i = 0
        while i < n-k+1:
            j = i+k-1

            if s[i] == s[j] and dpal[i+1][j-1]==True:
                dpal[i][j] = True
                sind = i
                mxlen = k
            i += 1
        k += 1

    return s[sind:sind+mxlen]


if __name__ == '__main__':

    s = 'forgeeksskeegfor'

    print(longPalSub(s))















def palcheck(s,i,j):
    
    while i < j:
    
        if s[i] != s[j]:
            return False
        else:
            i += 1
            j -= 1

    return True




def isPalRot(s):
    
    i = 0
    j = len(s)-1

    if palcheck(s,i,j):
        return True
    else:

        for x in range(j):
            s1 = s[x+1:j+1]
            s2 = s[0:x+1]

            s1 += s2

            if palcheck(s1,i,j):
                return True


    return False
        

        



if __name__ == '__main__':

    s = 'aad'

    print(isPalRot(s))















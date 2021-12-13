
def findpair(arr,s):
    d = dict()

    l = len(arr)

    for i in range(l):
        sec = s-arr[i]

        if sec in d:
            print("Pair of Given Sum " + str((sec, arr[i])) +' Index Positions are '+str((d[sec], i)))
        
        d[arr[i]] = i
        



if __name__ == '__main__':

    arr = [1,-1,13,10, 11,2,24,-12,1000,-988,76]
    
    s=12

    findpair(arr,s)













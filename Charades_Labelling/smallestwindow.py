
def findwindow(s,p):

    p_ln = len(p)
    s_ln = len(s)

    minlen = s_ln

    pdict = {}
    sdict = {}

    for i in range(p_ln):
        pdict[p[i]] = pdict.get(p[i],0) + 1

    cnt = 0
    start = 0
    str_ind = -1

    for i in range(s_ln):
        sdict[s[i]] = sdict.get(s[i],0) + 1

        if s[i] in pdict and pdict[s[i]] >= sdict[s[i]]:
            cnt += 1

        if cnt == p_ln and s[i] in pdict:

            while s[start] not in pdict or pdict[s[start]] < sdict[s[start]]:
                ch = s[start]

                if ch in pdict and pdict[ch] < sdict[ch]:
                    sdict[ch] = sdict.get(ch,0) - 1
                
                start += 1

            l = i - start + 1

            if minlen > l:
                minlen = l
                str_ind = start

    return s[str_ind:str_ind+minlen]
        

        



if __name__ == '__main__':

    s = 'geeksforgeeks'
    p = 'ork'

    w = findwindow(s,p)
    print(w)


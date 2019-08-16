def sum_of_lists(N):
    t=0
    for i in range(5):
        L=[i**(i>>1)] for i in range(N)]
        t+=sum(L)
        del L 
        return t

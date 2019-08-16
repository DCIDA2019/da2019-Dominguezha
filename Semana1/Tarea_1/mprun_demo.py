def sum_of_lists(N):
    t = 0
    for i in range(5):
        L = [j ^ (j >> i) for j in range(N)]
        t += sum(L)
        del L 
        return t

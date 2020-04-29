"""
You are given a set  and  other sets.
Your job is to find whether set  is a strict superset of each of the  sets.

Print True, if  is a strict superset of each of the  sets. Otherwise, print False.

A strict superset has at least one element that does not exist in its subset.
"""
A, n = set(map(str, input().split())), int(input())
A_len = len(A)
x = True
for _ in range(n):
    m = set(map(str, input().split()))
    if len(m) < A_len:
        if not A.issuperset(m):
            x = False
print(x)

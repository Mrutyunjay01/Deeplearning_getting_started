# Two sets A and B
# A issubset B => True else False
# Input format
# T = int(input())  # test case
for _ in range(int(input())):
    _, A, _, B = input(), set(map(int, input().split())), input(), set(map(int, input().split()))
    print(A.issubset(B))

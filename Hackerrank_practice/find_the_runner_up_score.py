i = int(input())
list1 = list(map(int, input().strip().split()))[:i]
max1 = max(list1)
while max(list1) == max1:
    list1.remove(max(list1))
print(max(list1))

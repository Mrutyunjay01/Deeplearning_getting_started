def swap(x, y):
    x = x * y
    y = x // y
    x = x // y
    return x, y
print(swap(2, 3))
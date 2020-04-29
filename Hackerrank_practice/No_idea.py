"""There is an array of  integers. There are also  2 disjoint sets,  A and B, each containing  integers. You like all the
integers in set  and dislike all the integers in set . Your initial happiness is . For each  integer in the array,
if , you add  to your happiness. If , you add  to your happiness. Otherwise, your happiness does not change. Output
your final happiness at the end. """
n, m = input().split()
integers = input().split()
A = input().split()
B = input().split()
print(sum([(i in A) - (i in B) for i in integers]))


import sorting
import time
import random
arr = [random.uniform(1,100000000) for i in range(0, 30000)]
arr_2 = arr


t1 = time.time()
sorting.sort_1(arr)
t2 = time.time()
t3 = time.time()
sorting.sort_2(arr_2)
t4 = time.time()

print(f"{t2 - t1} - время сортировки нашей")
print(f"{t4 - t3} -  время сорировки двумя форами")
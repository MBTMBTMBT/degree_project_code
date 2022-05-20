import itertools
import random


x = range(-3000, 3000)
y = range(-2000, 2000)
n = 20000000
random_list = list(itertools.product(x, y))
location = random.sample(random_list, n)
print(location)
with open(r"D:\pythonproject\mapreduce\test.txt", "w") as f:
    for i in range(n):
        f.write(str(str(i).zfill(8)))
        #f.write(" ")
        f.write("(%05d,%05d)" % (location[i][0], location[i][1]))
        f.write("\n")
x1 = [i[0] for i in location]
y1 = [i[1] for i in location]
# print(x1)
# print(y1)


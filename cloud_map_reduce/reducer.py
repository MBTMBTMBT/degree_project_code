from operator import itemgetter
import sys

locations = []
num = []
index = -1
area = [[0], [1], [2], [3], [4], [5], [6], [7]] # whole 8 areas
# for i in range(len(list)):
#     for j in range(len(area)):
#         if list[i][1] == area[j][0]:
#             area[j].append(list[i][0])
#print(area)
a = 0
for i in sys.stdin:
    i = i.strip("\n")
    word = i.split()
    # print(word)
    # locations.append(word)
    for j in range(len(area)):
        if str(word[1]) == str(area[j][0]):
            area[j].append(word[0]) # put the vehicle in each area
    a = a + 1
for i in range(8):
    print(area[i])




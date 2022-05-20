import sys
locations = []
for i in sys.stdin:
        i = i.strip("\n")
        word = i.split()
        locations.append(word)
for i in locations:
        for j in i:
                #print(j, 1)
                number = j[0:8]
                x1 = j[9:14]
                #print("x1", x1)
                y1 = j[15:20]
                #print("y1", y1)
                if -3000 <= int(x1) <= -1000 and -500 <= int(y1) <= 2000:
                        print(number, "0")  # 0 stands for area 0
                elif -1500 <= int(x1) <= 1500 and 500 <= int(y1) <= 2000:
                        print(number, "1") # 1 stands for area 1
                elif -3000 <= int(x1) <= -1000 and -2000<= int(y1) <= 0:
                        print(number, "2") # 2 stands for area 2
                elif -1500 <= int(x1) <= 500 and -2000 <= int(y1) <= 700:
                        print(number, "3") # 3 stands for area 3
                elif 1000 <= int(x1) <= 3000 and 0 <= int(y1) <= 2000:
                        print(number, "4") # 4 stands for area 4
                elif 2000 <= int(x1) <= 3000 and -1000 <= int(y1) <= 2000:
                        print(number, "5") # 5 stands for area 5
                elif 1500 <= int(x1) <= 3000 and -2000 <= int(y1) <= 200:
                        print(number, "6") # 6 stands for area 6
                elif 0 <= int(x1) <= 1700 and -2000 <= int(y1) <= 1000:
                        print(number, "7") # 7 stands for area 7
                #print(number, 1)
                #print(number)






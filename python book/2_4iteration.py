import numpy as np
num = np.zeros(10)
i = 0
while(i<=9):
    num[i] = int(input('please input an integer '))
    i += 1
odd = []
for n in num:
    if(n%2 != 0):
        odd.append(n)
biggest = 0
if(len(odd) == 0):
    print('You have input 10 evens!')
else:
    for o in odd:
        if(o>biggest):
            biggest = o
    print(biggest)

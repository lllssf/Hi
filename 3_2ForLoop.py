s = '1.23,2.4,3.123'
a = ''
total = 0
for i in s:
    if(i==','):
        total += float(a)
        a = ''
    else:
        a += i
total += float(a)
print(total)

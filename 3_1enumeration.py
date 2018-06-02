x = int(input("please enter an integer: "))
pwr = 2
while(pwr < 6):
    root = 0
    while(root**pwr < abs(x)):
        root += 1
    if(root**pwr == abs(x)):
        print(root,'**',pwr,'=',x)
        break
    pwr += 1
if(root**pwr != abs(x)):
    print("The root and pwr are not exsit!")

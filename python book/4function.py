def findRoot(x,power,epsilon):
    """Assume x and epsilon int or float, power an int,
        epsilon > 0, power > 0
    Returns float y such that y**power is within epsilon of x.
        If such a float does not exit, it returns None"""
    # negative number has no even-powered roots
    if(x < 0 and power % 2 == 0):
        return None
    low = min(-1,x)
    high = max(1,x)
    ans = (low + high)/2
    while(abs(ans**power - x)>epsilon):
        if(ans**power<x):
            low = ans
        else:
            high = ans
        ans = (low+high)/2
    return ans

def testFindRoot():
    """Tesr the performance of the findRoot function"""
    epsilon = 0.0001
    for x in [0.25, -0.25, 8, -8]:
        for power in range(1,4):
            print('Testing x =',str(x),'and power = ',power)
            result = findRoot(x,power,epsilon)
            if(result == None):
                print(' No root')
            else:
                print(' ',result**power,'~=',x)

testFindRoot()

from tkinter import *
from tkinter import ttk

#---------------------------------------------------------------
# FibRecur
#---------------------------------------------------------------
def FibRecur( n, C ):
    global depth
    if n <= 2:
        print(8*depth*"-" + " " + C + " n =", n, " ret 1")
        return 1
    else:
        print(8*depth*"-" + " " + C + " n =", n)
        depth = depth + 1

        f = FibRecur(n - 1, 'L') + FibRecur(n - 2, 'R')

        depth = depth - 1
        print(8*depth*"-" + " " + C + " ret", f)

        return f

depth = 0
f = FibRecur(6, 'X')

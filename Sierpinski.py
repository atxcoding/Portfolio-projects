from tkinter import *
from math import *


#------------------------------------------------------------------------
# ST_Recur
#------------------------------------------------------------------------
def ST_Recur(canvas, level, p1, p2, p3):
    canvas.create_polygon(p1, p2, p3)

#------------------------------------------------------------------------
# DrawSierpinskiTriangle
#------------------------------------------------------------------------
def DrawSierpinskiTriangle(event=None):
    global size
    level = int(levels.get())
    canvas.delete("all")
    p1 = (0.1*size, 0.9*size)   # bottom left
    p2 = (0.5*size, 0.1*size)   # top
    p3 = (0.9*size, 0.9*size)   # bottom right
    ST_Recur(canvas, level, p1, p2, p3)



#===================================================================================
root = Tk()
root.title("Sierpinski Triangle")

#---- entry box for level number
Label(root, text="Levels:").grid(row=1, column=1, sticky=W)
levels = StringVar()
levels_entry = Entry(root, width=7, textvariable=levels)
levels_entry.grid(row=1, column=2)

#---- button to draw
Button(root, text="Draw", command=DrawSierpinskiTriangle).grid(row=1, column=3)
#---- return key to draw
root.bind("<Return>", DrawSierpinskiTriangle)

#---- canvas to draw
size = 500
canvas = Canvas(root, width=size, height=size, borderwidth=1, highlightbackground='black', background='white')
canvas.grid(row=1, column=4)

#---- space out widgets
for child in root.winfo_children():
    child.grid_configure(padx=5, pady=5)

#---- start event loop
root.mainloop()

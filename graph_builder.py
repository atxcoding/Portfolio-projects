from tkinter import *
from tkinter import ttk
from pyqtree import Index
# pip install pyqtree



#---- callback functions
def hello():
    print( "hello!")
def PrintToConsole():
    print(graph)

#---- the root window
root = Tk()
root.title("Graph Builder")

#---- a frame inside the root window
mainframe = ttk.Frame(root, padding="20 20 20 20")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

# create a menu bar and pulldown menus
menubar = Menu(root)

filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", command=hello)
filemenu.add_command(label="Save", command=hello)
filemenu.add_command(label="Print to Console", command=PrintToConsole)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)

menubar.add_cascade(label="File", menu=filemenu)

editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Cut", command=hello)
editmenu.add_command(label="Copy", command=hello)
editmenu.add_command(label="Paste", command=hello)
menubar.add_cascade(label="Edit", menu=editmenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=hello)
menubar.add_cascade(label="Help", menu=helpmenu)

root.config(menu=menubar)

#---- drawing canvas
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
class Node():
    def __init__(self, label, xtl, ytl, xbr, ybr ):
        self.bbox = (xtl, ytl, xbr, ybr)
        self.label = label
        self.parent = None
        self.children = list()

nodes = list()
spindex = Index(bbox=(0, 0, 500, 500))
BBOX = 10
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

lastx, lasty = 0, 0
start_label = ' '
finish_label = ' '
graph = dict()
global start_node

def XY(event):
    global lastx, lasty, start_label, start_node
    lastx, lasty = event.x, event.y
    matches = spindex.intersect((event.x-BBOX, event.y-BBOX, event.x+BBOX, event.y+BBOX))
    print(matches[0].label)
    start_label = matches[0].label
    start_node = matches[0]
    if start_label not in graph:
        graph[start_label] = []

def AddLine(event):
    global lastx, lasty, start_label, finish_label, start_node
    canvas.create_line((lastx, lasty, event.x, event.y))
    lastx, lasty = event.x, event.y
    matches = spindex.intersect((event.x-BBOX, event.y-BBOX, event.x+BBOX, event.y+BBOX))
    if matches and matches[0].label != start_label and start_label != ' ':
        print(matches[0].label)
        finish_label = matches[0].label
        start_node.children.append(matches[0])
        graph[start_label].append(matches[0].label)
        start_label = ' '


count = ord('A')
def AddNode(event):
    global nodes, spindex, count
    new_node = Node(chr(count), event.x-BBOX, event.y-BBOX, event.x+BBOX, event.y+BBOX)
    nodes.append(new_node)
    spindex.insert(new_node, new_node.bbox)
    canvas.create_oval(event.x-BBOX, event.y-BBOX, event.x+BBOX, event.y+BBOX)
    canvas.create_text(event.x, event.y, text=new_node.label)
    count = count + 1


canvas = Canvas(root, width=500, height=500, borderwidth=5, highlightbackground='black', background='white')
canvas.grid(column=5, row=5, sticky=(N, W, E, S))
canvas.bind("<Button-1>", XY)
canvas.bind("<B1-Motion>", AddLine)
canvas.bind("<Button-2>", AddNode)


#---- space out the widgets
for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

#---- start main event loop
root.mainloop()
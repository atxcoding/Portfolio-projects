class Vertex:
    def __init__(self, name, discovered=0, finished=0, explored = False, distance = 0, predecessor = 0):
        self.name = name
        self.explored = explored
        self.distance = distance
        self.d = discovered
        self.f = finished
        self.p = predecessor


graph1 = {'r': [('s',0), ('v',0)],
         's': [('r',0), ('w',0)],
         't': [('u',0), ('w',0), ('x',0)],
         'u': [('t',0), ('x',0), ('y',0)],
         'v': [('r',0)],
         'w': [('s',0), ('t',0), ('x',0)],
         'x': [('t',0), ('u',0), ('w',0), ('y',0)],
         'y': [('x',0), ('u',0)]
         }

V1 = {'r': Vertex('r'),
     's': Vertex('s'),
     't': Vertex('t'),
     'u': Vertex('u'),
     'v': Vertex('v'),
     'w': Vertex('w'),
     'x': Vertex('x'),
     'y': Vertex('y')
     }



graph2 = {'u': [('v',0), ('x',0)],
         'v': [('y',0)],
         'w': [('y',0), ('z',0)],
         'x': [('v',0)],
         'y': [('x',0)],
         'z': [('z',0)]
         }

V2 = {'u': Vertex('u'),
     'v': Vertex('v'),
     'w': Vertex('w'),
     'x': Vertex('x'),
     'y': Vertex('y'),
     'z': Vertex('z')
     }

graph3 = {'a': [('c',0), ('d',0)],
         'b': [('d',0)],
         'c': [('d',0), ('e',0)],
         'd': [],
         'e': [('h',0)],
         'f': [('e',0), ('g',0)],
         'g': [('h',0)],
         'h': [],
         'i': []
         }

V3 = {'a': Vertex('a'),
     'b': Vertex('b'),
     'c': Vertex('c'),
     'd': Vertex('d'),
     'e': Vertex('e'),
     'f': Vertex('f'),
     'g': Vertex('g'),
     'h': Vertex('h'),
     'i': Vertex('i')
     }

#-------------------------------------------------------
def Traversal(graph, V, start, type ):
    V[start].explored = True
    frontier = []
    frontier.append(start)
    while frontier:
        # ---- remove a vertex from the frontier
        if type == 'BFS':
            pvertex = frontier.pop(0)  # like a queue
        else:
            pvertex = frontier.pop()  # like a stack

        # ---- 'visit' the vertex
        print(pvertex)
        # ---- expand this vertex by iterating over its neighbors
        for edge in graph[pvertex]:
            vertex = edge[0]
            # ---- check if the vertex is not already explored
            if V[vertex].explored == False:
                # ---- mark the vertex as 'explored'
                V[vertex].explored = True
                V[vertex].distance = V[pvertex].distance + 1
                V[vertex].p = pvertex
                # ---- put vertex in frontier
                frontier.append(vertex)

def PrintPath(graph, V, start, end):
    if start == end:
        print(start)
    elif V[end].p == 0:
        print("you can't get there from here")
    else:
        PrintPath(graph, V, start, V[end].p)
        print(end)

#-------------------------------------------------------
time = 0
top_sort = []
def TraversalDFS(graph, V):
    global time
    time = 0
    for v in V:
        if V[v].explored == False:
            V[v].explored = True
            TraversalDFS_Recur(graph, V, v)

def TraversalDFS_Recur(graph, V, v):
    global time, top_sort
    time = time + 1
    V[v].d = time
    print(v)
    # Recur for all the vertices adjacent to this vertex
    for edge in graph[v]:
        vertex = edge[0]
        # ---- check if the vertex is not already explored
        if V[vertex].explored == False:
            # ---- mark the vertex as 'explored'
            V[vertex].explored = True
            TraversalDFS_Recur(graph, V, vertex)
    time = time + 1
    V[v].f = time
    top_sort.append(v)

#-------------------------------------------------------

Traversal(graph1, V1, 's', 'BFS')
print('-----------------------')
PrintPath(graph1, V1, 's', 'y')
print('-----------------------')
#Traversal(graph2, V2, 'u', 'DFS')
#print('-----------------------')
TraversalDFS(graph2, V2)
print('-----------------------')
print(top_sort)
top_sort.reverse()
print(top_sort)
print('-----------------------')
for v in V2:
    print("vertex: ", V2[v].name, V2[v].d, V2[v].f)
    #print("vertex: ", V[v].name, V[v].distance)

top_sort = [(V2[v].f, V2[v].name) for v in V2]
ts = sorted(top_sort, reverse=True)
print(ts)

print('-----------------------')
TraversalDFS(graph3, V3)
top_sort = [(V3[v].f, V3[v].name) for v in V3]
ts = sorted(top_sort, reverse=True)
print(ts)
print('-----------------------')

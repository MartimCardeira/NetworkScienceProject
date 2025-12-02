import random

import numpy as np
import xgi
import matplotlib.pyplot as plt

#my model: set amount of triangles,
#add 2 new nodes, connect to existing node with probability p
#add 1 new node, connect to either end of existing edge with probability 1 - p

p = 1.0
target_triangles = 30

SC = xgi.SimplicialComplex()
nodes = 3
triangles = 1
SC.add_nodes_from([0,1,2])
SC.add_simplex([0,1,2])

while triangles < 20:
    roll = np.random.random()

    if roll <= p: #add 2 nodes
        target_node = np.random.randint(nodes)
        SC.add_node(nodes)
        nodes += 1
        SC.add_node(nodes)
        nodes += 1
        SC.add_simplex([target_node, nodes-1, nodes-2])

    else: #add 1 node
        candidate_edges = SC.edges.filterby("order", 1).members()
        target_node1, target_node2 = random.choice(candidate_edges)
        SC.add_node(nodes)
        nodes += 1
        SC.add_simplex([target_node1, target_node2, nodes - 1])

    triangles += 1

xgi.draw(SC)
plt.show()

import pickle
from DMN2 import Node,Tree
import numpy as np
import DMN2 




with open('root.pkl','rb') as f:
    output = pickle.load(f)

x = DMN2.read_data('Dummy_Data.csv')



for i in range(10):
    p1 = x[i][0]
    p2 = x[i][1]

    pdns = x[i][2]

    net = Tree(5)

    #net.BFS(output,display_layer = True)


    net.FeedForward(output,p1,p2)


    print(output.compliance,pdns)

import numpy as np
import random
import csv
import sympy as sym
from math import ceil
import time


class Node:
    def __init__(self,compliance):
        self.left = None
        self.right = None
        self.compliance = compliance
        
        self.error = np.zeros(6)

        self.layer = 0
        self.index = 0
        
        self.f = 0
        self.weight = 0
        self.activation = 0
        #self.activation = (self.weight)
        
class Tree: 
    def __init__(self,layers):
        self.layers = layers

    def initialise(self):
        queue = []
        rootNode = Node([0]*6)
        
        rootNode.index = 1
        rootNode.layer = 0
    
        queue.append(rootNode)
        for i in range(1,self.layers + 1):
            counter = 0
            for j in range(1,len(queue) + 1):
                node = queue.pop(0)

                node.left = Node(np.array([i]*6))
                node.right = Node(np.array([i]*6))
                
                node.left.layer = i
                node.right.layer = i
                
                node.left.index = j + counter 
                node.right.index = j + counter +1
                counter +=1
                
                
                node.left.weight = random.random()
                node.right.weight = random.random()
                
                queue.append(node.left)
                queue.append(node.right)
        return rootNode
    
    # phase1 and phase2 are Node() objects
    # inputs the 2 phases into the bottom layer            
    def initialise_bottom_layer(self,root,phase1,phase2):
        assert type(phase1) == Node
        assert type(phase2) == Node
        
        if root is None:
            return
        queue = [root]
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

            if node.layer == self.layers:
                if node.index%2 == 1:
                    node.compliance = phase1.compliance
                    node.f = phase1.f
                else:
                    node.compliance = phase2.compliance
                    node.f = phase2.f

    def BFS(self,root,display_layer = False):
        if root is None:
            return
        queue = [root]
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

            
            print((node.layer,node.index))
            if display_layer:
                if node.layer == self.layers:
                    print(node.error)
                    #print(node.index)

    def DFS(self,rootnode):
      
        if rootnode is None:
            return
       
    
        self.DFS(rootnode.left)
        self.DFS(rootnode.right)
        if rootnode.layer == self.layers:
            print(rootnode.weight)
        #print((rootnode.layer,rootnode.index), end =' ')

    # performs a DFS in the binary tree to update the network bottom-up    
    def homogenise_system(self,rootnode):
        if rootnode is None:
            return
        self.homogenise_system(rootnode.left)
        self.homogenise_system(rootnode.right)

        
        if rootnode.left is not None and rootnode.right is not None:
            # computes the matrix at the parent node 
            # updates the weight of the parent node based on its child's nodes
            #rootnode.weight = rootnode.left.weight + rootnode.right.weight
            
             # very strange definition of f
            #rootnode.left.f = rootnode.left.weight/rootnode.weight
            #rootnode.right.f = rootnode.right.weight/rootnode.weight
            
            # computes the matrix at the parent node 
            # updates the weight of the parent node based on its child's nodes
            rootnode.weight = rootnode.left.weight + rootnode.right.weight
            
            # very strange definition of f
            rootnode.left.f = rootnode.left.weight/rootnode.weight
            rootnode.right.f = rootnode.right.weight/rootnode.weight
            rootnode.compliance = homogenise(rootnode.left,rootnode.right)
        
            
           
    # Inputs phase1 and phase2 compliances matrices to the bottom layer, and  
    # outputs the final compliance matrix
    def FeedForward(self,root,phase1,phase2):
        # intialise_bottom_layer, just sets the input. Now once we have the input, can
        # feed it forward
        self.initialise_bottom_layer(root,phase1,phase2)
        self.homogenise_system(root)

    # backpropagates the error into the bottom layer, for a single example
    def BackProp(self,P1,P2,D_dns,D_r):
        # now have the output of the net 
        
        self.FeedForward(D_r,P1,P2)
        
        D_r.error += calc_error_0(D_dns.compliance,D_r.compliance)

        # use BFS to search the tree top to bottom. At each node, finds the error of the 
        # children node
        if D_r is None:
            return
        queue = [D_r]
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
         
            if node.layer   < self.layers-1:
                
                # REMEMBER ABOUT +=. ERROR NEEDS TO BE RESET AFTER EACH 
                # MINIBATCH
                calc_error(node)
                
    
    # apply the backpropagation algorithm for a minibatch to 
    # update the systems weights by a gradient which comes from 
    # several training examples. D_r is the rootnode which
    # effectively holds all the information in the system.
    
    def update_mini_batch(self, mini_batch,D_r,learning_rate):
        # mini_batch is a list of training examples.
        # i.e [(p1, p2, p_dns), () , ]
        # each p1 is in vectorised form
        M = len(mini_batch) 
        nabla_C = np.zeros(2**self.layers)
        
        
            
        #convert elements in mini_batch to node objects
       
        D_r.error = np.zeros(6)
        #print(D_r.error)
        for index,example in enumerate(mini_batch):
            
            self.FeedForward(D_r,example[0],example[1])
            D_r.error += calc_error_0(example[2].compliance,D_r.compliance)
            

            #self.BackProp(example[0],example[1],example[2],D_r)
        D_r.error = D_r.error/M
        
        if D_r is None:
            return
        queue = [D_r]
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
         
            if node.layer   < self.layers-1:
                
                # REMEMBER ABOUT +=. ERROR NEEDS TO BE RESET AFTER EACH 
                # MINIBATCH
                calc_error(node)
                


           
        nabla_C = self.calc_cost_gradient(D_r)/(2*M)



        self.update_weights(D_r,learning_rate,nabla_C)    
       
        #self.homogenise_system(D_r)
    

    def update_weights(self,rootnode,learning_rate,nabla_C):
            if rootnode is None:
                return
            self.update_weights(rootnode.left,learning_rate,nabla_C)
            self.update_weights(rootnode.right,learning_rate,nabla_C)
            if rootnode.left == None and rootnode.right == None:
                rootnode.weight += -learning_rate * nabla_C[rootnode.index - 1]
                
            elif rootnode.left is not None and rootnode.right is not None:
                # computes the matrix at the parent node 
                # updates the weight of the parent node based on its child's nodes
                rootnode.weight = rootnode.left.weight + rootnode.right.weight
                
                # very strange definition of f
                rootnode.left.f = rootnode.left.weight/rootnode.weight
                rootnode.right.f = rootnode.right.weight/rootnode.weight
                


    def SGD(self,epochs,D_r,mini_batch_size,training_data,learning_rate,track_cost= False):
        cost_array = []
        self.update_weights(D_r,0,np.zeros(2**self.layers))
        for i in range(epochs):
            random.shuffle(training_data)
            c=0
            
            for j in range(0,len(training_data),mini_batch_size):
                mini_batch = training_data[j:j + mini_batch_size] 
                #print(len(mini_batch))
                self.update_mini_batch(mini_batch,D_r,learning_rate)
                
            if track_cost == True:
                for example in training_data:
                    self.FeedForward(D_r,example[0],example[1])
                    c += cost(example[2].compliance,D_r.compliance)
            cost_array.append(c/len(training_data))
            print(f'Current Cost is {c/len(mini_batch)}')
            print(f'Epoch {i} is complete')
            
        return cost_array
    # uses a BFS to sum over all the errors and 
    # perform the correct differentiations to output 
    # the gradient of the cost function. Order in whihc you travserse,
    # doesn't matter
    def calc_cost_gradient(self,root):
        sum1,sum2 = 0,0
        vect_sum1,vect_sum2 = np.zeros(2**self.layers),np.zeros(2**self.layers)
        
        bottom_layer = []
        
        for i in range(1,2**self.layers + 1):
            bottom_layer.append( self.fetch_node(root,self.layers,i))
        
        

        if root is None:
            return
        queue = [root]
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

            if node.layer < self.layers:
               
                df = differentiate_fraction(node,bottom_layer)
                #start = time.time()
                #sums take up all the time
                sum1 += np.dot(node.error,differentiate_parent(node,'f1'))
                sum2 += np.dot(node.error,differentiate_parent(node,'f2'))
                
                #print(time.time()-start)
                vect_sum1 += df[0]
                vect_sum2 += df[1]
                
       
        bottom_activations = relu_prime(np.array([node.weight for node in bottom_layer]))
       
        res = sum1 * vect_sum1 + sum2 * vect_sum2

        #hadamard product
       
        return  np.multiply(res,bottom_activations)



            
    # uses a BFS to return a node from a specific position
    def fetch_node(self,rootnode,layer_num,index):
        if rootnode is None:
            return
        queue = [rootnode]
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

            if (node.layer ==layer_num) and (node.index == index) :
                return node
            
   

# phase1 and phase2 are node classes
# node.compliance is the compliance matrix of vectorised form:
# [D_11, D_12, D_13, D_22, D_23, D_33]
def homogenise(phase1,phase2):
    # follows the definiton of the homogenisation procedure as in the paper  
    D_r = np.zeros(6)
    
    gamma = phase1.f * phase2.compliance[0] + phase2.f * phase1.compliance[0]
    
    D_r[0] = (phase1.compliance[0] * phase1.compliance[0])/gamma
    D_r[1] = (phase1.f * phase1.compliance[1] * phase2.compliance[0] + phase2.f * phase2.compliance[1] * phase1.compliance[0])/gamma
    D_r[2] = (phase1.f * phase1.compliance[2] * phase2.compliance[0] + phase2.f * phase2.compliance[2] * phase1.compliance[0])/gamma
    D_r[3] = phase1.f * phase1.compliance[3] + phase2.f * phase2.compliance[3] - (1/gamma) * (phase1.f * phase2.f) * (phase1.compliance[1] - phase2.compliance[1])**2
    D_r[4] = phase1.f * phase1.compliance[4] + phase2.f * phase2.compliance[4] - (1/gamma) * (phase1.f * phase2.f) * (phase1.compliance[2]- phase2.compliance[2]) * (phase1.compliance[1] - phase2.compliance[1])
    D_r[5] = phase1.f * phase1.compliance[5] + phase2.f * phase2.compliance[5] - (1/gamma) * (phase1.f * phase2.f) * (phase1.compliance[2]- phase2.compliance[2])**2

    return D_r


def convert_vectorised(vectorised_compliance_matrix):
    assert(len(vectorised_compliance_matrix) == 6)
   
    
    compliance_3x3 = np.array([[vectorised_compliance_matrix[0], vectorised_compliance_matrix[1], vectorised_compliance_matrix[2]],
                               [vectorised_compliance_matrix[1],vectorised_compliance_matrix[3],vectorised_compliance_matrix[4]],
                               [vectorised_compliance_matrix[2],vectorised_compliance_matrix[4],vectorised_compliance_matrix[5]]
                                ])

    return compliance_3x3

def convert_matrix(matrix_compliance_matrix):
    assert(np.shape(matrix_compliance_matrix) == (3,3))

    compliance_vector = np.zeros((1,6))
    
    compliance_vector[0,0] = matrix_compliance_matrix[0][0]
    compliance_vector[0,1] = matrix_compliance_matrix[0][1]
    compliance_vector[0,2] = matrix_compliance_matrix[0][2]
    compliance_vector[0,3] = matrix_compliance_matrix[1][1]
    compliance_vector[0,4] = matrix_compliance_matrix[1][2]
    compliance_vector[0,5] = matrix_compliance_matrix[2][2]
    return compliance_vector.flatten()

def relu(x):
	return max(0.0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def read_data(file_name):
    data_array = []
    with open(file_name,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            
            f = np.array(row[0:2],dtype=float)
            p1 = np.array(row[2:8],dtype = float)
            p2 = np.array(row[8:14],dtype = float)
            p_dns = np.array(row[14:20],dtype = float)
            data_array.append([f,p1,p2,p_dns])
            
    return data_array

            
# takes a single training example
# outputs the contribution of the example to the cost function
# inputs are in the matrix form
def cost(D_dns,D_h):
    if np.shape(D_dns) ==  (3,3):
        # checks whihch input has been supplied
        return 0.5 * np.linalg.norm(D_dns-D_h)**2 / (np.linalg.norm(D_dns)**2)
    elif np.shape(D_dns) == (6,):
        return 0.5 * np.linalg.norm(convert_vectorised(D_dns)- convert_vectorised(D_h))**2 / (np.linalg.norm(convert_vectorised(D_dns))**2)
     
        
def calc_error_0(D_dns,D_h):  
    if np.shape(D_dns) == (3,3):
        norm_factor = np.linalg.norm(D_dns)
        
        return 0.5 * (convert_matrix(D_h) - convert_matrix(D_dns))/norm_factor**2
        
    elif np.shape(D_dns) == (6,):
        norm_factor = np.linalg.norm(convert_vectorised(D_dns))
        
        return (D_h - D_dns) /norm_factor**2
        
    

# updates the error vector of the children nodes,
# given the input of parent. Doesn't need child 
# as input since they are related by parent.left 
# and parent.right
def calc_error(parent):  
    if type(parent) != Node:
        raise TypeError('Parent and child must be Node types')
    
    element_map = {'0':'11', '1' :'12', '2' : '13', '3': '22', '4' : '23', '5' : '33' }

    children = [parent.left, parent.right]
    
    for child_index, child in enumerate(children,start = 1):
        
        for i in range(6):
            
           
            diff = differentiate_parent(parent, f'D_{child_index}_{element_map[str(i)]}')
           
            child.error[i] = np.dot(parent.error, diff)
    #print((parent.left.layer,parent.left.index), ' ', (parent.right.layer,parent.right.index))

# differentiates the parent node w.r.t a particular 
# element of child compliance. Can either be 1st or 2nd child.
# Notation is the same as in the paper

def differentiate_parent(parent,element):
    

    if type(parent) != Node:
        raise TypeError('Parent and child must be Node types')
    

    allowed_elements = ['D_1_11', 'D_1_12', 'D_1_13', 'D_1_22', 'D_1_23', 'D_1_33',
                        'D_2_11', 'D_2_12', 'D_2_13', 'D_2_22', 'D_2_23', 'D_2_33',
                        'f1','f2']
    assert element in allowed_elements, 'Incorrect symbol used'


    child1= parent.left
    child2 = parent.right

    f1,f2 = sym.symbols('f1 f2')
    
    D_1_11, D_1_12, D_1_13, D_1_22, D_1_23, D_1_33 = sym.symbols('D_1_11 D_1_12 D_1_13 D_1_22 D_1_23 D_1_33')
    D_2_11, D_2_12, D_2_13, D_2_22, D_2_23, D_2_33 = sym.symbols('D_2_11 D_2_12 D_2_13 D_2_22 D_2_23 D_2_33')
    
   
    gamma = (f1* D_2_11 + f2 * D_1_11)
    
    
    D_r_11 = (D_1_11 * D_1_11)/gamma
    D_r_12 = (f1 * D_1_12 * D_2_11 + f2 * D_2_12 * D_1_11)/gamma
    D_r_13 = (f1 * D_1_13 * D_2_11 +f2 * D_2_13 * D_1_11)/gamma
    D_r_22 = f1 * D_1_22 + f2 * D_2_22- (1/gamma) * (f1 *f2) * (D_1_12 - D_2_12)**2
    D_r_23 = f1 * D_1_23 + f2 * D_2_23 - (1/gamma) * (f1 *f2) * (D_1_13- D_2_13) * (D_1_12 - D_2_12)
    D_r_33 = f1 * D_1_33 + f2 * D_2_33 - (1/gamma) * (f1 *f2) * (D_1_13- D_2_13)**2
    

    D_r = np.array([D_r_11,D_r_12,D_r_13,D_r_22,D_r_23,D_r_33])

    # copy just makes a placeholder so that the array holds
    # the same type 
    derivative_symbolic = D_r.copy()
    
   

    # symbolically differentiates the elements of D_r
    # and stores in derivative_symbolic
    for index,function in enumerate(D_r):
        derivative_symbolic[index] = function.diff(element)
    





    derivative_numeric = np.zeros(6)
   



    
    # subsitsutes the actual values and returns the numeric
    # derivative 
    for index,function in enumerate(derivative_symbolic):
        derivative_numeric[index] = function.subs({f1: child1.f, f2:child2.f,
                                                   D_1_11: child1.compliance[0], D_1_12 :child1.compliance[1], D_1_13:child1.compliance[2],
                                                   D_1_22 : child1.compliance[3], D_1_23 : child1.compliance[4], D_1_33 : child1.compliance[5], 
                                                   D_2_11: child2.compliance[0], D_2_12 :child2.compliance[1], D_2_13:child2.compliance[2],
                                                   D_2_22 : child2.compliance[3], D_2_23 : child2.compliance[4], D_2_33 : child2.compliance[5]}
        )
    
    
    return derivative_numeric

# differentiates the volume fraction of the children
#  node with respect to a series of weights in the bottom layer, with 
# input of a parent node. Returns an array of the two derivatives 
def differentiate_fraction(parent,bottom_layer_array):
    assert type(parent) == Node
    assert type(bottom_layer_array) == list
    
    
    children = [parent.left, parent.right]
    num_bottom_layer_elements = len(bottom_layer_array)
    
    
    df_dw = np.array([np.zeros(num_bottom_layer_elements),np.zeros(num_bottom_layer_elements)])
    



    for index,child in enumerate(children):
        f_grad = np.zeros(num_bottom_layer_elements)
        
        for node_index,node_j in enumerate(bottom_layer_array):
            dw_parent_dw_j = differentiate_weight(parent,node_j)
            f_grad[node_index] = (1/parent.weight) * (differentiate_weight(child,node_j) -  child.f * dw_parent_dw_j)
           
        df_dw[index] = f_grad
    return df_dw
       
        
# differentiates a weight in the network wrt 
# a bottom layer weight. node_k is in the network and
# node_j is the jth weight in the bottom layer. Must be in bottom
# layer. Fromula straight from the paper. Feels very dody. Need to make
# sure that it works correctly. Possible source of error.
def differentiate_weight(node_k,node_j):
    assert type(node_j) == Node
    assert type(node_k) == Node

    if node_k.index == ceil(node_j.index/(2**(node_j.layer-node_k.layer))):
        return 1.0
    else:
        return 0.0
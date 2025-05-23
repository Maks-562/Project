import numpy as np
import random
import csv
import symengine as sym
from math import ceil
import time
from tqdm import tqdm




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
      
class Tree: 
    def __init__(self,layers):
        self.layers = layers

    def initialise(self):
        queue = []
        rootNode = Node([0]*6)
        
        # rootnode is the first node in the tree, i.e the output

        # defines the index and layer of the root node
        rootNode.index = 1
        rootNode.layer = 0
    
        queue.append(rootNode)
        for i in range(1,self.layers + 1):
            counter = 0
            for j in range(1,len(queue) + 1):
                node = queue.pop(0)

                # links parent nodes to child nodes
                node.left = Node(np.array([i]*6))
                node.right = Node(np.array([i]*6))
                
                # sets the layer number of the child nodes
                node.left.layer = i
                node.right.layer = i
                
                # sets the index of the child nodes. Slighlty convoluted, but
                # the labelling scheme works.
                
                node.left.index = j + counter 
                node.right.index = j + counter +1
                counter +=1
                
                # activations are the z's in the paper. Values
                # are taken from equation (47).

                if node.left.layer == self.layers:
                    node.left.activation = random.uniform(0.2,0.8)
                if node.right.layer == self.layers: 
                    node.right.activation = random.uniform(0.2,0.8)
                
                queue.append(node.left)
                queue.append(node.right)
        
        # returns the root of the tree. This variable holds all the information
        # of the tree, since can access all other nodes through .left and .right . 
        
        return rootNode


    # phase1 and phase2 are compliance matrices in the vectoreised form.
    # inputs the 2 phases into the bottom layer.            
    def initialise_bottom_layer(self,root,phase1,phase2):
       
        if root is None:
            return
        queue = [root]
        
        # uses the same idea of a queue as above.
        
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

            if node.layer == self.layers:
                # Point is to set the phases at the bottom layer, 
                # to be alternating. See Fig.3 in the paper (Layer N)
                
                if node.index%2 == 1:
                    node.compliance = phase1
                else:
                    node.compliance = phase2
                    

    # A Breadth First Search to search all the elements in the tree,
    # from top to bottom. See https://www.geeksforgeeks.org/introduction-to-binary-tree/
    # Only used for debugging purposes, to view parameters in certain layers etc.
    # I've used the idea of the BFS throughout the code.

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

            if display_layer:
                if node.layer == self.layers:
                    # views the coordinates and compliance matrices of 
                    # all the nodes in the bottom layer, if display_layer is True
                    print(node.layer,node.index,node.compliance)
                    
            
    # A Depth First Search to search all the elements in the tree, in a different manner to 
    # the BFS. This searches the tree from bottom to top, through a recursive definition.
    # Implementation also taken from https://www.geeksforgeeks.org/introduction-to-binary-tree/
    # I've also used the idea of the DFS throughout the code.   

    def DFS(self,rootnode):
      
        if rootnode is None:
            return
       
    
        self.DFS(rootnode.left)
        self.DFS(rootnode.right)
        if rootnode.layer == self.layers:
            # prints the coordinates and compliance matrices of all the nodes in the bottom layer.
            # can show other parameters too, i.e rootnode.activation etc.
            print(rootnode.layer,rootnode.index,rootnode.compliance)
                    

    # Uses a DFS in the binary tree to homogenise the network, bottom-up .   
    # Be wary that it is recursive, so it may look criptic at first.
    
    def homogenise_system(self,rootnode):
        if rootnode is None:
            return
        
        # recursive part of the function
        self.homogenise_system(rootnode.left)
        self.homogenise_system(rootnode.right)

        # makes sure that the current node is not in the bottom layer.
        # Doesn't make sense to homogenise the bottom layer, since there
        # are no elements below it.
        if rootnode.left is not None and rootnode.right is not None:

            rootnode.compliance = homogenise(rootnode.left,rootnode.right)
            

           
    # Inputs phase1 and phase2 compliances matrices to the bottom layer, and  
    # outputs the final compliance matrix i.e feeedsforward the input
    # and gets the output. The total RVE homogenised compliance matrix is 
    # then stored in root.compliance.
    def FeedForward(self,root,phase1,phase2):
       
        # intialise_bottom_layer, just sets the input. Now once we have the input, can
        # feed it forward
        self.initialise_bottom_layer(root,phase1,phase2)

        self.homogenise_system(root)
        
    # apply the backpropagation algorithm for a minibatch to 
    # update the systems weights by a gradient which comes from 
    # several training examples. D_r is the rootnode which
    # effectively holds all the information in the system.
    
    def update_mini_batch(self, mini_batch,D_r):
        # mini_batch is a list of training examples.
        # i.e [(p1_example1, p2_example1, p_dns_example1), ... , (p1_example_M, p2_example_M, p_dns_example_M) ]
        # each p2,p2,p_dns is the complaince matrix in vectorised form
        
        M = len(mini_batch) 
        
        
        nabla_C = np.zeros(2**self.layers)
        
        
        # makes sure that the error in the output layer is reset after each minibatch
        D_r.error = np.zeros(6)
        
        # loops over all the training examples in the minibatch
        for example in mini_batch:
            
         
            self.FeedForward(D_r,example[0],example[1])

            D_r.error += calc_error_0(example[2],D_r.compliance)

        # the output error for the minibatch is normalised by the number of training examples
        # M    
        
        D_r.error = D_r.error/M
           

        # the lines below are the backpropagation algorithm. 
        # I couldn't figure out a way to write this as a single function, 
        # so it is incorporated directly in the update_mini_batch function.
        
        if D_r is None:
            return
        queue = [D_r]
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
            if node.layer   < self.layers -1:
                # calculates the error of each node in the network
                # until the second last layer. Doesnn't calculate the error
                # for the bottom layer (layer N in Fig.3, because it is not needed in 
                # equation (42) )
                
                calc_error(node)

        # calculates the gradient of the cost function for the minibatch,
        # based on the errors in the system.

        nabla_C = self.calc_cost_gradient_alt(D_r)
   
        return nabla_C/(2*M)


    # Stochastic Gradient Descent algorithm to train the network.
    
    def SGD(self,epochs,D_r,mini_batch_size,training_data,learning_rate,track_error= False):
        error_array = []
        
        # makes sure that each node has .weight and .f attributes 
        # at the beginnig of the training.
        
        self.propagate_weights(D_r) 

        for i in range(0,epochs):
            # stochastic part 
            random.shuffle(training_data)
            
            c=0
            
            nabla_C = np.zeros(2**self.layers)
        
            # goes through the training data in mini batches
            
            for j in tqdm(range(0,len(training_data),mini_batch_size)):
                
                mini_batch = training_data[j:j + mini_batch_size] 
                
                nabla_C = self.update_mini_batch(mini_batch,D_r)
                
                # calculates the regularisation term in the cost function.
                # See equation (34) and (35) in the paper.
                
                L = self.regularisation_term(D_r)
                
                # based on the  gradient of the cost function, updates the activations in 
                # the bottom layer (the network parameters). 
                
                # README: I am not sure about my implementation of the regularisation term. 
                # Hence, for best results lam should be set to 0 such that it doesn't influence 
                # the cost function at all, or should be set to an extremely small value
                # i.e lam ~ 10^-7 I think gives OK results.
                
                self.update_weights(D_r,learning_rate,nabla_C,L,lam =0)    
                
                # since the activations in the bottom layer were updated, the change
                # needs to be popagated/fedforward through the network.
                
                self.propagate_weights(D_r)
                
         
            
            # calculates the error of the network after each epoch
            if track_error == True:
                for example in training_data:
                    self.FeedForward(D_r,example[0],example[1])
                    c += error_relative(example[2],np.round(D_r.compliance,4))
            error_array.append(c/len(training_data) * 100)
           
            print(f'Current Relative error is {error_array[i]:.3f} % ')
            print(f'Epoch {i} is complete')

            if i%15==0 and i >0:

              last15 = error_array[i-15:i]
              std_error = np.std(last15)
              if std_error < 10**-2:
                  print(f' The training process has terminated as the deviation in the last 15 error values is: {std_error:.3f}')
                  return error_array
            
            
        return error_array

    

    def update_weights(self,rootnode,learning_rate,nabla_C,L,lam):
            if rootnode is None:
                return
            self.update_weights(rootnode.left,learning_rate,nabla_C,L,lam)
            self.update_weights(rootnode.right,learning_rate,nabla_C,L,lam)
            if rootnode.left == None and rootnode.right == None:
                    
                rootnode.activation += -learning_rate * nabla_C[rootnode.index - 1] - learning_rate * lam * L[rootnode.index - 1]
    
    # See Equation 33-35 in the paper.
    
    def regularisation_term(self,rootnode):
        bottom_layer = []
     
        for i in range(1,2**self.layers + 1):
            bottom_layer.append( self.fetch_node(rootnode,self.layers,i))
            
        bottom_activations = relu(np.array([node.activation for node in bottom_layer]))
        
        bottom_activations_derivatives = relu_prime(np.array([node.activation for node in bottom_layer]))
        
        L = (np.sum(bottom_activations) - 2**(self.layers-2) ) **2
    
        return 2 * np.sqrt(L) * bottom_activations_derivatives

    # propagates the bottom layer activations, through the network.
    
    def propagate_weights(self,rootnode):
        if rootnode is None:
                return
        self.propagate_weights(rootnode.left)
        self.propagate_weights(rootnode.right)
        
        if rootnode.left is None and rootnode.right is None:
            # see equation (21)
            
            rootnode.weight = relu(rootnode.activation)
           
        else:
            # computes the matrix at the parent node 
            # updates the weight of the parent node based on its child's nodes
                
            # see equation (19) and Fig.4 in the paper.
            rootnode.weight = rootnode.left.weight + rootnode.right.weight
            
            # see equation (23) in the paper. It seems very strange,
            # but works
            rootnode.left.f = rootnode.left.weight/rootnode.weight
            rootnode.right.f = rootnode.right.weight/rootnode.weight

    # The first implementation of calculating the cost gradient.
    # I don't think it works correctly so use calc_cost_gradient_alt instead.
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
            
                sum1 += np.dot(node.error,differentiate_parent(node,'f1')) *df[0]
                sum2 += np.dot(node.error,differentiate_parent(node,'f2')) * df[1]

              
                vect_sum1 += df[0]
                vect_sum2 += df[1]
                
     
        bottom_activations = relu_prime(np.array([node.activation for node in bottom_layer]))
      
        res = sum1 * vect_sum1 + sum2 * vect_sum2
     
        res = np.multiply(res,bottom_activations)
    
     
        return  res

    def calc_cost_gradient_alt(self,root):

      
        bottom_layer = []
        nabla_c = np.zeros(2**self.layers)
        
        # creates a list of all the nodes in the bottom layer
        for i in range(1,2**self.layers + 1):
            bottom_layer.append(self.fetch_node(root,self.layers,i))

        
        # implements equation (42) in the paper.
        for node_j in bottom_layer:
            gradient = 0
            
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
                    children = [node.left,node.right]
                    modes = ['f1','f2']
                    for (child,mode) in zip(children,modes):
                        # alpha_da_df is the first two terms in equation (42).
                        # df_dw is the last term in equation (42), which is defined
                        # in equation (24).
                        
                        alpha_da_df = np.dot(node.error,differentiate_parent(node,mode))
                        df_dw = (1/node.weight) * (differentiate_weight(child,node_j) -  child.f * differentiate_weight(node,node_j))
                        
                        
                        gradient +=  alpha_da_df* df_dw 
            
            # the full form of equation (42)
            
            gradient = gradient * relu_prime(node_j.activation)
            
            nabla_c[node_j.index-1] = gradient     

        return nabla_c
            
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
    # follows the definiton of the homogenisation procedure as in the paper.
    # See equation (8)
    
    # Initialises the homogenised compliance matrix.
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
	return x * (x > 0)

def relu_prime(x):
    return np.where(x > 0, 1, 0)


# used to read in the data.
def read_data(file_name):
    data_array = []
    with open(file_name,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            
         
            p1 = np.array(row[0:6],dtype = float)
            p2 = np.array(row[6:12],dtype = float)
            p_dns = np.array(row[12:18],dtype = float)
            data_array.append([p1,p2,p_dns])
            
    return data_array



# Calculates the error of a single training example, at the output layer.
# D_dns is the training data compliance matrix in vectorised form and the D_h is the
# output of the network in vectorised form (rootnode.compliance). 
# See equation (37).      

def calc_error_0(D_dns,D_h):  
    # these lines are just precautions to make sure that the inputs are in the correct form
    if np.shape(D_dns) == (3,3):
        norm_factor = np.linalg.norm(D_dns)
        
        return 0.5 * (convert_matrix(D_h) - convert_matrix(D_dns))/norm_factor**2


    # See also equation (30) in the paper. The below is the hard coded derivative of 
    # equation (30) in the paper. 
    
    elif np.shape(D_dns) == (6,):
        norm_factor = np.linalg.norm(convert_vectorised(D_dns))
        return (D_h - D_dns) /norm_factor**2

# outputs the error of the network, of a single training example. See
# equation (78) & (79) in the paper.

def error_relative(D_dns,D_h):
    # again have precautions.
    if np.shape(D_dns) ==  (3,3):

        return 0.5 * np.linalg.norm(D_dns-D_h)**2 / (np.linalg.norm(D_dns)**2)
    elif np.shape(D_dns) == (6,):
        return np.linalg.norm(convert_vectorised(D_dns)- convert_vectorised(D_h)) / (np.linalg.norm(convert_vectorised(D_dns)))
     

 
        
# Calculates the error vector of the children nodes,
# given the input of a single parent node. Doesn't need child 
# as input since they are related by parent.left 
# and parent.right. See the differentiate_parent function below for more details.

def calc_error(parent):  
    if type(parent) != Node:
        raise TypeError('Parent and child must be Node types')
    
    element_map = {'0':'11', '1' :'12', '2' : '13', '3': '22', '4' : '23', '5' : '33' }

    children = [parent.left, parent.right]
    
    for child_index, child in enumerate(children,start = 1):
        
        for i in range(6):
            
           
            diff = differentiate_parent(parent, f'D_{child_index}_{element_map[str(i)]}')
           
            child.error[i] = np.dot(parent.error, diff)
    
# differentiates the parent compliance matrix w.r.t a particular 
# element of the child compliance matrix or the child volume fraction.
# This function combines equations (16) and (17) in the paper. It works 
# for both child nodes.
# Notation is the same as in the paper.

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
    
    # forms the actual structure of the parent compliance matrix, based on the children's compliance matrices.
    
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
    # returns the numerical value of the derivative
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
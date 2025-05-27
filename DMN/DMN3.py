import numpy as np
import random
import sympy as sym
import pickle
from math import ceil
# from DMN_funcs import convert_matrix,convert_vectorised,calc_error_0,R,homogenise,error_relative,cost,relu, relu_prime,differentiate_D_wrt_theta
from helperfuncs import *

class Node:
    def __init__(self,compliance):
        self.left = None
        self.right = None
        
        self.compliance = compliance
        self.rotated_compliance = np.zeros(6)


        self.error_delta = np.zeros(6)
        self.error_alpha = np.zeros(6)


        self.layer = 0
        self.index = 0
        
        self.f = 0
        self.weight = 0
        self.activation = 0

        self.theta = 0

        self.delta_eps = np.zeros(3)
        self.delta_sigma =np.zeros(3)
        self.eff_plas_strain = 0
        self.res_strain = np.zeros(3)

        self.eps = np.zeros(3)
        self.sigma =np.zeros(3)


class Tree: 
    def __init__(self,layers):
        self.layers = layers
        self.derivatives_cache = []
    def initialise(self):
        queue = []
        rootNode = Node([0]*6)
        
        # rootnode is the first node in the tree, i.e the output

        # defines the index and layer of the root node
        rootNode.index = 1
        rootNode.layer = 0


        rootNode.theta = random.uniform(-np.pi/2,np.pi/2)
        # rootNode.theta = np.random.normal(0,0.01)


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
                

                node.left.theta = random.uniform(-np.pi/2,np.pi/2)
                node.right.theta = random.uniform(-np.pi/2,np.pi/2)
                
                # node.left.theta = np.random.normal(0,0.01)
                # node.right.theta = np.random.normal(0,0.01)


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
                    # node.compliance = convert_matrix(R(-node.theta) @ convert_vectorised(phase1) @ R(node.theta))
                    node.compliance = phase1
                    node.rotated_compliance = convert_matrix(R(-node.theta) @ convert_vectorised(phase1) @ R(node.theta))
                    node.res_strain = np.random.uniform(-1,1,(3,))
                    # node.compliance = np.zeros(6)
                    # node.rotated_compliance = phase1
                else:
                    # node.compliance = convert_matrix(R(-node.theta) @ convert_vectorised(phase2) @ R(node.theta))
                    
                    node.compliance = phase2
                    node.rotated_compliance = convert_matrix(R(-node.theta) @ convert_vectorised(phase2) @ R(node.theta))
                    node.res_strain = np.random.uniform(-1,1,(3,))
                    
                    # node.compliance =np.zeros(6)
                    # node.rotated_compliance = phase2
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
                    print(node.layer,node.index,node.compliance,node.theta)
                    
            
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
    
    def homogenise_system(self,rootnode,phase1,phase2):
        if rootnode is None:
            return
        
        # recursive part of the function
        self.homogenise_system(rootnode.left,phase1,phase2)
        self.homogenise_system(rootnode.right,phase1,phase2)


      

        # makes sure that the current node is not in the bottom layer.
        # Doesn't make sense to homogenise the bottom layer, since there
        # are no elements below it  
        
        if rootnode.left is not None and rootnode.right is not None:
            p1 = rootnode.left.rotated_compliance
            p2 = rootnode.right.rotated_compliance
            f1 = rootnode.left.f
            f2 = rootnode.right.f

            rootnode.compliance = homogenise(p1,p2,f1,f2)
            
            # rotation line equation (11) 
            rootnode.rotated_compliance = convert_matrix(R(-rootnode.theta) @ convert_vectorised(rootnode.compliance) @ R(rootnode.theta))
          
    # Inputs phase1 and phase2 compliances matrices to the bottom layer, and  
    # outputs the final compliance matrix i.e feeedsforward the input
    # and gets the output. The total RVE homogenised compliance matrix is 
    # then stored in root.compliance.
    def FeedForward(self,root,phase1,phase2):
       
        # intialise_bottom_layer, just sets the input. Now once we have the input, can
        # feed it forward
        self.initialise_bottom_layer(root,phase1,phase2)

        self.homogenise_system(root,phase1,phase2)



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
        D_r.error_delta = np.zeros(6)
        D_r.error_alpha= np.zeros(6)
        
        # loops over all the training examples in the minibatch
        for example in mini_batch:
            
         
            self.FeedForward(D_r,example[0],example[1])

            D_r.error_delta += calc_error_0(example[2],D_r.rotated_compliance)


        # the output error for the minibatch is normalised by the number of training examples
        # M    
        
        D_r.error_delta = D_r.error_delta /(2*M)

        
        # print(D_r.compliance,D_r.rotated_compliance)
        # print(differentiate_D_wrt_D_r(D_r.theta))
       
        # D_r.error_alpha = differentiate_D_wrt_D_r(D_r)/M
       
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

            if node.layer   < self.layers:
                # print('I am doing node: ',node.layer,node.index) 
                # calc_error_alpha(node)
                # print('I have error alpha before: ',node.error_alpha)
                # node.error_alpha = differentiate_D_wrt_D_r(node)
                self.calc_error_alpha(node)
                
                
                # print('I have error alpha after: ',node.error_alpha)

            # if node.layer >= self.layers:
            #     error_alpha_flag = False
            if 0 <=  node.layer   < self.layers:

            # calculates the error of each node in the network
            # until the second last layer. Doesn't calculate the error
            # for the bottom layer (layer N in Fig.3, because it is not needed in 
            # equation (42) )

                self.calc_error_delta(node)
                # print('I am doing node: ',node.layer,node.index) 
        # calculates the gradient of the cost function for the minibatch,
        # based on the errors in the system.

        nabla_C = self.calc_cost_gradient_alt(D_r)
        # nabla_C = self.calc_cost_gradient(D_r)
    
        nabla_C_theta = self.calc_cost_gradient_angles(D_r)
      
        # print(np.mean(nabla_C),np.mean(nabla_C_theta))
       
        return (nabla_C,nabla_C_theta)


    # Stochastic Gradient Descent algorithm to train the network.
    
    def SGD(self,epochs,D_r,mini_batch_size,training_data,learning_rate,validation_data = None,lam = 0,filename = None):
        error_array = []
        cost_array = []
        error_train = []

        min_validation_error = 10**10
        self.get_derivatives()


        # makes sure that each node has .weight and .f attributes 
        # at the beginnig of the training.
        
        self.propagate_weights(D_r) 
    


        for i in range(0,epochs):
            # stochastic part 
            random.shuffle(training_data)
            
            c=0
            cost_funct = 0
            err_train = 0
            # nabla_C_z = np.zeros(2**self.layers)


            # goes through the training data in mini batches
            
            for j in range(0,len(training_data),mini_batch_size):
                
                mini_batch = training_data[j:j + mini_batch_size] 
                
                nabla_C_z,nabla_C_theta = self.update_mini_batch(mini_batch,D_r)
                
                # calculates the regularisation term in the cost function.
                # See equation (34) and (35) in the paper.
                
                L = self.regularisation_term(D_r)
                
                # based on the  gradient of the cost function, updates the activations in 
                # the bottom layer (the network parameters). 
                
                # README: I am not sure about my implementation of the regularisation term. 
                # Hence, for best results lam should be set to 0 such that it doesn't influence 
                # the cost function at all, or should be set to an extremely small value
                # i.e lam ~ 10^-7 I think gives OK results.
                # print(np.mean(nabla_C_z),np.mean(nabla_C_theta))
                self.update_weights(D_r,learning_rate,nabla_C_z,nabla_C_theta,L,lam =lam)    
                # self.compression(D_r)
                # since the activations in the bottom layer were updated, the change
                # needs to be popagated/fedforward through the network.
                
                self.propagate_weights(D_r)
            # self.compression(D_r)    
         
            # calculates the error of the network after each epoch
            if validation_data is not None:
                for example in validation_data:
                    self.FeedForward(D_r,example[0],example[1])
                    c += error_relative(example[2],D_r.rotated_compliance)

                for example in training_data:
                    self.FeedForward(D_r,example[0],example[1])
                    cost_funct += cost(example[2],D_r.rotated_compliance)   
                    err_train += error_relative(example[2],D_r.rotated_compliance)
            
            error_array.append(c/len(validation_data) * 100)
            cost_array.append(cost_funct/len(training_data) + self.reg_mag(D_r))
            error_train.append(err_train/len(training_data) * 100)
            if i%5==0:
                print(f'Current Relative Validation error is {error_array[i]:.3f} % ')
                print(f'Current Relative Training Error is: {error_train[i]:.3f} % ')
                print(f'Epoch {i} is complete')

            if i%15==0 and i >0:

              last15 = error_array[i-15:i]
              std_error = np.std(last15)
              if std_error < 10**-4:
                  print(f' The training process has terminated as the deviation in the last 15 error values is: {std_error:.3f}')
                  return error_array,cost_array,error_train
            
            
            if error_array[i] < min_validation_error:
                min_validation_error = error_array[i]
                
                metadata = {'N':self.layers, 'Epochs':i, 'learming_rate':learning_rate, 'mini_batch_size':mini_batch_size, 'regularisation':lam, 'num_training_examples':len(training_data),'optimiser':'SGD'}
                
                self.save_model(D_r,metadata,error_train,error_array)
                
        return error_array,cost_array,error_train
            





    def ADAM(self,epochs,D_r,mini_batch_size,training_data,learning_rate,validation_data = None, lam = 0,beta = 0,gamma = 0,filename= None):
      
        error_array = []
        cost_array = []
        error_train = []
     

        min_validation_error = 10**10
        
        self.get_derivatives()



        self.propagate_weights(D_r) 

       

        for i in range(0,epochs):
            # stochastic part 
            random.shuffle(training_data)
            
            c=0
            cost_funct = 0
            err_train = 0
            # nabla_C_z = np.zeros(2**self.layers)
        
            # goes through the training data in mini batches
            
            m1= np.zeros(2**self.layers)
            m2 = np.zeros((self.layers+1,2**self.layers))
            
            v1 = np.zeros_like(m1)
            v2 = np.zeros_like(m2)

            # if i ==100:
            #     learning_rate/= 10
            # if i == 500:
            #     learning_rate/= 10

            # if i == 1000:
            #     learning_rate/= 10

            for j in range(0,len(training_data),mini_batch_size):
                
                mini_batch = training_data[j:j + mini_batch_size] 
                
                nabla_C_z,nabla_C_theta = self.update_mini_batch(mini_batch,D_r)

                
                m1  = beta*m1 + (1-beta)*nabla_C_z
                m2  = beta*m2 + (1-beta)*nabla_C_theta
                v1  = gamma*v1 + (1-gamma)*nabla_C_z**2
                v2  = gamma*v2 + (1-gamma)*nabla_C_theta**2
                m1_hat = m1/(1-beta**(i+1))
                m2_hat = m2/(1-beta**(i+1))
                v1_hat = v1/(1-gamma**(i+1))
                v2_hat = v2/(1-gamma**(i+1))



                nabla_C_z = m1_hat/(np.sqrt(v1_hat) + 10**-8)
                nabla_C_theta = m2_hat/(np.sqrt(v2_hat) + 10**-8)


                # calculates the regularisation term in the cost function.
                # See equation (34) and (35) in the paper.
                
                L = self.regularisation_term(D_r)
                
                # based on the  gradient of the cost function, updates the activations in 
                # the bottom layer (the network parameters). 
                
                # README: I am not sure about my implementation of the regularisation term. 
                # Hence, for best results lam should be set to 0 such that it doesn't influence 
                # the cost function at all, or should be set to an extremely small value
                # i.e lam ~ 10^-7 I think gives OK results.
                # print(np.mean(nabla_C_z),np.mean(nabla_C_theta))
                self.update_weights(D_r,learning_rate,nabla_C_z,nabla_C_theta,L,lam =lam)    
                # self.compression(D_r)
                # since the activations in the bottom layer were updated, the change
                # needs to be popagated/fedforward through the network.
                
                self.propagate_weights(D_r)
            # self.compression(D_r)    
         
            # calculates the error of the network after each epoch
            if validation_data is not None:
                for example in validation_data:
                    self.FeedForward(D_r,example[0],example[1])
                    c += error_relative(example[2],D_r.rotated_compliance)

                for example in training_data:
                    self.FeedForward(D_r,example[0],example[1])
                    cost_funct += cost(example[2],D_r.rotated_compliance)   
                    err_train += error_relative(example[2],D_r.rotated_compliance)
            error_array.append(c/len(validation_data) * 100)
            cost_array.append(cost_funct/len(training_data) + self.reg_mag(D_r))
            error_train.append(err_train/len(training_data) * 100)
            if i%5==0:
                print(f'Current Relative Validation error is {error_array[i]:.3f} % ')
                print(f'Current Relative Training Error is: {error_train[i]:.3f} % ')
                print(f'Epoch {i} is complete')

            if i%15==0 and i >0:

              last15 = error_array[i-15:i]
              std_error = np.std(last15)
              if std_error < 10**-4:
                  print(f' The training process has terminated as the deviation in the last 15 error values is: {std_error:.3f}')
                  return error_array

            if error_array[i] < min_validation_error:
                min_validation_error = error_array[i]
                
                metadata = {'N':self.layers, 'Epochs':i, 'learming_rate':learning_rate, 'mini_batch_size':mini_batch_size, 'regularisation':lam, 'num_training_examples':len(training_data),'optimiser':'ADAM'}
                
                self.save_model(D_r,metadata,error_train,error_array,filename=filename)
                
     
        return error_array,cost_array,error_train
        
    

    def update_weights(self,rootnode,learning_rate,nabla_C_z,nabla_C_theta,L,lam):
            if rootnode is None:
                return
            self.update_weights(rootnode.left,learning_rate,nabla_C_z,nabla_C_theta,L,lam)
            self.update_weights(rootnode.right,learning_rate,nabla_C_z,nabla_C_theta,L,lam)
            
            rootnode.theta += -learning_rate * nabla_C_theta[rootnode.layer,rootnode.index - 1]
           
            if rootnode.left is None and rootnode.right is None:
                rootnode.activation += -learning_rate * nabla_C_z[rootnode.index - 1] - learning_rate * lam * L[rootnode.index - 1]
                # print(learning_rate * nabla_C_z[rootnode.index - 1],rootnode.activation)
    # See Equation 33-35 in the paper.
    
    def regularisation_term(self,rootnode):
        bottom_layer = []

        for i in range(1,2**self.layers + 1):
            bottom_layer.append( self.fetch_node(rootnode,self.layers,i))
            
        bottom_activations = relu(np.array([node.activation for node in bottom_layer]))
        # bottom_activations = sigmoid(np.array([node.activation for node in bottom_layer]))
        bottom_activations_derivatives = relu_prime(np.array([node.activation for node in bottom_layer]))
        # bottom_activations_derivatives = sigmoid_prime(np.array([node.activation for node in bottom_layer]))
        L = (np.sum(bottom_activations) - 2**(self.layers-2) ) **2
       
        return 2 * np.sqrt(L) * bottom_activations_derivatives

    def reg_mag(self,rootnode):
        bottom_layer = []

        for i in range(1,2**self.layers + 1):
            bottom_layer.append( self.fetch_node(rootnode,self.layers,i))
        bottom_activations = relu(np.array([node.activation for node in bottom_layer]))
        L = (np.sum(bottom_activations) - 2**(self.layers-2) ) **2
        return L
    # propagates the bottom layer activations, through the network.
    
    def propagate_weights(self,rootnode):
        if rootnode is None:
                return
        self.propagate_weights(rootnode.left)
        self.propagate_weights(rootnode.right)
        
        if rootnode.left is None and rootnode.right is None:
            # see equation (21)
           
            rootnode.weight = relu(rootnode.activation) + 10**-8
            # rootnode.weight = sigmoid(rootnode.activation)
        else:
            # computes the matrix at the parent node 
            # updates the weight of the parent node based on its child's nodes
                
            # see equation (19) and Fig.4 in the paper.
            rootnode.weight = rootnode.left.weight + rootnode.right.weight
            
            # see equation (23) in the paper. It seems very strange,
            # but works
            if rootnode.weight ==0:
                print('ERROR:',rootnode.weight,rootnode.f,rootnode.layer,rootnode.index)
                print('CHILDREN:',rootnode.left.weight,rootnode.right.weight,rootnode.left.layer,rootnode.left.index)
                print('CHILDREN:',rootnode.left.activation,rootnode.right.activation,rootnode.left.f,rootnode.right.f)
            rootnode.left.f = rootnode.left.weight/rootnode.weight
            # rootnode.right.f = rootnode.right.weight/rootnode.weight
            rootnode.right.f = 1 - rootnode.left.f
  
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
                        
                        alpha_da_df = node.error_alpha @ self.differentiate_parent_sympy(node,mode,diff_wrt_rot=True)
                        # alpha_da_df = node.error_alpha @ differentiate_parent_jax(node,mode,diff_wrt_rot=True)
                        
                        # df_dw = (1/node.weight) * (differentiate_weight(child,node_j) -  child.f * differentiate_weight(node,node_j))
                        df_dw = (1/node.weight) * (differentiate_weight(child,node_j) -  child.f * differentiate_weight(node,node_j))
                        
                       
                        gradient +=  alpha_da_df* df_dw 
            
            # the full form of equation (42)
            
            gradient = gradient * relu_prime(node_j.activation)
            
            # gradient = gradient * sigmoid_prime(node_j.activation)
            nabla_c[node_j.index-1] = gradient     
        # print(nabla_c)
        return nabla_c


    def calc_cost_gradient_angles(self,root):       
        cost = np.zeros((self.layers+1,2**self.layers))
        # print(self.layers)
        if root is None:
            return
        queue = [root]
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            # print(node.layer)
            cost[node.layer,node.index-1] = node.error_delta @ differentiate_D_wrt_theta(node.compliance,node.theta)
            # print(np.dot(node.error_delta ,differentiate_D_wrt_theta(node.rotated_compliance,node.theta)))
            # if node.layer == self.layers:
            # print(differentiate_D_wrt_theta(node.rotated_compliance,node.theta))    
                # 
            #     D_temp = convert_vectorised(node.compliance)
            #     D_1  = -R_prime(-node.theta) @ D_temp @ R(node.theta)  
            #     D_2 = R(-node.theta)@D_temp@ R_prime(node.theta)
            #     print(D_1,D_2)
        # print(cost)
        return cost

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
            
    def get_weights(self,rootnode):
        weights = []
        fractions = []
        indices = []
        if rootnode is None:
            return
        queue = [rootnode]
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

            if (node.layer ==self.layers) and (node.layer == self.layers):
                weights.append(node.activation) 
                fractions.append(node.f) 
                indices.append(node.index)
        return np.array(weights),np.array(fractions),np.array(indices)

    
  

    def get_all_nodes(self,rootnode):
        nodes = []
        
        if rootnode is None:
            return
        queue = [rootnode]
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            if node.left is not None:
                nodes.append(node)
            elif rootnode.right is not None:
                nodes.append(node)
        return nodes
 
    def save_model(self,rootnode,metadata,training_error,validation_error,filename = None):
        if filename is None:
            filename = f'models/model_N={self.layers}.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(metadata, f)
            pickle.dump(training_error, f)
            pickle.dump(validation_error, f)
            pickle.dump(rootnode, f)



    def get_derivatives(self):
    
        allowed_elements = ['D_1_11', 'D_1_12', 'D_1_13', 'D_1_22', 'D_1_23', 'D_1_33',
                            'D_2_11', 'D_2_12', 'D_2_13', 'D_2_22', 'D_2_23', 'D_2_33',
                            'f1','f2']
        
        f1,f2 = sym.symbols('f1 f2')  
        D_1_11, D_1_12, D_1_13, D_1_22, D_1_23, D_1_33 = sym.symbols('D_1_11 D_1_12 D_1_13 D_1_22 D_1_23 D_1_33')
        D_2_11, D_2_12, D_2_13, D_2_22, D_2_23, D_2_33 = sym.symbols('D_2_11 D_2_12 D_2_13 D_2_22 D_2_23 D_2_33')
        
    
        gamma = (f1* D_2_11 + f2 * D_1_11)
        
        # forms the actual structure of the parent compliance matrix, based on the children's compliance matrices.
        
        D_r_11 = (D_1_11 * D_2_11)/gamma
        D_r_12 = (f1 * D_1_12 * D_2_11 + f2 * D_2_12 * D_1_11)/gamma
        D_r_13 = (f1 * D_1_13 * D_2_11 +f2 * D_2_13 * D_1_11)/gamma
        D_r_22 = f1 * D_1_22 + f2 * D_2_22- (1/gamma) * (f1 *f2) * (D_1_12 - D_2_12)**2
        D_r_23 = f1 * D_1_23 + f2 * D_2_23 - (1/gamma) * (f1 *f2) * (D_1_13- D_2_13) * (D_1_12 - D_2_12)
        D_r_33 = f1 * D_1_33 + f2 * D_2_33 - (1/gamma) * (f1 *f2) * (D_1_13- D_2_13)**2
        

        D_r = np.array([D_r_11,D_r_12,D_r_13,D_r_22,D_r_23,D_r_33])


        # copy just makes a placeholder so that the array holds
        # the same type 
    
        derivatives = [] 
        for element in allowed_elements:
            # symbolically differentiates the elements of D_r
            # and stores in derivative_symbolic
            derivative_symbolic = D_r.copy()
            for index,function in enumerate(D_r):
                lam_func = sym.lambdify(( D_1_11, D_1_12, D_1_13, D_1_22, D_1_23, D_1_33, D_2_11, D_2_12, D_2_13, D_2_22, D_2_23, D_2_33, f1,f2), function.diff(element))
                
                derivative_symbolic[index] = lam_func
    
            derivatives.append(derivative_symbolic)
    
        
        self.derivatives_cache = derivatives
       
    def differentiate_parent_sympy(self,parent, element, diff_wrt_rot=True):
    
        
        child1= parent.left
        child2 = parent.right
        allowed_elements = ['D_1_11', 'D_1_12', 'D_1_13', 'D_1_22', 'D_1_23', 'D_1_33',
                            'D_2_11', 'D_2_12', 'D_2_13', 'D_2_22', 'D_2_23', 'D_2_33',
                            'f1','f2']
        index = allowed_elements.index(element)
        
        derivative_symbolic = np.array(self.derivatives_cache[index])
        

        derivative_numeric = np.zeros(6)

      
       
        derivative_numeric = np.array([f(*child1.rotated_compliance,*child2.rotated_compliance, child1.f, child2.f) for f in derivative_symbolic])
       
       
 
        return derivative_numeric
    

    # Calculates the error vector of the children nodes,
# given the input of a single parent node. Doesn't need child 
# as input since they are related by parent.left 
# and parent.right. See the differentiate_parent function below for more details.

    def calc_error_delta(self,parent): 
        if type(parent) is not Node:
            raise TypeError('Parent and child must be Node types')
        
        element_map = {'0':'11', '1' :'12', '2' : '13', '3': '22', '4' : '23', '5' : '33' }

        children = [parent.left, parent.right]
        
        # parent.error_alpha = differentiate_D_wrt_D_r(parent)
    
        for child_index, child in enumerate(children,start = 1):
        
            for i in range(6):
                
            
                diff = self.differentiate_parent_sympy(parent, f'D_{child_index}_{element_map[str(i)]}',diff_wrt_rot = True)
                # diff = differentiate_parent_jax(parent, f'D_{child_index}_{element_map[str(i)]}',diff_wrt_rot = True)

                child.error_delta[i] = parent.error_alpha @ diff
            # print(child.error_delta)

    
           
    def calc_error_alpha(self,parent):

        parent.error_alpha = differentiate_D_wrt_D_r(parent)

    def homogenise_system_res(self,rootnode,phase1,phase2):
        if rootnode is None:
            return
        
        # recursive part of the function
        self.homogenise_system_res(rootnode.left,phase1,phase2)
        self.homogenise_system_res(rootnode.right,phase1,phase2)


      

        # makes sure that the current node is not in the bottom layer.
        # Doesn't make sense to homogenise the bottom layer, since there
        # are no elements below it  
        
        if rootnode.left is not None and rootnode.right is not None:
            p1 = rootnode.left.rotated_compliance
            p2 = rootnode.right.rotated_compliance
            f1 = rootnode.left.f
            f2 = rootnode.right.f
            res1 = rootnode.left.res_strain
            res2 = rootnode.right.res_strain
            theta = rootnode.theta
            
            rootnode.compliance = homogenise(p1,p2,f1,f2)
            rootnode.res_strain = homogenise_res(p1,p2,f1,f2,res1,res2)
            # rotation line equation (11) 
            rootnode.rotated_compliance = convert_matrix(R(-theta) @ convert_vectorised(rootnode.compliance) @ R(theta))
            rootnode.res_strain = R(-theta) @ rootnode.res_strain
    
    def backwards_pass(self,rootnode):
        if rootnode is None:
            return
        queue = [rootnode]
        while queue:
            node = queue.pop(0)
        
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            if node.left is not None and node.right is not None:
                update_stress(node)
                # print("Updating stress of nodes: ", (node.left.layer,node.left.index),(node.right.layer,node.right.index))
            node.delta_eps = convert_vectorised(node.rotated_compliance) @ node.delta_sigma + node.res_strain
            # print("Updating the strain of node: ", node.layer,node.index)
# differentiates the volume fraction of the children
#  node with respect to a series of weights in the bottom layer, with 
# input of a parent node. Returns an array of the two derivatives 
def differentiate_fraction(parent,bottom_layer_array):
    assert type(parent) is Node
    assert type(bottom_layer_array) is list
    
    
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
    assert type(node_j) is Node
    assert type(node_k) is Node

    if node_k.index == ceil(node_j.index/(2**(node_j.layer-node_k.layer))):
        return 1.0
    else:
        return 0.0

def differentiate_D_wrt_D_r(node):
    assert type(node) is Node
    
    element_map = {'0':[1,1], '1' :[1,2], '2' : [1,3], '3': [2,2], '4' : [2,3], '5' : [3,3] }
    
    
    theta = node.theta
    
    R_pos = R(theta)
    R_neg = R(-theta)
    error = np.zeros((6,))
    
    for i in range(6):
        row = np.zeros(6)
        D_r_indices = element_map[str(i)]
        
        for j in range(6):
            D_bar_indices = element_map[str(j)]
            # indxs_neg =  np.array([D_bar_indices[0]-1,D_r_indices[0]-1])
            # indxs_pos = np.array([D_bar_indices[1]  -1,D_r_indices[1]-1])
            
            
            # print(indxs_neg,indxs_pos) 
            # print(R_neg[indxs_neg])
            # print(R_neg[indxs_neg] * R_pos[indxs_pos])
            
            
            row[j] = R_neg[D_bar_indices[0] - 1 ,D_r_indices[0] - 1] * R_pos[D_bar_indices[1]  - 1, D_r_indices[1] - 1]
        
    
        error[i] = node.error_delta @ row
    
    return error

# calculates the incremental stress of the children ndoes
# given the pareent node. 
def update_stress(node):
    assert type(node) is Node

    children = [node.left, node.right]

    for child in children:
        child.delta_sigma[0] = node.delta_sigma[0] * node.rotated_compliance[0]/child.rotated_compliance[0]
        child.delta_sigma[1] = node.delta_sigma[1]
        child.delta_sigma[2] = node.delta_sigma[2]

# computes the elasto-plastic operator of the node
# method is in Box 9.6 in the book "Computational Plasticity".
# Only valid for dgamma>0
def calc_elasto_plastic_operator(node,H=0,dgamma=0):
    assert type(node) is Node
    assert dgamma >= 0 
    
    
    if dgamma == 0:
        return convert_vectorised(node.rotated_compliance)
    elif dgamma>0:
        xi  = node.sigma.T @ P @ node.sigma
        D_mat = convert_vectorised(node.rotated_compliance)

        E = np.linalg.inv(D_mat + dgamma*P)

        n = E @ P @ node.sigma
    
        alpha = 1/(node.sigma.T @P@n + 2*xi * H/(3-2*H*dgamma))

        
        print(alpha ,np.outer(n,n),n,E)
        return np.linalg.inv(E-alpha* np.outer(n,n))

# evaluates the trial elastic state. Checks for plastic adimssability.
# only for bottom layer
def return_mapping(node):
    assert type(node) is Node

    eps_trial = node.eps + node.delta_eps
    effective_plastic_strain_trial = node.eff_plas_strain
    sigma_trial = convert_vectorised(node.rotated_compliance) @ eps_trial

    # a1 = (sigma_trial[0] + sigma_trial[1]) **2
    # a2 = (sigma_trial[1] - sigma_trial[0]) **2
    # a3 = (sigma_trial[2])**2
    # xi_trial = (1/6 * a1 + 1/2*a2 +2*a3)
    xi_trial = xi(node,0,sigma_trial)

    phi_trial = 1/2 * xi_trial - 1/3 * hardening_law(effective_plastic_strain_trial)**2

    
    # plastic condition
    if phi_trial <=0:
        # elastic state
        node.eps = eps_trial
        node.eff_plas_strain = effective_plastic_strain_trial
        node.sigma = sigma_trial

    else:
        delta_gamma = evaluate_plastic_state(node,sigma_trial)
        D = convert_vectorised(node.rotated_compliance)
        A = np.linalg.inv(D + delta_gamma *P)@ D
        node.sigma = A@sigma_trial
        node.eps = np.linalg.inv(D) @ node.sigma
        xi_temp = xi(node,delta_gamma,node.sigma)
        node.eff_plas_strain += delta_gamma * np.sqrt(2*xi_temp/3)

# Evaluates at the CURRENT effective plastiv strain.
# But uses the TRIAL stress
def evaluate_plastic_state(node,sigma_trial):
    assert type(node) is Node
    N_max_iters = 100
    delta_gamma = 0
    
    xi_temp = xi(node,delta_gamma,sigma_trial)
    phi = 1/2 * xi_temp - 1/3 * hardening_law(node.eff_plas_strain)**2
    
    for i in range(N_max_iters):
        H_temp = H(node.eff_plas_strain +delta_gamma * np.sqrt(2*xi_temp/3) )
        xi_prime_temp = xi_prime(node,delta_gamma,sigma_trial)
        
        H_prime = 2 * hardening_law(node.eff_plas_strain +delta_gamma * np.sqrt(2*xi_temp/3) ) * H_temp *np.sqrt(2/3) *(np.sqrt(xi_temp) + delta_gamma*xi_prime_temp/(2*np.sqrt(xi_temp)))
        
        phi_prime  = 1/2 * xi_prime_temp - 1/3 * H_prime

        delta_gamma+= -phi/phi_prime

        xi_temp = xi(node,delta_gamma,sigma_trial)
        phi = 1/2 * xi_temp - 1/3 * hardening_law(node.eff_plas_strain + delta_gamma * np.sqrt(2*xi_temp/3))**2
        
        if abs(phi) <10**-5:
            return delta_gamma
    raise ValueError(f'Plastic state not converged after {N_max_iters} iterations')



# returns sigma_Y as a function of effective plastic strain.
def hardening_law(eff_plas_strain):
    if eff_plas_strain < 0.008:
        return 0.1 + 5*eff_plas_strain
    elif eff_plas_strain > 0.008:
        return 0.14 + 2*eff_plas_strain


def H(eff_plas_strain):
    if eff_plas_strain < 0.008:
        return 5
    elif eff_plas_strain>0.008:
        return 2



def xi(node,delta_gamma,sigma_trial=np.zeros((3,))):
    D = convert_vectorised(node.rotated_compliance)
    A = np.linalg.inv(D + delta_gamma *P)@ D
    
    
    return sigma_trial.T @A.T@P@A@sigma_trial

def xi_prime(node,delta_gamma,sigma_trial=np.zeros((3,))):
    D = convert_vectorised(node.rotated_compliance)
    C = np.linalg.inv(D)
    A = np.linalg.inv(D + delta_gamma *P)@ D
    
    middle_mat = P.T @C.T@ A.T + P@A@C


    return -sigma_trial.T@A.T@middle_mat@P@A@sigma_trial
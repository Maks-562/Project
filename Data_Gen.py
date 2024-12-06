import random as rand
import csv
import pickle
import DMN 

# Creates a set of dummy data points into a csv file. 
# for each row, have 18 values. Each row is one 
# training example. The first 6 values represent phase1's compliance
# matrix etc.


def create_proper_dataset(material_list):
    
    with open('Dummy_Data.csv','w',newline='') as file:
        writer = csv.writer(file)
        for index,mat in enumerate(material_list,start = 1):
            for i in range(index,len(material_list)):
                D1 = [1/mat['E'], -mat['v']/mat['E'],0, 1/mat['E'],0 ,1/(2*mat['G']) ]
                D2 = [1/material_list[i]['E'], -material_list[i]['v']/material_list[i]['E'],0, 1/material_list[i]['E'],0,1/(2*material_list[i]['G']) ]
                
                D_node_1 = DMN.Node(D1)
                D_node_2 = DMN.Node(D2)
                
                D_node_1.f = 0.5
                D_node_2.f = 1-D_node_1.f


                D3 = DMN.homogenise(D_node_1,D_node_2)
                

                # rounds each value, since anything more is unphysical.
                
                
                D1 = [ '%.4f' % elem for elem in D1 ]
                D2 = ['%.4f' % elem for elem in D2 ]
                D3 = ['%.4f' % elem for elem in D3 ]
                
                writer.writerow(D1 + D2 + D3)
        
# creates a dataset with random values.
def create_random_dataset(num_train_data):
    with open('Dummy_Data.csv','w',newline='') as file:
        writer = csv.writer(file)
        for i in range(num_train_data):
            E1 = rand.uniform(1,10**2) 
            E2 = rand.uniform(1,10**2) 

            nu1 = rand.uniform(0.1,0.5)
            nu2 = rand.uniform(0.1,0.5)

            G1 = rand.uniform(1,10**2)
            G2 = rand.uniform(1,10**2)

            D1 = [0] * 6
            D2 = [0] * 6

            D1[0] = 1/E1
            D1[1] = -nu1/E1
            D1[3] = 1/E1
            D1[5] = 1/(2*G1)

            D2[0] = 1/E2
            D2[1] = -nu2/E2
            D2[3] = 1/E2
            D2[5] = 1/(2*G2)



            D3 = [sum(x)/2 for x in zip(D1, D2)]

            f1 =0.2
            writer.writerow( [f1,1-f1] + D1 + D2 + D3)
          

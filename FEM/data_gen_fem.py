from FenicsTest import *

import numpy as np


data = []
num_data_points = 3
for i in range(num_data_points):
    range_bulk = np.random.uniform(-0.5,0.5)
    range_inclusion = np.random.uniform(-0.5,0.5)
    
   
    
    E_inclusion =  10**(9+range_inclusion) 
    E_bulk = 10**(9+range_bulk)
    # E_inclusion = E_bulk
   

    nu_inclusion = np.random.uniform(0.05,0.495)
    nu_bulk = np.random.uniform(0.05,0.495)
    # nu_inclusion = nu_bulk
    
    C_inclusion = elasticity_matrix(E_inclusion,nu_inclusion)
    C_bulk = elasticity_matrix(E_bulk,nu_bulk)

    
    E,nu,D = calc_eff_elastic_prop(E_bulk,nu_bulk,E_inclusion,nu_inclusion)
    # print(C)
    # print(elasticity_matrix(E_bulk,nu_bulk))
   
    G = E/(2*(1+nu))

    # print(C)
    # print(np.linalg.inv(D))

    # D_1 = [(1-nu_inclusion)/E_inclusion,-nu_inclusion/E_inclusion,0,(1-nu_inclusion)/E_inclusion,0,(1+nu_inclusion)/(E_inclusion)]
    # D_2 = [(1-nu_bulk)/E_bulk,-nu_bulk/E_bulk,0,(1-nu_bulk)/E_bulk,0,(1+nu_bulk)/(E_bulk)]
    # D_3 = [(1-nu)/E,-nu/E,0,(1-nu)/E,0,(1+nu)/(E)]

    # D_1 = [1/E_inclusion, -nu_inclusion/E_inclusion,0, 1/E_inclusion,0 ,1/(2*G_inclusion) ]
    # D_2 =  [1/E_bulk, -nu_bulk/E_bulk,0, 1/E_bulk,0 ,1/(2*G_bulk) ]
    # D_3 = [1/E, -nu/E,0, 1/E,0 ,1/(2*G) ]   


    # D_1 = [(1-nu_inclusion**2)/E_inclusion,-nu_inclusion*(1+nu_inclusion)/E_inclusion,0,(1-nu_inclusion**2)/E_inclusion,0,2*(1+nu_inclusion)/(E_inclusion)]
    # D_2 = [(1-nu_bulk**2)/E_bulk,-nu_bulk*(1+nu_bulk)/E_bulk,0,(1-nu_bulk**2)/E_bulk,0,2*(1+nu_bulk)/(E_bulk)]
    # D_3 = [(1-nu**2)/E,-nu*(1+nu)/E,0,(1-nu**2)/E,0,2*(1+nu)/(E)]

    # C[0,2] = 0
    # C[1,2] = 0

    # C[2,0] = 0
    # C[2,1] = 0
    
    D_1 = convert_matrix(np.linalg.inv(C_inclusion))
    D_2 = convert_matrix(np.linalg.inv(C_bulk))
    D_3 = convert_matrix(D)


    # print(C_inclusion,C_bulk)
    # print(C-C_bulk,C_inclusion)
    # print(E_bulk,nu_bulk,E_inclusion,nu_inclusion)
    D_1 = [j*10**9 for j in D_1]
    D_2 = [j*10**9 for j in D_2]
    D_3 = [j*10**9 for j in D_3]

    # D_1 = [j for j in D_1]
    # D_2 = [j for j in D_2]
    # D_3 = [j for j in D_3]
    


    data.append(D_1 + D_2 + D_3)



# with open('building_block_data.csv','w',newline = '') as f:
#     writer = csv.writer(f)
#     # writer.writerow(['D1_11','D1_12','D1_16','D1_22','D1_26','D1_66','D2_11','D2_12','D2_16','D2_22','D2_26','D2_66','D3_11','D3_12','D3_16','D3_22','D3_26','D3_66'])
    
#     writer.writerows(data)  


import DMN3 as DMN
import pickle
import time
import os
import numpy as np
import sys


x = DMN.read_data("optimised_data.csv")

training_data = []

M = len(x)


for i in range(M):
    p1 = x[i][0]
    p2 = x[i][1]
    p_dns = x[i][2]

    training_data.append([p1, p2, p_dns])

start = time.time()


validation_data = training_data[200:]
training_data = training_data[:200]


N = int(sys.argv[1])

learning_rate = float(sys.argv[2])

lam = float(sys.argv[3])

epoch_num = int(sys.argv[4])

algorithm = sys.argv[5]


mini_batch_size = 20


Model = DMN.Tree(N)

root = Model.initialise()

if algorithm == "SGD":
    error_test, cost, error_train = Model.SGD(
        epoch_num,
        root,
        mini_batch_size,
        training_data,
        learning_rate,
        validation_data=validation_data,
        lam=lam,
        filename=f"model_N={N},lr={learning_rate},lam={lam}.pkl",
    )
elif algorithm == "ADAM":
    error_test, cost, error_train = Model.ADAM(
        epoch_num,
        root,
        mini_batch_size,
        training_data,
        learning_rate,
        validation_data=validation_data,
        lam=lam,
        beta=0.90,
        gamma=0.99,
        filename=f"model_N={N},lr={learning_rate},lam={lam}.pkl",
    )

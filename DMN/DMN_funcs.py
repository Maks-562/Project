import numpy as np
from numba import njit
import pickle
import csv


# phase1 and phase2 are node classes
# node.compliance is the compliance matrix of vectorised form:
# [D_11, D_12, D_13, D_22, D_23, D_33]
@njit
def homogenise(phase1, phase2, f1, f2):
    assert np.shape(phase1) == (6,)
    assert np.shape(phase2) == (6,)
    assert (type(f1) is float) or (type(f1) is np.float64)
    assert (type(f2) is float) or (type(f2) is np.float64)

    # Initialises the homogenised compliance matrix.
    D_r = np.zeros(6)

    gamma = f1 * phase2[0] + f2 * phase1[0]

    D_r[0] = (phase1[0] * phase2[0]) / gamma
    D_r[1] = (f1 * phase1[1] * phase2[0] + f2 * phase2[1] * phase1[0]) / gamma
    D_r[2] = (f1 * phase1[2] * phase2[0] + f2 * phase2[2] * phase1[0]) / gamma
    D_r[3] = (
        f1 * phase1[3]
        + f2 * phase2[3]
        - (1 / gamma) * (f1 * f2) * (phase1[1] - phase2[1]) ** 2
    )
    D_r[4] = (
        f1 * phase1[4]
        + f2 * phase2[4]
        - (1 / gamma) * (f1 * f2) * (phase1[2] - phase2[2]) * (phase1[1] - phase2[1])
    )
    D_r[5] = (
        f1 * phase1[5]
        + f2 * phase2[5]
        - (1 / gamma) * (f1 * f2) * (phase1[2] - phase2[2]) ** 2
    )

    return D_r


@njit
def convert_vectorised(vectorised_compliance_matrix):
    assert np.shape(vectorised_compliance_matrix) == (6,)

    compliance_3x3 = np.array(
        [
            [
                vectorised_compliance_matrix[0],
                vectorised_compliance_matrix[1],
                vectorised_compliance_matrix[2],
            ],
            [
                vectorised_compliance_matrix[1],
                vectorised_compliance_matrix[3],
                vectorised_compliance_matrix[4],
            ],
            [
                vectorised_compliance_matrix[2],
                vectorised_compliance_matrix[4],
                vectorised_compliance_matrix[5],
            ],
        ]
    )

    return compliance_3x3


@njit
def convert_matrix(matrix_compliance_matrix):
    assert np.shape(matrix_compliance_matrix) == (3, 3)

    compliance_vector = np.zeros((1, 6))

    compliance_vector[0, 0] = matrix_compliance_matrix[0][0]
    compliance_vector[0, 1] = matrix_compliance_matrix[0][1]
    compliance_vector[0, 2] = matrix_compliance_matrix[0][2]
    compliance_vector[0, 3] = matrix_compliance_matrix[1][1]
    compliance_vector[0, 4] = matrix_compliance_matrix[1][2]
    compliance_vector[0, 5] = matrix_compliance_matrix[2][2]
    return compliance_vector.flatten()


@njit
def relu(x):
    return x * (x > 0)


@njit
def relu_prime(x):
    return np.where(x > 0, 1, 0)


# used to read in the data.
def read_data(file_name):
    N = 0
    with open(file_name, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            N += 1

    data_array = np.zeros((N, 3, 6))

    with open(file_name, "r") as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            p1 = np.array(row[0:6], dtype=float)
            p2 = np.array(row[6:12], dtype=float)
            p_dns = np.array(row[12:18], dtype=float)

            data_array[index] = np.array([p1, p2, p_dns])
    return data_array


# Calculates the error of a single training example, at the output layer.
# D_dns is the training data compliance matrix in vectorised form and the D_h is the
# output of the network in vectorised form (rootnode.compliance).
# See equation (37).
@njit
def calc_error_0(D_dns, D_h):
    assert np.shape(D_dns) == np.shape(D_h)
    assert np.shape(D_dns) == (6,)
    # See also equation (30) in the paper. The below is the hard coded derivative of
    # equation (30) in the paper.

    norm_factor = np.linalg.norm(D_dns)

    a = 2 * (D_h - D_dns)
    a[1] *= 2
    a[2] *= 2
    a[4] *= 2

    return a / norm_factor**2


@njit
def cost(D_dns, D_h):
    norm_factor = np.linalg.norm(convert_vectorised(D_dns))
    c = 0.5 * np.linalg.norm(D_dns - D_h) ** 2
    c += 0.5 * abs(D_dns[1] - D_h[1])
    c += 0.5 * abs(D_dns[2] - D_h[2])
    c += 0.5 * abs(D_dns[4] - D_h[4])
    return c / norm_factor**2


# outputs the error of the network, of a single training example. See
# equation (78) & (79) in the paper.
@njit
def error_relative(D_dns, D_h):
    assert np.shape(D_dns) == (6,)

    e = np.linalg.norm(convert_vectorised(D_dns) - convert_vectorised(D_h)) / (
        np.linalg.norm(convert_vectorised(D_dns))
    )
    # print(D_dns-D_h)
    return e


@njit
def R(x):
    c = np.cos(x)
    s = np.sin(x)
    root_2 = np.sqrt(2)
    r = np.array(
        [
            [c**2, s**2, root_2 * c * s],
            [s**2, c**2, -root_2 * c * s],
            [-root_2 * s * c, root_2 * s * c, c**2 - s**2],
        ]
    )

    return r


@njit
def R_prime(x):
    c = np.cos(2 * x)
    s = np.sin(2 * x)
    root_2 = np.sqrt(2)

    r_prime = np.array(
        [[-s, s, root_2 * c], [s, -s, -root_2 * c], [-root_2 * c, root_2 * c, -2 * s]]
    )
    return r_prime


@njit
def differentiate_D_wrt_theta(D_r, theta):
    assert np.shape(D_r) == (6,)
    assert (type(theta) is float) or (type(theta) is np.float64)

    D_temp = convert_vectorised(D_r)
    D_theta = -R_prime(-theta) @ D_temp @ R(theta) + R(-theta) @ D_temp @ R_prime(theta)

    return convert_matrix(D_theta)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def load_model(filename):
    with open(filename, "rb") as f:
        metadata = pickle.load(f)
        training_error = pickle.load(f)
        validation_error = pickle.load(f)
        rootnode = pickle.load(f)

    return metadata, training_error, validation_error, rootnode

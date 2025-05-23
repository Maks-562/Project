import numpy as np
from matplotlib import pyplot as plt
# N is number of elements. Hence, Number of basis functions = N+1 and
# number of nodes = N+1


def hat_func(xs, offset, dx):
    hat = []
    offset = offset
    for x in xs:
        if offset - dx < x < offset:
            hat.append((x - (offset - dx)) / dx)
        elif offset <= x < offset + dx:
            hat.append(((offset + dx) - x) / dx)
        else:
            hat.append(0)
    return hat


def hat_func_derivative(xs, offset, dx):
    prime = []
    offset = offset
    for x in xs:
        if offset - dx < x < offset:
            prime.append(1 / dx)
        elif offset < x < offset + dx:
            prime.append(-1 / dx)
        else:
            prime.append(0)
    return prime


def gen_node_positions(N, bounds):
    left = bounds[0]
    right = bounds[1]
    return np.linspace(left, right, N + 1)


def basis_set(node_positions, bounds, N, n_el=100):
    dx = node_positions[1] - node_positions[0]

    data = {}

    x = np.linspace(bounds[0], bounds[1], n_el * (N))

    # treating the first basis function:
    x_element = np.linspace(node_positions[0], node_positions[2], n_el)

    phi = hat_func(x, (node_positions[1] - dx), dx)
    phi_prime = hat_func_derivative(x, (node_positions[1] - dx), dx)
    phi = np.array(phi)
    phi_prime = np.array(phi_prime)
    plt.plot(x, phi, color="k")
    # plt.plot(x,phi_prime)
    data[f"BasisFunction:{0}"] = {"phi": phi, "phi_prime": phi_prime, "x": x_element}

    # treating the last basis function:
    x_element = np.linspace(node_positions[N - 2], node_positions[N], n_el)

    phi = hat_func(x, (node_positions[N - 1] + dx), dx)
    phi_prime = hat_func_derivative(x, (node_positions[N - 1] + dx), dx)
    phi = np.array(phi)
    phi_prime = np.array(phi_prime)
    plt.plot(x, phi, color="k")
    # plt.plot(x,phi_prime)

    data[f"BasisFunction:{N}"] = {"phi": phi, "phi_prime": phi_prime, "x": x_element}

    # looping over the inner basis functions
    for i in range(0, N - 1):
        x_element = np.linspace(node_positions[i], node_positions[i + 2], n_el)

        phi = hat_func(x, (node_positions[i + 1]), dx)

        phi_prime = hat_func_derivative(x, (node_positions[i + 1]), dx)

        phi = np.array(phi)
        phi_prime = np.array(phi_prime)
        # if i == 0:
        plt.plot(x, phi)
        # plt.plot(x,phi_prime)

        data[f"BasisFunction:{i + 1}"] = {
            "phi": phi,
            "phi_prime": phi_prime,
            "x": x_element,
        }
    plt.savefig("basis_functions.jpg", dpi=500)

    plt.close()

    return data


def calc_stiffness_3(data_dict, l, r, a, b, c, f, bounds):
    K = np.zeros((N, N))
    F = np.zeros(N)

    x = np.linspace(bounds[0], bounds[-1], n_el * (N))

    # looping over the basis functions:
    for i in range(1, N + 1):
        A = data_dict[f"BasisFunction:{i}"]["phi"]
        A_prime = data_dict[f"BasisFunction:{i}"]["phi_prime"]
        for j in range(1, N + 1):
            B = data_dict[f"BasisFunction:{j}"]["phi"]

            B_prime = data_dict[f"BasisFunction:{j}"]["phi_prime"]

            kij = np.trapz(y=-a * A_prime * B_prime + b * A * B_prime + c * A * B, x=x)

            K[i - 1, j - 1] = kij

        Fi = np.trapz(y=A * f(x), x=x)

        F[i - 1] = Fi

    A = data_dict[f"BasisFunction:{1}"]["phi"]
    A_prime = data_dict[f"BasisFunction:{1}"]["phi_prime"]

    B = data_dict[f"BasisFunction:{0}"]["phi"]
    B_prime = data_dict[f"BasisFunction:{0}"]["phi_prime"]

    F[0] -= np.trapz(-a * A_prime * B_prime + b * A * B_prime + c * A * B, x=x) * l

    A = data_dict[f"BasisFunction:{N - 1}"]["phi"]
    A_prime = data_dict[f"BasisFunction:{N - 1}"]["phi_prime"]
    B = data_dict[f"BasisFunction:{N}"]["phi"]
    B_prime = data_dict[f"BasisFunction:{N}"]["phi_prime"]

    # F[-1] -= np.trapz(-a*A_prime*B_prime + b*A_prime *B + c*A*B, x = x)* r
    F[-1] += -a * r

    print(K)
    print(F)
    return np.linalg.solve(K, F)


def get_u(alphas, data_dict, l, r):
    u = np.zeros(n_el * (N))

    u += l * data_dict[f"BasisFunction:{0}"]["phi"]
    # u += r * data_dict[f'BasisFunction:{N}']['phi']

    for i in range(1, N + 1):
        A = data_dict[f"BasisFunction:{i}"]["phi"]

        u += alphas[i - 1] * A
    return u


def finite_elem_solver(N, bounds, left, right, a, b, c, f):
    nodes = gen_node_positions(N, bounds)
    data = basis_set(nodes, bounds, N)
    alphas = calc_stiffness_3(data, left, right, a, b, c, f, bounds)

    u = get_u(alphas, data, left, right)

    return u


N = 40
L = 0.05
bounds = [0.0, L]
# h = (bounds[1] - bounds[0]) / N
n_el = 100


g = 0.0005

E = 0.025e9
A = np.pi * 1e-4

left = 0.0

# rho = 5000
# sigma = 50*10**3
right = g / (A * E)
BCs = [left, right]

a = 1.0
b = 0.0
c = 0.0
B = 0.03


def f(x):
    return -B / (A * E)


def analytic(x):
    # return rho*g/(2*E)*x**2 -rho * g*L/E  * x - sigma/E*x
    return 0


u = finite_elem_solver(N, bounds, *BCs, a, b, c, f)
x_ex = np.linspace(0, L, 10)
u_ex = [-0.5 * B * x_ex_i**2 / E / A + (g + B * L) * x_ex_i / E / A for x_ex_i in x_ex]


# print(u_ex[-1])
zers = np.zeros(len(u))
x = np.linspace(bounds[0], bounds[1], len(u))
plt.plot(x, u, label="My FEM Solution")
plt.plot(x_ex, u_ex, "x", label="Analytic Solution")


# plt.plot(x,analytic(x),label = 'Analytic Solution')
# plt.plot(x,zers)
plt.legend()
plt.savefig("solution.jpg", dpi=500)

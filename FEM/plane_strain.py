import fenics as fe
import matplotlib.pyplot as plt
import numpy as np


def left(x, on_boundary):
    return fe.near(x[0], 0.0) and on_boundary



# --------------------
# Parameters
# --------------------
E = 0.025e9  # Young's modulus
A = np.pi*1e-4  # Cross-section area of bar
L = 0.05  # Length of bar
n = 20  # Number of elements
b = 0.03  # Load intensity
g = 0.0005 # External force

# --------------------
# Geometry
# --------------------
mesh = fe.IntervalMesh(n, 0.0, L)

fe.plot(mesh)
plt.savefig('1dmesh.jpg',dpi= 500)
plt.close()
# --------------------
# Function spaces
# --------------------
V = fe.FunctionSpace(mesh, "CG", 1)
u_tr = fe.TrialFunction(V)
u_test = fe.TestFunction(V)

# --------------------
# Boundary marking
# --------------------
boundary = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1,0)
for v in fe.facets(mesh):
    if fe.near(v.midpoint()[0], 0.0):
        boundary[v] = 1 # left boundary
    elif fe.near(v.midpoint()[0], L):
        boundary[v] = 2 # right boundary

dx = fe.Measure("dx", mesh)
ds = fe.Measure("ds", subdomain_data = boundary)

# --------------------
# Boundary conditions
# --------------------
bc = fe.DirichletBC(V, 0.0, left)

# --------------------
# Weak form
# --------------------
a = E*A*fe.inner(fe.grad(u_tr),fe.grad(u_test))*dx
l = b*u_test*dx + g*u_test*ds(2)

# --------------------
# Solver
# --------------------
u = fe.Function(V)
fe.solve(a == l, u, bc)

# --------------------
# Exact solution
# --------------------
x_ex = np.linspace(0, L, 10)
u_ex = [-0.5*b*x_ex_i**2/E/A + (g+b*L)*x_ex_i/E/A for x_ex_i in x_ex]

# --------------------
# Post-process
# --------------------
fe.plot(u)
plt.plot(x_ex, u_ex, "x")
plt.xlabel("x [m]")
plt.ylabel("u [m]")
plt.legend(["Fenics FEM solution","exact solution"])
plt.savefig('1d.jpg',dpi =600)
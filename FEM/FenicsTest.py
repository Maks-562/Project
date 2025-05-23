from IPython.display import display, Math
import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
from dolfinx.io import gmshio
import scipy.optimize
import csv 
import scipy

def fenics_simple_test(E_inclusion,E_bulk,nu_inclusion,nu_bulk,load_type= '1'):
   
    # domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny,cell_type=mesh.CellType.quadrilateral)
    # V = fem.functionspace(domain, element = ("CG", 1, (2,)))    

    L = 0.01
    N=300

    msh = mesh.create_rectangle(MPI.COMM_WORLD,points=[(0,0),(L,L)],n=(N,N),cell_type=mesh.CellType.quadrilateral)
    
    # def phase1(x):
    #     return x[1] <=L/2
    # def phase2(x):
    #     return x[1] >=L/2

    def phase1(x):
        return(np.logical_and.reduce((
            x[1] >= L/3 -L/(N),
            x[1] <= 2*L/3 +L/(N),
            x[0] >= L/3 -L/(N),
            x[0] <= 2*L/3 +L/(N)
        )))   
    
    def phase2(x):
        return np.logical_not(phase1(x))



    phase1_cells = mesh.locate_entities(msh, dim=2, marker=phase1)
    phase2_cells = mesh.locate_entities(msh, dim=2, marker = phase2)



    cell_tags = np.full(msh.topology.index_map(2).size_local, -1, dtype=np.int32)
    cell_tags[phase1_cells] = 1
    cell_tags[phase2_cells] = 2  

    msh.topology.create_connectivity(2, 0)

    # cell_markers = mesh.meshtags(msh, 2, np.concatenate([left_cells, right_cells]),
    #                  np.concatenate([np.full(len(left_cells), 1, dtype=np.int32),
    #                                  np.full(len(right_cells), 2, dtype=np.int32)]))






    # msh, cell_markers, facet_markers = gmshio.read_from_msh("randomInclusions2DCirlce.msh", MPI.COMM_WORLD, gdim=2)

    V = fem.functionspace(msh, element = ("CG", 1, (2,))) 





    # uniaxial load on rhs
    if load_type == '1':
        def traction_boundary1(x):
            return np.isclose(x[0], 0)

        def traction_boundary2(x):
            return np.isclose(x[0], 0.01)
        
        def traction_boundary3(x):
            return np.isclose(x[0], 213123)

        def traction_boundary4(x):
            return np.isclose(x[0], 313131)
        
        
        def clamped_boundary(x):
            return np.isclose(x[0], 0)
        
        def clamped_boundary2(x):
            return np.isclose(x[0], 0.01)
                  

        traction1 = (0,0)
        traction2 = (0,0)
    
        traction3 = (0,0)
        traction4 = (0,0)

        u_D = np.array([0.000001, 0], dtype=default_scalar_type)
        u_D2 = np.array([-0.000001, 0], dtype=default_scalar_type)


    # uniaxial load on top
    elif load_type == '2':
        def traction_boundary1(x):
            return np.isclose(x[1], 0)
        
        def traction_boundary2(x):
            return np.isclose(x[1], 0.01)
                 
        def traction_boundary3(x):
            return np.isclose(x[0], 213123)

        def traction_boundary4(x):
            return np.isclose(x[0], 313131)
    

        def clamped_boundary(x):
            return np.isclose(x[1], 0)
        
        def clamped_boundary2(x):
            return np.isclose(x[1], 0.01)


        traction1 = (0,0)
        traction2 = (0,0)
        traction3 = (0,0)
        traction4 = (0,0)
        
        u_D = np.array([0,0.000001], dtype=default_scalar_type)
        u_D2 = np.array([0,-0.000001], dtype=default_scalar_type)
    # shear load on rhs 
    elif load_type == '3':
        def traction_boundary1(x):
            return np.isclose(x[1],0.01 )    
        
        def traction_boundary2(x):
            return np.isclose(x[1],0.0)
        
          
        def traction_boundary3(x):
            return np.isclose(x[1], 0.01)

        def traction_boundary4(x):
            return np.isclose(x[0], 0)

        
        def clamped_boundary(x):
            return np.isclose(x[1],0)
        def clamped_boundary2(x):
            return np.isclose(x[1],0.01)
        

        traction1 = (0,0)
        traction2 = (0,0)
        traction3 = (0,0)
        traction4 = (0,0)
        
        u_D = np.array([0.000001, 0], dtype=default_scalar_type)
        u_D2 = np.array([-0.000001, 0], dtype=default_scalar_type)

   

    fdim = msh.topology.dim - 1
    
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, clamped_boundary)
 
       
    
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    bc2 = fem.dirichletbc(u_D2, fem.locate_dofs_topological(V, fdim, mesh.locate_entities_boundary(msh, fdim, clamped_boundary2)), V)
    
    # traction boundary

    # traction_boundary_facets = mesh.locate_entities_boundary(msh, fdim, traction_boundary)

    # facet_markers = np.full(len(traction_boundary_facets), 1, dtype=np.int32)

    # facet_tags = mesh.meshtags(msh, fdim, traction_boundary_facets,facet_markers)

        
    traction_boundary_facets1 = mesh.locate_entities_boundary(msh, fdim, traction_boundary1)
    traction_boundary_facets2 = mesh.locate_entities_boundary(msh, fdim, traction_boundary2)

    traction_boundary_facets3 = mesh.locate_entities_boundary(msh, fdim, traction_boundary3)
    traction_boundary_facets4 = mesh.locate_entities_boundary(msh, fdim, traction_boundary4)


    facet_markers1 = np.full(len(traction_boundary_facets1), 1, dtype=np.int32)
    facet_markers2 = np.full(len(traction_boundary_facets2), 2, dtype=np.int32)
  

    facet_markers3 = np.full(len(traction_boundary_facets3), 3, dtype=np.int32)
    facet_markers4 = np.full(len(traction_boundary_facets4), 4, dtype=np.int32)
    

    
    facet_tags1 = mesh.meshtags(msh, fdim, traction_boundary_facets1,facet_markers1)
    facet_tags2 = mesh.meshtags(msh, fdim, traction_boundary_facets2,facet_markers2)
 
    facet_tags3 = mesh.meshtags(msh, fdim, traction_boundary_facets3,facet_markers3)
    facet_tags4 = mesh.meshtags(msh, fdim, traction_boundary_facets4,facet_markers4)


    # combined_facets = np.concatenate([traction_boundary_facets1, traction_boundary_facets2])
    # combined_markers = np.concatenate([facet_markers1, facet_markers2])
  
    combined_facets = np.concatenate([traction_boundary_facets1, traction_boundary_facets2, traction_boundary_facets3, traction_boundary_facets4])
    combined_markers = np.concatenate([facet_markers1, facet_markers2, facet_markers3, facet_markers4])

    facet_tags = mesh.meshtags(msh, fdim, combined_facets, combined_markers)



    
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    # defining the elastic properties of the two inclusions 
    Elastic_func_space = fem.functionspace(msh,("DG",0))
    E = fem.Function(Elastic_func_space)
    nu = fem.Function(Elastic_func_space)
    
    E_inclusion = E_inclusion
    E_bulk = E_bulk

    E.x.array[phase1_cells] =np.full_like(phase1_cells,E_bulk, dtype=default_scalar_type)
    E.x.array[phase2_cells] =np.full_like(phase2_cells,E_inclusion, dtype=default_scalar_type)

    nu.x.array[phase1_cells] =np.full_like(phase1_cells,nu_bulk, dtype=default_scalar_type)
    nu.x.array[phase2_cells] =np.full_like(phase2_cells,nu_inclusion, dtype=default_scalar_type)


        
    for index,val in enumerate(E.x.array):
        if val == 0:
            E.x.array[index] = E_bulk
            nu.x.array[index] = nu_bulk
    if 0 in E.x.array:
        assert False, "E is zero"


    # E.x.array[:] = np.where(cell_markers.values == 2, E_inclusion, E_bulk)

    # nu.x.array[:] = np.where(cell_markers.values == 2, nu_inclusion, nu_bulk)
    
        

    # plane strain
    lam = E*nu/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))

    # plane stress
    # lambda_ = 2*lam*mu/(lam + 2*mu)
    lambda_ = E*nu/(1-nu**2)



    def epsilon(u):
        return 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        

    def sigma(u):
        return lam * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

        
    T1 = fem.Constant(msh, default_scalar_type(traction1))
    T2 = fem.Constant(msh, default_scalar_type(traction2))
    T3 = fem.Constant(msh, default_scalar_type(traction3))
    T4 = fem.Constant(msh, default_scalar_type(traction4))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(msh, default_scalar_type((0, 0)))

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T1, v) *ds(1)  + ufl.dot(T2, v) *ds(2) + ufl.dot(T3, v) *ds(3) + ufl.dot(T4, v) *ds(4)
   
   
    problem = LinearProblem(a, L, bcs=[bc,bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    
    uh = problem.solve()

    indices = [[0,0],[1,1],[0,1]]
    V_interp = fem.functionspace(msh, ("DG", 1))

    stresses = fem.Function(V_interp)
    strains = fem.Function(V_interp)

    S_arr  = []
    Eps_arr = []

    for s_index in indices:
    
        s_indices = s_index
    
        s = sigma(uh)[s_indices[0],s_indices[1]]
        eps = epsilon(uh)[s_indices[0],s_indices[1]]




        stress_expr = fem.Expression(s, V_interp.element.interpolation_points())
        strains_expr = fem.Expression(eps, V_interp.element.interpolation_points())


        stresses.interpolate(stress_expr)
        strains.interpolate(strains_expr)

        integral_stress = fem.assemble_scalar(fem.form(s * ufl.dx))/(0.01)**2
        intergal_strain = fem.assemble_scalar(fem.form(eps * ufl.dx))/(0.01)**2

        S_arr.append(integral_stress)
        if s_index == [0,1]:
            Eps_arr.append(2*intergal_strain)
        else:
            Eps_arr.append(intergal_strain)

    S = np.vstack((S_arr[0],S_arr[1],S_arr[2])).reshape(3,1)
    Eps = np.vstack((Eps_arr[0],Eps_arr[1],Eps_arr[2])).reshape(3,1)
    print('load type:',load_type)
    print(S,Eps)
    print('.................')
    return S,Eps





def homogenise(phase1,phase2,f1,f2):
    # follows the definiton of the homogenisation procedure as in the paper.
    # See equation (8)
    
    # Initialises the homogenised compliance matrix.
    D_r = np.zeros(6)
    
    gamma =f1 * phase2[0] +f2 * phase1[0]
   
    D_r[0] = (phase1[0] * phase2[0])/gamma
    D_r[1] = (f1 * phase1[1] * phase2[0] +f2 * phase2[1] * phase1[0])/gamma
    D_r[2] = (f1 * phase1[2] * phase2[0] +f2 * phase2[2] * phase1[0])/gamma
    D_r[3] =f1 * phase1[3] +f2 * phase2[3] - (1/gamma) * (f1 *f2) * (phase1[1] - phase2[1])**2
    D_r[4] =f1 * phase1[4] +f2 * phase2[4] - (1/gamma) * (f1 *f2) * (phase1[2]- phase2[2]) * (phase1[1] - phase2[1])
    D_r[5] =f1 * phase1[5] +f2 * phase2[5] - (1/gamma) * (f1 *f2) * (phase1[2]- phase2[2])**2
    



    return D_r



def elasticity_matrix(E,nu):
    # plane strain elasticity matrix   
    Ce = np.array([[1-nu,nu,0],
                    [nu,1-nu,0],
                    [0,0,(1-2*nu)/2]])
    Ce*= E/((1-2*nu)*(1+nu))
    
    return Ce

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

def convert_vectorised(vectorised_compliance_matrix):
    assert(len(vectorised_compliance_matrix) == 6)
   
    
    compliance_3x3 = np.array([[vectorised_compliance_matrix[0], vectorised_compliance_matrix[1], vectorised_compliance_matrix[2]],
                               [vectorised_compliance_matrix[1],vectorised_compliance_matrix[3],vectorised_compliance_matrix[4]],
                               [vectorised_compliance_matrix[2],vectorised_compliance_matrix[4],vectorised_compliance_matrix[5]]
                                ])

    return compliance_3x3

def calculate_nu_E(C):
    c11 = C[0,0]
    c12 = C[0,1]
    c33 = C[2,2]

    nu_calc = 1/2 - c33/(c11 + c12)
    E =2* (1+nu_calc)*c33
    
    return E,nu_calc 


def residual(c,S_tot,eps_tot):
    c[2] = 0
    c[4] = 0
    C_mat = convert_vectorised(c)
    res = 0
    for i in range(3):
        res += np.linalg.norm(S_tot[:,i] - C_mat @ eps_tot[:,i])**2
    return res

def calc_eff_elastic_prop(E_bulk,nu_bulk,E_inclusion,nu_inclusion):
   

    S_tot = np.zeros((3,3))
    Eps_tot = np.zeros((3,3))

    for i in range(1,4):
        
        S, Eps = fenics_simple_test(E_inclusion,E_bulk,nu_inclusion,nu_bulk,load_type=str(i))
       
        S_tot[:,i-1] = S.flatten()
        Eps_tot[:,i-1] = Eps.flatten()
    
    x0 = convert_matrix(elasticity_matrix((E_bulk + E_inclusion)/2,(nu_bulk + nu_inclusion)/2))


    opt = scipy.optimize.least_squares(residual,x0 =x0,args = (S_tot,Eps_tot))
    # print('err:',opt.fun)
    C1 = convert_vectorised(opt.x)
    C = np.linalg.solve(Eps_tot,S_tot)
    
    D = np.linalg.inv(C)
    D1 = np.linalg.inv(C1)
    D3 = convert_vectorised(homogenise(convert_matrix(np.linalg.inv(elasticity_matrix(E_inclusion,nu_inclusion))),convert_matrix(np.linalg.inv(elasticity_matrix(E_bulk,nu_bulk))),0.5,0.5))
    
    
    print('Direct iinv approach:')
    print(D)
    print('Optimised approach:')
    print(D1)
    print('Analytical approach:')
    print(D3)
  
    E,nu = calculate_nu_E(C)

    print(f'The elastic moduli of the bulk and inclusion are {E_bulk:.2e} Pa and {nu_bulk:.2f} and {E_inclusion:.2e} Pa and {nu_inclusion:.2f} respectively')
    print(f'The homogenised elastic moduli are {E:.2e} Pa and {nu:.2f}')
    return E,nu,D1


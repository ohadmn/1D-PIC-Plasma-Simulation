import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import coo_matrix, spdiags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.fftpack import fftshift
from numpy.fft import fft
from multiprocessing import Pool, cpu_count, Manager

np.set_printoptions(threshold=np.inf)

def find_f_max(f, x_min, x_max, v_0, v_th):
    """ Finds the maximum extremum of a given function."""
    x_vals = np.linspace(x_min, x_max, 400)
    f_vals = f(x_vals, v_0, v_th)
    return np.max(f_vals)

def maxwell_boltzman_dist_1D(v, v_0, v_th):
    """ Maxwell-Boltzmann speed distribution for speeds. """
    n_0 = 1 # Particle reference density
    return ((n_0 / (2 * (np.sqrt(2 * np.pi)) * v_th)) * np.exp(-((v - v_0) ** 2)/(2 * v_th **2))) 

def acceptence_rejection(f, v_min, v_max, v_0, v_th, N):
    random_speeds = []
    rejected = []
    f_max = find_f_max(f, v_min, v_max, v_0, v_th)
    while len(random_speeds) < N:
        r_1 = np.random.uniform(0,1)
        v = v_min + r_1 * (v_max - v_min)
        r_2 = np.random.uniform(0,1)
        u = r_2 * f_max
        if u < f(v, v_0, v_th):
            random_speeds.append(v)
        else:
            rejected.append(v)
    return np.array(random_speeds)

def initial_loading(N, v_0, v_th, v_min, v_max, amplitude, mode, L):
    position = np.linspace(0, L, N)
    position = np.transpose(position)  # Ensure it's a column vector

    # Maxwellian distribution of velocities of the Beam
    velocity = acceptence_rejection(maxwell_boltzman_dist_1D, v_min, v_max, v_0, v_th, N)

    # Perturbation
    if amplitude != 0:
        position = position + amplitude * np.cos(2 * np.pi * mode * position / L)

    return position, velocity

def aux_vector(N):
    if N != 0:
        p = np.arange(N)
        p = np.concatenate((p, p))
    else:
        p = np.array([0])
    return p

def periodic_boundaries(Project, lower, upper):
    Project = np.mod(Project - lower, upper - lower) + lower
    return Project

def interpolation(dx, Ng, positions, N, aux_vector):
    # Project Particles to grid
    left_node_idx = np.floor(positions / dx).astype(int)
    right_node_idx = left_node_idx + 1

    # Concatenate the grid points which particle is between
    Project = np.vstack((left_node_idx, right_node_idx))

    # Boundary conditions on the projection
    Project = periodic_boundaries(Project, 0, Ng - 1)

    # Compute what fraction of the particle size that lies on the two nearest cells
    Fim = 1 - np.abs((positions / dx) - left_node_idx)
    Fip = 1 - Fim
    Fraction = np.vstack((Fim, Fip))
    
    # Apply threshold operation like in MATLAB
    Fraction[Fraction > 0.5] = 1
    Fraction[Fraction < 0.5] = 0

    # Interpolation matrix
    rows = aux_vector  # Use aux_vector directly to match MATLAB indexing
    cols = Project.flatten()
    data = Fraction.flatten()
    
    # Create sparse interpolation matrix
    interp = coo_matrix((data, (rows, cols)), shape=(N, Ng))

    return interp

def charge(re_1, re_2, N_ions, N1, N2, L, wp_e):
    Q_1 = (wp_e**2 * L) / (N1 * re_1)
    
    if re_2 > 0:
        Q_2 = -Q_1 * (N1 / N2)
    elif re_2 < 0:
        Q_2 = Q_1 * (N1 / N2)
    else:
        Q_2 = 0
    
    rho_ions = (-Q_1 / L) * N_ions  # Background density
    
    return Q_1, Q_2, rho_ions

def charge_density(charge, interp, dx):
    rho = (charge / dx) * np.sum(interp.toarray(), axis=0)
    return rho

def poisson_equation_preparation(Ng, L):
    # Poisson's equation preparation
    un = np.ones(Ng - 1)  # Ng-1 * 1
    Ax = spdiags([un, -2 * un, un], [-1, 0, 1], Ng - 1, Ng - 1)  # Matrix for Poisson's equation Ng-1 * Ng-1
    
    return Ax

def field(charge_density, Ng, dx, Ax):
    # Convert Ax to CSC format
    Ax = csc_matrix(Ax)
    
    # Ensure charge_density is the correct shape
    charge_density = charge_density[:Ng-1]
    
    # Solve for Phi
    Phi = spsolve(Ax, -charge_density * dx**2)
    Phi = np.append(Phi, 0)
    
    # Compute Eg
    Eg = (np.append([Phi[Ng-1]], [Phi[0:(Ng-1)]]) - np.append([Phi[1:(Ng)]], [Phi[0]])) / (2*dx)

    return Phi, Eg

def theoretical_dispersion_relation(k, wp_e, lambda_d):
    return wp_e * np.sqrt(1 + 3/2 * (k**2) * (lambda_d**2))

def simulate_step(args):
    it, shared_data, lock = args
    with lock:
        xp1 = np.copy(shared_data['xp1'])
        vp1 = np.copy(shared_data['vp1'])
        xp2 = np.copy(shared_data['xp2'])
        vp2 = np.copy(shared_data['vp2'])
        Q1 = shared_data['Q1']
        Q2 = shared_data['Q2']
        re_1 = shared_data['re_1']
        re_2 = shared_data['re_2']
        dt = shared_data['dt']
        dx = shared_data['dx']
        Ng = shared_data['Ng']
        Ax = shared_data['Ax']
        rho_back = shared_data['rho_back']

    mat1 = interpolation(dx, Ng, xp1, len(xp1), aux_vector(len(xp1)))
    mat2 = interpolation(dx, Ng, xp2, len(xp2), aux_vector(len(xp2)))

    rho1 = charge_density(Q1, mat1, dx)
    rho2 = charge_density(Q2, mat2, dx)
    rhot = rho1 + rho2 + rho_back

    Phi, Eg = field(rhot, Ng, dx, Ax)

    # Electric field experienced by the particles (interpolated from the grid)
    E_field_particles1 = mat1 @ Eg
    E_field_particles2 = mat2 @ Eg

    vp1 += (re_1 * E_field_particles1 * dt)
    vp2 += (re_2 * E_field_particles2 * dt)

    xp1 += vp1 * dt
    xp2 += vp2 * dt

    xp1 = periodic_boundaries(xp1, 0, Ng*dx)
    xp2 = periodic_boundaries(xp2, 0, Ng*dx)

    E_kin = 0.5 * np.abs(Q1 / re_1) * np.sum(vp1**2) + 0.5 * np.abs(Q2 / re_2) * np.sum(vp2**2)
    E_pot = 0.5 * np.sum(Eg**2) * dx
    mom = (Q1 / re_1) * np.sum(vp1) + (Q2 / re_2) * np.sum(vp2)

    with lock:
        shared_data['xp1'] = xp1
        shared_data['vp1'] = vp1
        shared_data['xp2'] = xp2
        shared_data['vp2'] = vp2

    return (it, E_kin, E_pot, mom, Eg)

def main():
    ### Simulation Constants ###
    L = 1024  # Domain size
    wp_e = 1  # Plasma frequency
    Ng = 8192  # No. grid cells
    dx = L / Ng
    Nt = 8000  # Number of time steps
    dt = 0.05
    v_th = 1
    v_min = -10
    v_max = 10
    N_ions = 100000

    # 1st beam
    N1 = 50000
    v0_1 = 0
    X1 = 0
    Mode1 = 0
    re_1 = -1

    # 2nd beam
    N2 = 50000
    v0_2 = 0
    X2 = 0
    Mode2 = 0 
    re_2 = -1

    print("Initializing phase-space distribution.")
    # Phase-space distribution initialization
    xp1, vp1 = initial_loading(N1, v0_1, v_th, v_min, v_max, X1, Mode1, L)
    xp2, vp2 = initial_loading(N2, v0_2, v_th, v_min, v_max, X2, Mode2, L)

    # Particles' charge and background charge computation
    Q1, Q2, rho_back = charge(re_1, re_2, N_ions, N1, N2, L, wp_e)
    rho_back = np.full(Ng, rho_back)  # Create an array of background charge density

    # Preparation of Poisson's equation
    Ax = poisson_equation_preparation(Ng, L)
    
    # Initialize arrays for FFT results
    E_field_time = np.zeros((Nt, Ng))
    E_kin = np.zeros(Nt)
    E_pot = np.zeros(Nt)
    mom = np.zeros(Nt)

    manager = Manager()
    shared_data = manager.dict()
    shared_data['xp1'] = xp1
    shared_data['vp1'] = vp1
    shared_data['xp2'] = xp2
    shared_data['vp2'] = vp2
    shared_data['Q1'] = Q1
    shared_data['Q2'] = Q2
    shared_data['re_1'] = re_1
    shared_data['re_2'] = re_2
    shared_data['dt'] = dt
    shared_data['dx'] = dx
    shared_data['Ng'] = Ng
    shared_data['Ax'] = Ax
    shared_data['rho_back'] = rho_back

    lock = manager.Lock()

    print("Starting main simulation loop.")
    # Prepare arguments for multiprocessing
    args = [(it, shared_data, lock) for it in range(Nt)]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(simulate_step, args), total=Nt))

    for res in results:
        it, E_kin[it], E_pot[it], mom[it], Eg = res
        E_field_time[it, :] = Eg

    print("Applying FFT to obtain the dispersion plot.")
    # Apply FFT to obtain the dispersion plot
    E_field_freq = fftshift(fft(E_field_time, axis=0), axes=0)
    E_field_freq = np.abs(E_field_freq)**2

    plt.imshow(E_field_freq[:Nt//2, :Ng//2].T, extent=[0, 0.3, 0, 1.5], aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel('$k$($2\pi/\lambda_{D}$) (unitless)')
    plt.ylabel('$\omega$($\omega_{p,e}$) (unitless)')
    plt.title('Ratio of the theoretical to the simulated dispersion relation for Langmuir waves')
    
    # Calculate theoretical dispersion relation
    k_vals = np.linspace(0, 0.3, 100)
    omega_vals = theoretical_dispersion_relation(k_vals, wp_e, v_th / wp_e)
    plt.plot(k_vals, omega_vals, 'w-', lw=1.5)

    plt.show()

if __name__ == '__main__':
    main()

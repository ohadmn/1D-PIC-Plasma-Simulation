import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import coo_matrix, spdiags, csc_matrix
from scipy.sparse.linalg import spsolve


np.set_printoptions(threshold=np.inf)

def find_f_max(f, x_min, x_max, v_0, v_th):
    """
    Finds the maximum value of a given function f within the range [x_min, x_max].

    Parameters:
    f (function): The function to be evaluated.
    x_min (float): The minimum x-value.
    x_max (float): The maximum x-value.
    v_0 (float): The mean velocity of the distribution.
    v_th (float): The thermal velocity of the distribution.

    Returns:
    float: The maximum value of the function f within the specified range.
    """
    x_vals = np.linspace(x_min, x_max, 400)
    f_vals = f(x_vals, v_0, v_th)
    return np.max(f_vals)

def maxwell_boltzman_dist_1D(v, v_0, v_th):
    """
    Calculates the Maxwell-Boltzmann speed distribution for a given velocity.

    Parameters:
    v (float): The velocity at which to evaluate the distribution.
    v_0 (float): The mean velocity of the distribution.
    v_th (float): The thermal velocity of the distribution.

    Returns:
    float: The value of the Maxwell-Boltzmann distribution at velocity v.
    """
    n_0 = 1 # Particle refernce density
    return ((n_0 / (2 * (np.sqrt(2 * np.pi)) * v_th)) * np.exp(-((v - v_0) ** 2)/(2 * v_th **2))) 

def acceptence_rejection(f, v_min, v_max, v_0, v_th, N):
    """
    Generates N random samples from a distribution f using the acceptance-rejection method.

    Parameters:
    f (function): The target distribution function.
    v_min (float): The minimum velocity.
    v_max (float): The maximum velocity.
    v_0 (float): The mean velocity of the distribution.
    v_th (float): The thermal velocity of the distribution.
    N (int): The number of samples to generate.

    Returns:
    np.array: Array of N random samples from the distribution f.
    """
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
    """
    Initializes the positions and velocities of particles in a beam.

    Parameters:
    N (int): The number of particles.
    v_0 (float): The mean velocity of the beam.
    v_th (float): The thermal velocity of the beam.
    v_min (float): The minimum velocity.
    v_max (float): The maximum velocity.
    amplitude (float): The amplitude of the position perturbation.
    mode (int): The mode of the position perturbation.
    L (float): The length of the simulation domain.

    Returns:
    tuple: A tuple containing arrays of initial positions and velocities of particles.
    """
    position = np.linspace(0, L, N)
    position = np.transpose(position)  # Ensure it's a column vector

    # Maxwellian distribution of velocities of the Beam
    velocity = acceptence_rejection(maxwell_boltzman_dist_1D, v_min, v_max, v_0, v_th, N)

    # Perturbation
    if amplitude != 0:
        position = position + amplitude * np.cos(2 * np.pi * mode * position / L)

    return position, velocity

def aux_vector(N):
    """
    Generates an auxiliary vector for interpolation.

    Parameters:
    N (int): The number of particles.

    Returns:
    np.array: An auxiliary vector used in interpolation.
    """
    if N != 0:
        p = np.arange(N)
        p = np.concatenate((p, p))
    else:
        p = np.array([0])
    return p

def periodic_boundaries(Project, lower, upper):
    """
    Applies periodic boundary conditions to a set of positions.

    Parameters:
    Project (np.array): The positions to be adjusted.
    lower (float): The lower bound of the domain.
    upper (float): The upper bound of the domain.

    Returns:
    np.array: The adjusted positions with periodic boundaries applied.
    """
    Project = np.mod(Project - lower, upper - lower) + lower
    return Project

def interpolation(dx, Ng, positions, N, aux_vector):
    """
    Computes the interpolation matrix for particle positions.

    Parameters:
    dx (float): The grid spacing.
    Ng (int): The number of grid points.
    positions (np.array): The positions of the particles.
    N (int): The number of particles.
    aux_vector (np.array): The auxiliary vector for interpolation.

    Returns:
    coo_matrix: The sparse interpolation matrix.
    """
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
    """
    Calculates the charge of particles and the background charge density.

    Parameters:
    re_1 (float): Charge to mass ratio of the first beam.
    re_2 (float): Charge to mass ratio of the second beam.
    N_ions (int): Number of ions.
    N1 (int): Number of particles in the first beam.
    N2 (int): Number of particles in the second beam.
    L (float): Length of the simulation domain.
    wp_e (float): Plasma frequency.

    Returns:
    tuple: A tuple containing the charges of the first and second beams and the background charge density.
    """
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
    """
    Computes the charge density on the grid from the particle charge.

    Parameters:
    charge (float): The charge of the particles.
    interp (coo_matrix): The interpolation matrix.
    dx (float): The grid spacing.

    Returns:
    np.array: The charge density on the grid.
    """
    rho = (charge / dx) * np.sum(interp.toarray(), axis=0)
    return rho

def poisson_equation_preparation(Ng, L):
    """
    Prepares the matrix for solving Poisson's equation.

    Parameters:
    Ng (int): The number of grid points.
    L (float): The length of the simulation domain.

    Returns:
    spdiags: The matrix for solving Poisson's equation.
    """
    # Poisson's equation preparation
    un = np.ones(Ng - 1)  # Ng-1 * 1
    Ax = spdiags([un, -2 * un, un], [-1, 0, 1], Ng - 1, Ng - 1)  # Matrix for Poisson's equation Ng-1 * Ng-1
    
    return Ax

def field(charge_density, Ng, dx, Ax):
    """
    Solves Poisson's equation for the electric potential and computes the electric field.

    Parameters:
    charge_density (np.array): The charge density on the grid.
    Ng (int): The number of grid points.
    dx (float): The grid spacing.
    Ax (spdiags): The matrix for solving Poisson's equation.

    Returns:
    tuple: A tuple containing the electric potential (Phi) and the electric field (Eg).
    """
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

def save_histogram_as_pdf(vp1, vp2, v0_1, v0_2, v_th, v_min, v_max, filename):
    """
    Saves a histogram of the particle velocities as a PDF.

    Parameters:
    vp1 (np.array): Velocities of the first beam.
    vp2 (np.array): Velocities of the second beam.
    v0_1 (float): Mean velocity of the first beam.
    v0_2 (float): Mean velocity of the second beam.
    v_th (float): Thermal velocity.
    v_min (float): Minimum velocity.
    v_max (float): Maximum velocity.
    filename (str): The filename for the saved PDF.
    """
    fig, ax = plt.subplots()
    ax.hist(vp1, bins=30, ec="black", fc="blue", density=True, alpha=0.5, lw=1)
    x_1 = np.linspace(min(vp1), max(vp1), 10000)
    ax.plot(x_1, 2 * maxwell_boltzman_dist_1D(x_1, v0_1, v_th), color='black', label='First beam', linewidth=1)
    ax.hist(vp2, bins=30, ec="black", fc="red", density=True, alpha=0.5, lw=1)
    x_2 = np.linspace(min(vp2), max(vp2), 10000)
    ax.plot(x_2, 2 * maxwell_boltzman_dist_1D(x_2, v0_2, v_th), color='black', label='Second beam', linewidth=1)
    ax.set_title('Superparticles Distribution for Speeds in 1D')
    ax.set_xlabel('$v$/$v_{th}$ (unitless)')
    ax.set_ylabel('$f(v//v_{th})$ (unitless)')
    ax.axis([v_min, v_max, 0, 2*0.19945547892115006])
    ax.legend()
    fig.savefig(filename)
    plt.close(fig)

def save_scatter_as_pdf(xp1, vp1, xp2, vp2, L, filename):
    """
    Saves a scatter plot of the particle phase space as a PDF.

    Parameters:
    xp1 (np.array): Positions of the first beam.
    vp1 (np.array): Velocities of the first beam.
    xp2 (np.array): Positions of the second beam.
    vp2 (np.array): Velocities of the second beam.
    L (float): The length of the simulation domain.
    filename (str): The filename for the saved PDF.
    """
    fig, ax = plt.subplots()
    ax.scatter(xp1, vp1, s=0.1, color='blue', alpha=0.5)
    ax.scatter(xp2, vp2, s=0.1, color='red', alpha=0.5)
    ax.axis([0, L, -15, 15])
    ax.set_xlabel('$x$/$\lambda_{D}$ (unitless)')
    ax.set_ylabel('$v$/$v_{th}$ (unitless)')
    ax.set_title('Phase Space')
    fig.savefig(filename)
    plt.close(fig)

def save_subplot_as_pdf(ax, filename, legend=False):
    """
    Saves a specific subplot as a PDF.

    Parameters:
    ax (matplotlib.axes._subplots.AxesSubplot): The subplot to be saved.
    filename (str): The filename for the saved PDF.
    legend (bool): Whether to include the legend in the saved plot.
    """
    fig = plt.figure()
    new_ax = fig.add_subplot(111)
    
    # Copy properties from the original ax to the new_ax
    new_ax.set_xlim(ax.get_xlim())
    new_ax.set_ylim(ax.get_ylim())
    new_ax.set_title(ax.get_title())
    new_ax.set_xlabel(ax.get_xlabel())
    new_ax.set_ylabel(ax.get_ylabel())
    
    # Copy lines and other plot elements
    for line in ax.get_lines():
        new_ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(), linewidth=line.get_linewidth())
        
    for text in ax.texts:
        new_ax.text(text.get_position()[0], text.get_position()[1], text.get_text(), fontsize=text.get_fontsize(), color=text.get_color())
    if legend:
        new_ax.legend()
    fig.savefig(filename)
    plt.close(fig)

def main():
    ### Simulation Constants ###
    L = 64  # Domain size
    wp_e = 1  # Plasma frequency
    Ng = 256  # Number of grid cells
    dx = L / Ng  # Grid spacing
    Nt = 16000  # Number of time steps
    dt = 0.1  # Time step size
    v_th = 1  # Thermal velocity
    v_min = -10  # Minimum velocity
    v_max = 10  # Maximum velocity
    N_ions = 20000  # Number of ions

    # 1st beam parameters
    N1 = 10000  # Number of particles in the first beam
    v0_1 = 5  # Mean velocity of the first beam
    X1 = 0  # Initial position offset for the first beam
    Mode1 = 0  # Mode for the initial perturbation of the first beam
    re_1 = -1  # Charge to mass ratio of the first beam

    # 2nd beam parameters
    N2 = 10000  # Number of particles in the second beam
    v0_2 = -5  # Mean velocity of the second beam
    X2 = 0  # Initial position offset for the second beam
    Mode2 = 0  # Mode for the initial perturbation of the second beam
    re_2 = -1  # Charge to mass ratio of the second beam

    time = np.arange(0, Nt * dt, dt)  # Array of time steps

    # Phase-space distribution initialization
    xp1, vp1 = initial_loading(N1, v0_1, v_th, v_min, v_max, X1, Mode1, L)
    xp2, vp2 = initial_loading(N2, v0_2, v_th, v_min, v_max, X2, Mode2, L)

    # Particles' charge and background charge computation
    Q1, Q2, rho_ions = charge(re_1, re_2, N_ions, N1, N2, L, wp_e)

    # Initialization of the auxiliary vectors
    p1 = aux_vector(N1)
    p2 = aux_vector(N2)

    # Preparation of Poisson's equation
    Ax = poisson_equation_preparation(Ng, L)
    
    # Others
    mat1 = 0
    mat2 = 0
    Eg = 0
    Phi = 0
    E_kin = np.zeros(Nt)
    E_pot = np.zeros(Nt)
    E_tot = np.zeros(Nt)
    relative_energy_error = np.zeros(Nt)

    # Intitialzie Simulation Loop Figure
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Simulation Plots')
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Main simulation loop
    for it in tqdm(range(Nt)):
        mat1 = interpolation(dx, Ng, xp1, N1, p1)
        mat2 = interpolation(dx, Ng, xp2, N2, p2)

        rho1 = charge_density(Q1, mat1, dx)
        rho2 = charge_density(Q2, mat2, dx)
        rhot = rho1 + rho2 + rho_ions

        Phi, Eg = field(rhot, Ng, dx, Ax)
        
        vp1 += (re_1 * mat1 * Eg * dt)
        vp2 += (re_2 * mat2 * Eg * dt)

        xp1 += vp1 * dt
        xp2 += vp2 * dt

        xp1 = periodic_boundaries(xp1, 0, L)
        xp2 = periodic_boundaries(xp2, 0, L)

        E_kin[it] = 0.5 * np.abs(Q1 / re_1) * np.sum(vp1**2) + 0.5 * np.abs(Q2 / re_2) * np.sum(vp2**2)
        E_pot[it] = 0.5 * np.sum(Eg**2) * dx
        E_tot[it] = E_kin[it] + E_pot[it]

        if it == 0:
            initial_total_energy = E_tot[it]
        
        relative_energy_error[it] = np.abs(E_tot[it] - initial_total_energy) / initial_total_energy * 100

        # Update the figure's headline with the current iteration number
        fig.suptitle(f'Simulation Plots - Iteration {it}', fontsize=16)

        # Plotting

        axs[0, 0].cla()
        axs[0, 0].hist(vp1, bins=30, ec="black", fc="blue", density=True, alpha=0.5, lw=1)
        x_1 = np.linspace(min(vp1), max(vp1), 10000)
        axs[0, 0].plot(x_1, 2 * maxwell_boltzman_dist_1D(x_1, v0_1, v_th), color='black', label='First beam', linewidth=1)
        axs[0, 0].hist(vp2, bins=30, ec="black", fc="red", density=True, alpha=0.5, lw=1)
        x_2 = np.linspace(min(vp2), max(vp2), 10000)
        axs[0, 0].plot(x_2, 2 * maxwell_boltzman_dist_1D(x_2, v0_2, v_th), color='black', label='Second beam', linewidth=1)
        axs[0, 0].set_title('Superparticles Distribution for Speeds in 1D')
        axs[0, 0].set_xlabel('$v$/$v_{th}$ (unitless)')
        axs[0, 0].set_ylabel('$f(v//v_{th})$ (unitless)')
        axs[0, 0].axis([v_min, v_max, 0, 2*0.19945547892115006])

        axs[0, 1].cla()
        axs[0, 1].scatter(xp1, vp1, s=0.1, color='blue', alpha=0.5)
        axs[0, 1].scatter(xp2 ,vp2, s=0.1, color='red', alpha=0.5)
        axs[0, 1].axis([0, L, -15, 15])
        axs[0, 1].set_xlabel('$x$/$\lambda_{D}$ (unitless)')
        axs[0, 1].set_ylabel('$v$/$v_{th}$ (unitless)')
        axs[0, 1].set_title('Phase Space')

        axs[1, 1].cla()
        axs[1, 1].plot(Phi, color='orangered', label='Potential Energy', linewidth=1)
        axs[1, 1].set_xlabel('$x$/$\lambda_{D}$ (unitless)')
        axs[1, 1].set_ylabel('$\\phi$/($T_{e}/e$) (unitless)')
        axs[1, 1].set_title('Electric Potential')

        axs[0, 2].cla()
        axs[0, 2].plot(time, E_kin, color='yellowgreen', label='Kinetic Energy', linewidth=1)
        axs[0, 2].plot(time, E_pot, color='orange', label='Potential Energy', linewidth=1)
        axs[0, 2].set_xlim([0, it * dt])
        axs[0, 2].legend()
        axs[0, 2].set_xlabel('$\omega_{p,e}t$ (unitless)')
        axs[0, 2].set_ylabel('$\\epsilon$/($n_{0}T_{e}/\epsilon_{0}$) (unitless)')
        axs[0, 2].set_title('The time dependence of the energies')
        
        axs[1, 0].cla()
        axs[1, 0].plot(Eg, color='royalblue', label='Electric Field Grid', linewidth=1)
        axs[1, 0].set_xlabel('$x$/$\lambda_{D}$ (unitless)')
        axs[1, 0].set_ylabel('$E$/($T_{e}/e\lambda_{D}$) (unitless)')
        axs[1, 0].set_title('Electric Field Grid')
        
        axs[1, 2].cla()
        axs[1, 2].plot(rhot, color='tomato', label='Charge Density', linewidth=1)
        axs[1, 2].axis([0, L, -15, 15])
        axs[1, 2].set_xlabel('$x$/$\lambda_{D}$ (unitless)')
        axs[1, 2].set_ylabel('$\\rho$/($e\lambda_{D}$) (unitless)')
        axs[1, 2].set_title('Charge Density')
        
        if it % 1000 == 0 or it == Nt-1:
            print(f"Iteration {it}, Relative Error: {relative_energy_error[it]}%")
        
        if it == 0:
            save_histogram_as_pdf(vp1, vp2, v0_1, v0_2, v_th, v_min, v_max, f'superparticles_distribution_{it}.pdf')
            save_scatter_as_pdf(xp1, vp1, xp2, vp2, L, f'phase_space_{it}.pdf')


        if it == 105:
            save_histogram_as_pdf(vp1, vp2, v0_1, v0_2, v_th, v_min, v_max, f'superparticles_distribution_{it}.pdf')
            save_scatter_as_pdf(xp1, vp1, xp2, vp2, L, f'phase_space_{it}.pdf')
            save_subplot_as_pdf(axs[1, 1], f'electric_potential_{it}.pdf')
            save_subplot_as_pdf(axs[0, 2], f'energy_time_dependence_{it}.pdf', True)
            save_subplot_as_pdf(axs[1, 0], f'electric_field_grid_{it}.pdf')
            save_subplot_as_pdf(axs[1, 2], f'charge_density_{it}.pdf')

        if it == 152:
            save_histogram_as_pdf(vp1, vp2, v0_1, v0_2, v_th, v_min, v_max, f'superparticles_distribution_{it}.pdf')
            save_scatter_as_pdf(xp1, vp1, xp2, vp2, L, f'phase_space_{it}.pdf')
            save_subplot_as_pdf(axs[1, 1], f'electric_potential_{it}.pdf')
            save_subplot_as_pdf(axs[0, 2], f'energy_time_dependence_{it}.pdf', True)
            save_subplot_as_pdf(axs[1, 0], f'electric_field_grid_{it}.pdf')
            save_subplot_as_pdf(axs[1, 2], f'charge_density_{it}.pdf')
        
        if it == 3500:
            save_histogram_as_pdf(vp1, vp2, v0_1, v0_2, v_th, v_min, v_max, f'superparticles_distribution_{it}.pdf')
            save_scatter_as_pdf(xp1, vp1, xp2, vp2, L, f'phase_space_{it}.pdf')
            save_subplot_as_pdf(axs[1, 1], f'electric_potential_{it}.pdf')
            save_subplot_as_pdf(axs[0, 2], f'energy_time_dependence_{it}.pdf', True)
            save_subplot_as_pdf(axs[1, 0], f'electric_field_grid_{it}.pdf')
            save_subplot_as_pdf(axs[1, 2], f'charge_density_{it}.pdf')

        plt.pause(0.001)      

if __name__ == '__main__':
    main()

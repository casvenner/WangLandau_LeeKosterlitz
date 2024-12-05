import numpy as np
from numba import njit
import utils

@njit
def initialize_lattice(q, N, M, seed):
    """Compute initial lattice configuration.
    
    :param q:
        Number of possible states per lattice site
    :param N:
        Number of lattice rows
    :param N:
        Number of lattice columns
    :param seed:
        Random number generator seed
    
    :Return:
        Numpy array containing the initial lattice configuration.
    """
    np.random.seed(seed)
    lattice = np.zeros((N,M), dtype=np.int8)
    for i in range(N):
        for j in range(M):
            lattice[i,j] = np.random.randint(0,q)
    return lattice

@njit
def compute_site_energy(i,j, lattice, J):
    """Compute energy contribution from lattice site (i,j).
    
    :param i:
        Row index of site
    :param j:
        Column index of site
    :param lattice:
        Numpy array containing the current lattice configuration
    :param J:
        Lattice interaction coefficient
    
    :Return:
        Energy contribution of lattice site.
    """
    E = 0
    i_max = lattice.shape[0]-1
    j_max = lattice.shape[1]-1
    nn_list = utils.PC_nearest_neighbour(i, j, i_max, j_max)
    for nn in nn_list:
        if lattice[i,j] == lattice[nn[0], nn[1]]:
            E -= J
    return E

@njit
def compute_total_energy(lattice, J):
    """Compute total energy of lattice configuration.
    
    :param lattice:
        Numpy array containing the current lattice configuration
    :param J:
        Lattice interaction coefficient
    
    :Return:
        Total energy of lattice configuration.
    """
    E = 0
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            E += compute_site_energy(i, j, lattice, J)
    return E/2


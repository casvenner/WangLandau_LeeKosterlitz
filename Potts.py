import numpy as np
from numba import njit
import utils
from matplotlib import pyplot as plt

@njit
def initialize_lattice(q, N, M, seed):
    np.random.seed(seed)
    lattice = np.zeros((N,M), dtype=np.int8)
    for i in range(N):
        for j in range(M):
            lattice[i,j] = np.random.randint(0,q)
    return lattice

@njit
def compute_site_energy(i,j, lattice, J):
    E = 0
    i_max = lattice.shape[0]-1
    j_max = lattice.shape[1]-1
    nn_list = utils.PC_nearest_neighbour(i, j, i_max, j_max)
    for nn in nn_list:
        if lattice[i,j] == lattice[nn[0], nn[1]]:
            E -= J
    #print("this is site E:"+str(E))
    return E

@njit
def compute_total_energy(lattice, J):
    E = 0
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            E += compute_site_energy(i, j, lattice, J)
    return E/2


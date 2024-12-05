from numba import njit
import numpy as np

@njit
def compute_bin_edges(E_min, E_max, num_bins):
    """Computes edges of energy histogram bins.
    
    :param E_min:
        The minimum value of the system energy range
    :param E_max:
        The maximum value of the system energy range
    :param num_bins:
        The number of bins of the energy histogram
    
    :Return:
        Numpy array containing the values of the bin edges of the 
        energy histogram.
    """
    bin_edges = np.zeros((num_bins+1,), dtype=np.float64)
    delta = (E_max - E_min)/num_bins
    for i in range(num_bins + 1):
        bin_edges[i] = E_min + i*delta
    bin_edges[-1] = E_max
    return bin_edges

@njit
def compute_bin(E, bin_edges):
    """Computes the energy histogram bin for the input energy.
    
    :param E:
        The energy to be placed in a bin
    :param num_bins:
        Numpy array containing the values of the bin edges.
    
    :Return:
        The index of the bin corresponding to the input energy.
    """
    n = bin_edges.shape[0] - 1
    E_min = bin_edges[0]
    E_max = bin_edges[-1]

    if E == E_max:
        return n-1
    
    bin = int(n*(E-E_min)/(E_max-E_min))

    return bin

@njit
def PC_nearest_neighbour(i,j, i_max, j_max):
    """Computes the indices of the nearest neighbors of lattice site (i,j)
    
    :param i:
        row index of initial site
    :param j:
        column index of initial site
    :param i_max:
        upper bound of site row index
    :param j_max:
        upper bound of site column index
        
    :Return:
        list of indices for the 4 different nearest neighbor sites of (i,j)"""
    nn_list = np.zeros((4,2), dtype=np.int32)
    if i == 0:
        if j ==0:
            nn_list[0] = np.array([i_max, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j_max])
            nn_list[3] = np.array([i, j+1])
        elif j == j_max:
            nn_list[0] = np.array([i_max, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, 0])
        else:
            nn_list[0] = np.array([i_max, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, j+1])
    elif i == i_max:
        if j ==0:
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([0, j])
            nn_list[2] = np.array([i, j_max])
            nn_list[3] = np.array([i, j+1])
        elif j == j_max:
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([0, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, 0])
        else:
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([0, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, j+1])
    else:
        if j ==0:
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j_max])
            nn_list[3] = np.array([i, j+1])
        elif j == j_max:
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, 0])
        else:
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, j+1])
    return nn_list
from numba import njit
import numpy as np

@njit
def part_diff(q):
    """Computes particle difference vectors.
    
    :param q:
        a numpy array of dimension (n_particle, dim) containing
        particle coordinates
    
    :Return:
        Numpy array with each row being equal to a particle difference vector.
    """

    N, dim = q.shape
    uti = np.triu_indices(N,k=1)

    dr = q[uti[0]] - q[uti[1]]            # computes differences between particle positions 

    return dr

@njit
def part_dist(dr):
    """Computes particle distances.
    
    :param dr:
        A numpy array with each row being a particle difference vector.
        
    :Return:
        a 1 x N(N-1)/2 numpy array containing the particle distances    
    """

    return np.sqrt(np.sum(dr*dr, axis=1))

@njit
def wrap(q,L):
    indices_1_row = q >= L*0.5
    indices_2_row = q < -L*0.5
    indices_1_row = np.where(indices_1_row)[0]
    indices_2_row = np.where(indices_2_row)[0]
    for i in indices_1_row:
        for j in range(3):
            if q[i,j] >= L*0.5:
                q[i,j] -= L*(np.floor(np.abs(q[i,j]/L))*0.5 + 1)
    for i in indices_2_row:
        for j in range(3):
            if q[i,j] < -L*0.5:
                q[i,j] += L*(np.floor(np.abs(q[i,j]/L))*0.5 + 1)
    return q

@njit
def compute_bin_edges(E_min, E_max, num_bins):
    bin_edges = np.zeros((num_bins+1,), dtype=np.float64)
    delta = (E_max - E_min)/num_bins
    for i in range(num_bins + 1):
        bin_edges[i] = E_min + i*delta
    bin_edges[-1] = E_max
    return bin_edges

@njit
def compute_bin(E, bin_edges):
    n = bin_edges.shape[0] - 1
    E_min = bin_edges[0]
    E_max = bin_edges[-1]

    if E == E_max:
        return n-1
    
    bin = int(n*(E-E_min)/(E_max-E_min))

    return bin

@njit
def PC_nearest_neighbour(i,j, i_max, j_max):
    nn_list = np.zeros((4,2), dtype=np.int32)
    if i == 0:
        if j ==0:
            #print("i is 0,j is 0")
            nn_list[0] = np.array([i_max, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j_max])
            nn_list[3] = np.array([i, j+1])
        elif j == j_max:
            #print("i is 0, j is max")
            nn_list[0] = np.array([i_max, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, 0])
        else:
            #print("i is 0 j is anything")
            nn_list[0] = np.array([i_max, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, j+1])
    elif i == i_max:
        if j ==0:
            #print("i is max, j is 0")
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([0, j])
            nn_list[2] = np.array([i, j_max])
            nn_list[3] = np.array([i, j+1])
        elif j == j_max:
            #print("is is max, j is max")
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([0, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, 0])
        else:
            #print("i is max, j is anything")
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([0, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, j+1])
    else:
        if j ==0:
            #print("i is anything, j is 0")
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j_max])
            nn_list[3] = np.array([i, j+1])
        elif j == j_max:
            #print("is is anything, j is max")
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, 0])
        else:
            #print("i is anything, j is anything")
            nn_list[0] = np.array([i-1, j])
            nn_list[1] = np.array([i+1, j])
            nn_list[2] = np.array([i, j-1])
            nn_list[3] = np.array([i, j+1])
    return nn_list
from numba import njit
import numpy as np
import utils
import Potts
import WangLandau

@njit(cache=True)
def simulation(lattice, N, S_hist, E_hist, E_bin_edges, num_bins, J, q, 
               f, seed):
    """Performs a WL simulation of the q-state Potts model.
    
    :param lattice:
        Numpy array containing the initial lattice configuration
    :param N:
        number of lattice sites
    :param S_hist:
        Numpy array containing the histogram of thenatural logarithm 
        of the DOS
    :param E_hist:
        Numpy array containing the energy histogram
    :param E_bin_edges:
        Numpy array containing the values of the bin edges of the
        energy histogram
    :param num_bins:
        The number of bins of the energy histogram
    :param J:
        Lattice interaction coefficient
    :param q:
        Number of possible states per lattice site
    :param f:
        initial WL modification factor
    :param seed:
        Random number generator seed
        
    :Return:
        histogram containing the natural logarithm of DOS"""
    np.random.seed(seed)
    accepted_steps=0
    rejected_steps=1
    U_current = Potts.compute_total_energy(lattice, J)
    i = 0
    lattice_test = np.zeros((lattice.shape[0], lattice.shape[1]), dtype=np.int32)
    while f > 1e-8:
        U_trial, lattice_test = WangLandau.trial_potts(lattice, q,
                                                        U_current, J)
        if U_trial > E_bin_edges[-1] or U_trial < E_bin_edges[0]:
            print(U_trial)
            continue
        current_bin = utils.compute_bin(U_current, E_bin_edges)
        trial_bin = utils.compute_bin(U_trial, E_bin_edges)
        Delta_S = S_hist[current_bin]-S_hist[trial_bin]
        test_bool = WangLandau.test(Delta_S)
        if test_bool:
            accepted_steps += 1
            U_current = U_trial
            lattice = lattice_test
            current_bin = trial_bin
        else:
            rejected_steps += 1
        S_hist[current_bin] = S_hist[current_bin] + f
        E_hist[current_bin] = E_hist[current_bin] + 1
        if (accepted_steps + rejected_steps)%(10000*N) == 0:
            print(np.max(E_hist))
            print(U_trial)
            print(np.min(E_hist[E_hist!=0])/np.max(E_hist))
            if np.min(E_hist[E_hist!=0])/np.max(E_hist) >0.8:
                i+=1
                f = f/2
                E_hist = np.zeros((num_bins,), dtype=np.float64)
                accepted_steps = 0
                print(i)
                print(f)
    return S_hist

def main():
    #initialize variables    
    seed = 12345
    L = 8
    q = 8
    J = 1
    E_min = -2*(L**2)
    E_max = 0
    num_bins = int(E_max - E_min)
    f = 1

    #initialize histograms and lattice configuration
    S_hist = np.zeros([num_bins])
    E_hist = np.zeros([num_bins])
    E_bin_edges = utils.compute_bin_edges(E_min, E_max, num_bins)
    lattice = Potts.initialize_lattice(q, L, L, seed)

    #run simulation
    S_hist = simulation(lattice, L**2, S_hist, E_hist, E_bin_edges, num_bins, J, q,
                        f, seed)

    np.savetxt("S_hist_Potts", S_hist)

if __name__ == "__main__":
    main()
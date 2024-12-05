from numba import njit
import numpy as np
import Potts
from random import randint
from random import normalvariate
from random import random

@njit
def test(Delta_S):
    """Test the trial configuration
    
    :param Delta_S:
        Change in entropy from WL trial step
        
    :Return:
        Boolean for if trial step was accepted or rejected"""

    u = np.random.random()
    e = np.exp(Delta_S)
    r = min(1,e)
    if u<=r:
        return True
    else:
        return False
    
@njit
def trial_potts(lattice, q, U_current, J):
    """Generates trial step and computes change in energy and lattice
    configuration.
    
    :param lattice:
        Numpy array containing the current lattice configuration
    :param q:
        Number of possible states per lattice site
    :param U_current:
        Total energy of lattice before trial step
    :param J:
        Lattice interaction coefficient
    
    :Return:
        Total energy of lattice and lattice configuration after trial step.
    """
    lattice_test = lattice.copy()
    U_trial=0
    p_row = np.random.randint(low=0, high=lattice.shape[0])
    p_col = np.random.randint(low=0, high=lattice.shape[1])
    tmp_q = np.random.randint(low=0, high=q)
    lattice_test[p_row, p_col] = tmp_q
    U_trial = (U_current - Potts.compute_site_energy(p_row, p_col, lattice, J) 
               + Potts.compute_site_energy(p_row, p_col, lattice_test, J))
    return U_trial, lattice_test
from numba import njit
import numpy as np
#import LennardJones
import utils
import Potts
import copy
from random import randint
from random import normalvariate
from random import random

#@njit
#def trial(q, mu, stdev, L):
#    """Computes the wrapped trial configuration
#    
#    :param q:
#        original configuration
#    :param mu:
#        mean of distribution
#    :param var:
#        variance of distribution
#    :param L:
#        box sidelength
#        
#    :Return:
#        N x 3 numpy array containing wrapped trial configuration coordinates
#        and potential energy of trial configuration."""
#    
#    qtest = q.copy()
#    p = np.random.randint(low=0,high=q.shape[0]-1)
#    for i in range(3):
#        qtest[p,i] += float(np.random.normal(loc=0,scale=stdev))
#
#    qtest = utils.wrap(qtest, L)
#    U_trial, D = LennardJones.lj(qtest, L)
#    return qtest, U_trial, D

@njit
def test(Delta_S):
    """Test the trial configuration"""

    u = np.random.random()
    e = np.exp(Delta_S)
    r = min(1,e)
    if u<=r:
        return True
    else:
        return False
    
@njit
def trial_potts(lattice, q, U_current, J):
    lattice_test = lattice.copy()
    U_trial=0
    p_row = np.random.randint(low=0, high=lattice.shape[0])
    p_col = np.random.randint(low=0, high=lattice.shape[1])
    tmp_q = np.random.randint(low=0, high=q)
    lattice_test[p_row, p_col] = tmp_q
    U_trial = (U_current - Potts.compute_site_energy(p_row, p_col, lattice, J) 
               + Potts.compute_site_energy(p_row, p_col, lattice_test, J))
    return U_trial, lattice_test
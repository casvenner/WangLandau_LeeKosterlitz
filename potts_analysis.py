import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def part_func(S_hist, E_list, T):
    """Computes partition function at temperature T using system 
    energy range and the natural logarithm of DOS

    :param S_hist:
        Numpy array containing the histogram of thenatural logarithm 
        of the DOS
    :param E_list:
        Numpy array containing a list of all energies of the system
    :param T:
        The temperature of the system
    
    :Return:
        Partition function of the system at temperature T
    """
    Z = 0
    for i, E in enumerate(E_list):
        Z += np.exp(S_hist[i]-E/T, dtype=np.float128)
    return Z


def boltzmann_pdf(S_hist, E_list, T):
    """Computes Boltzmann distribution at temperature T using system 
    energy range and the natural logarithm of DOS
    
    :param S_hist:
        Numpy array containing the histogram of thenatural logarithm 
        of the DOS
    :param E_list:
        Numpy array containing a list of all energies of the system
    :param T:
        The temperature of the system
    
    :Return:
        Boltzmann distribution of the system at temperature T
    """
    num_bins = len(E_list)
    pdf = np.zeros((num_bins,), dtype=np.float128)
    Z = part_func(S_hist, E_list, T)
    for i, E in enumerate(E_list):
        pdf[i] = np.exp(S_hist[i]-E/T, dtype=np.float128)
    pdf = pdf/Z
    return pdf

def pdf_to_txtfile(L, filename, T, save_file_name):
    """Loads ln(DOS) histogram file. Removes all entries of ln(DOS) 
    and energy histograms that correspond to 0 states in ln(DOS). 
    Normalizes DOS such that groundstate has DOS=8.
    Computes Boltzmann distribution at temperature T and normalizes
    the maximum of the distribution to 1.
    Stores energy histogram normalized pdf as txt-files.

    :param L:
        Sidelength of lattice
    :param filename:
        name of ln(DOS) histogram data file
    :param T:
        Temperature of system
    :param save_file_name:
        Name of stored pdf and energy txt-files
    
    :Return:
        None
    """
    E_min = -2*(L**2)
    E_max = 0
    E = range(E_min, E_max)
    S = np.loadtxt("data/"+filename)
    E = np.array(E)[S!=0]
    S = S[S!=0]
    S_norm = S - np.min(S) + np.log(8)
    pdf = boltzmann_pdf(S_norm, E, T)
    pdf = pdf/np.max(pdf)
    np.savetxt(fname="data/"+save_file_name+"_T"+str(T), X=pdf)
    np.savetxt(fname="data/E_"+save_file_name+"_T"+str(T), X=E)

def compute_deltaF_and_convert_F_to_txtfile(pdf_name):
    """Computes free energy F from Boltzmann distribution and stores
    F as txt-file.
    Identifies local minimum and maximum of F and plots
    them as vertical lines together with F.
    Computes height difference Delta F of minimum and maximum of F.

    :param pdf_name:
        Name of txt-file containing Boltzmann distribution
    
    :Return:
        Delta F
    """
    pdf = np.loadtxt("data/"+pdf_name)
    E = np.loadtxt("data/E_"+pdf_name)
    F = -np.log(pdf)
    np.savetxt(fname="data/F_"+pdf_name, X=F)
    F_min = signal.find_peaks(-F, width=5)
    F_max = signal.find_peaks(F, width=5)
    F_min = F_min[0][0]
    F_max = F_max[0][0]
    plt.plot(E/np.min(E),pdf)
    plt.show()
    plt.plot(-E/np.min(E),F)
    plt.xlim(-0.9,-0.4)
    plt.ylim(0,4)
    plt.axvline(-E[F_min]/np.min(E))
    plt.axvline(-E[F_max]/np.min(E))
    plt.show()
    DeltaF = F[F_max] - F[F_min]
    return DeltaF



Delta_F = []
L_list = [12, 16, 24, 32]
T_list = [0.756, 0.7515, 0.7481, 0.74675]
for i in range(4):
    L = L_list[i]
    T = T_list[i]
    Delta_F.append(compute_deltaF_and_convert_F_to_txtfile(
        "Boltzmann_hist_potts_q8_L"+str(L)+"_f1neg8_T"+str(T)))
np.savetxt(fname="data/Delta_F_12_16_24_32", X=Delta_F)
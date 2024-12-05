import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def part_func(S_hist, E_list, T):
    Z = 0
    for i, E in enumerate(E_list):
        #print(i)
        Z += np.exp(S_hist[i]-E/T, dtype=np.float128)
    return Z

#@njit
def boltzmann_pdf(S_hist, E_new, T):
    #E_list = range(E_min, E_max)
    num_bins = len(E_new)
    pdf = np.zeros((num_bins,), dtype=np.float128)
    Z = part_func(S_hist, E_new, T)
    for i, E in enumerate(E_new):
        #print(i)
        pdf[i] = np.exp(S_hist[i]-E/T, dtype=np.float128)
    pdf = pdf/Z
    return pdf

def pdf_to_txtfile(L, filename, T, save_file_name):
    E_min = -2*(L**2)
    E_max = 0
    E = range(E_min, E_max)
    S = np.loadtxt(filename)
    E = np.array(E)[S!=0]
    #E = E/(2*L**2)
    S = S[S!=0]
    S_norm = S - np.min(S) + np.log(8)
    pdf = boltzmann_pdf(S_norm, E, T)
    pdf = pdf/np.max(pdf)
    np.savetxt(fname=save_file_name+"_T"+str(T), X=pdf)
    np.savetxt(fname="E_"+save_file_name+"_T"+str(T), X=E)

def compute_deltaF_and_convert_F_to_txtfile(pdf_name):
    pdf = np.loadtxt(pdf_name)
    E = np.loadtxt("E_"+pdf_name)
    F = -np.log(pdf)
    np.savetxt(fname="F_"+pdf_name, X=F)
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
    print(DeltaF)
    return DeltaF


#pdf_to_txtfile(L=12,filename="S_hist_Potts_q8_L12_f1neg8",
#                T=0.756, save_file_name="Boltzmann_hist_potts_q8_L12_f1neg8")

L = 12
E_min = -2*(L**2)
E_max = 0
E = range(E_min,E_max)
#print(len(E))
##print("hi")
S = np.loadtxt("S_hist_Potts_q8_L12_f1neg8")
S_new = S[S!=0]
E_new = np.array(E)[S!=0]
#print(len(S))
S_norm = S_new - np.min(S_new) + np.log(8)
#print(S_norm)
pdf = [None]
#for i in range(1,6):
pdf = boltzmann_pdf(S_norm, E_new, 0.756)
pdf = pdf/np.max(pdf)
#delta_F = -np.log(pdf)
#fig, ax = plt.subplots(2,3)
#ind = 0
#E_new = E_new/(2*L**2)
#for i in range(2):
#    for j in range(3):
#        ax[i,j].plot(E, pdf[ind+j])
#    ind +=1
plt.plot(E_new,pdf)
#plt.ylim(1e-2,1.2)
#plt.ylim(0,4)
#plt.xlim(-0.9,-0.4)
plt.show()

Delta_F = []
L_list = [12, 16, 24, 32]
T_list = [0.756, 0.7515, 0.7481, 0.74675]
for i in range(4):
    L = L_list[i]
    T = T_list[i]
    Delta_F.append(compute_deltaF_and_convert_F_to_txtfile(
        "Boltzmann_hist_potts_q8_L"+str(L)+"_f1neg8_T"+str(T)))
print(Delta_F)
np.savetxt(fname="Delta_F_12_16_24_32", X=Delta_F)
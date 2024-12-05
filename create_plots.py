import numpy as np
from matplotlib import pyplot as plt

# Load data files
X_12 = np.loadtxt("data/E_Boltzmann_hist_potts_q8_L12_f1neg8_T0.756")
Y_12 = np.loadtxt("data/Boltzmann_hist_potts_q8_L12_f1neg8_T0.756")
F_12 = np.loadtxt("data/F_Boltzmann_hist_potts_q8_L12_f1neg8_T0.756")
X_16 = np.loadtxt("data/E_Boltzmann_hist_potts_q8_L16_f1neg8_T0.7515")
Y_16 = np.loadtxt("data/Boltzmann_hist_potts_q8_L16_f1neg8_T0.7515")
F_16 = np.loadtxt("data/F_Boltzmann_hist_potts_q8_L16_f1neg8_T0.7515")
X_24 = np.loadtxt("data/E_Boltzmann_hist_potts_q8_L24_f1neg8_T0.7481")
Y_24 = np.loadtxt("data/Boltzmann_hist_potts_q8_L24_f1neg8_T0.7481")
F_24 = np.loadtxt("data/F_Boltzmann_hist_potts_q8_L24_f1neg8_T0.7481")
X_32 = np.loadtxt("data/E_Boltzmann_hist_potts_q8_L32_f1neg8_T0.74675")
Y_32 = np.loadtxt("data/Boltzmann_hist_potts_q8_L32_f1neg8_T0.74675")
F_32 = np.loadtxt("data/F_Boltzmann_hist_potts_q8_L32_f1neg8_T0.74675")
Delta_F = np.loadtxt("data/Delta_F_12_16_24_32")
L_inv = [1/12, 1/16, 1/24, 1/32]

# Create labels and colors
colors = ["blue", "orange", "green", "red"]
labels=["$L=12$, $T_C=0.756$", "$L=16$, $T_C=0.7515$",
        "$L=24$, $T_C=0.7481$","$L=32$, $T_C=0.74675$" ]

# Create maximum noramlized plots of Boltzmann distributions
fig, axes = plt.subplots(2,2, figsize=(8,8))
axes[0,0].plot(X_12/np.min(X_12), Y_12, label=labels[0])
axes[0,1].plot(X_16/np.min(X_16), Y_16, label=labels[1])
axes[1,0].plot(X_24/np.min(X_24), Y_24, label=labels[2])
axes[1,1].plot(X_32/np.min(X_32), Y_32, label=labels[3])
axes[1,0].set_xlabel("$-E$", rotation=0, labelpad=20,
                     fontdict={'size':15,})
axes[1,1].set_xlabel("$-E$", rotation=0, labelpad=20,
                     fontdict={'size':15,})
axes[0,0].set_ylabel("$\dfrac{P(E)}{P(E)_{\max}}$", rotation=0, labelpad=25,
                     fontdict={'size':15,})
axes[1,0].set_ylabel("$\dfrac{P(E)}{P(E)_{\max}}$", rotation=0, labelpad=25,
                     fontdict={'size':15,})
axes[0,0].set_ylim(0,1.2)
axes[0,1].set_ylim(0,1.2)
axes[1,0].set_ylim(0,1.2)
axes[1,1].set_ylim(0,1.2)
axes[0,0].legend()
axes[0,1].legend()
axes[1,0].legend()
axes[1,1].legend()
fig.tight_layout()
plt.savefig("plots/boltzmann.pdf")
plt.show()

# Create translated free energy plots
plt.plot(X_12/np.min(X_12), F_12, label=labels[0])
plt.plot(X_16/np.min(X_16), F_16, label=labels[1])
plt.plot(X_24/np.min(X_24), F_24, label=labels[2])
plt.plot(X_32/np.min(X_32), F_32, label=labels[3])
plt.xlim(0.4,0.9)
plt.ylim(0,3.5)
plt.xlabel("$-E$", rotation=0, labelpad=15,
                     fontdict={'size':15,})
plt.ylabel("$F(E)-F(E)_{\min}$", rotation=0, labelpad=35,
                     fontdict={'size':15,})
plt.legend()
plt.tight_layout()
plt.savefig("plots/F.pdf")
plt.show()

# Create Delta F plots
for i in range(4):
    plt.scatter(L_inv[i], Delta_F[i], c=colors[i], 
                label=labels[i])
plt.plot(L_inv, Delta_F)
plt.ylim(0,2.5)
plt.xlabel("$L^{-1}$", rotation=0, labelpad=20,
                     fontdict={'size':15,})
plt.ylabel("$\Delta F$", rotation=0, labelpad=20,
                     fontdict={'size':15,})
plt.legend()
plt.tight_layout()
plt.savefig("plots/deltaF.pdf")
plt.show()
from matplotlib import pyplot as plt
import numpy as np
from functions.fisher_matrix import *
from math import radians, pi

LMAX = 2500

# Characteristic of Planck noise [theta beam (arcmin), Temperature noise (muK.deg), Polarisation noise (muK.deg)] from Planck18 p8
charact_Planck = {
    "100": [9.66 / 60 * pi / 180, 1.29 * pi / 180, 1.96 * pi / 180],
    "143": [7.22 / 60 * pi / 180, 0.55 * pi / 180, 1.17 * pi / 180],
    "217": [4.90 / 60 * pi / 180, 0.78 * pi / 180, 1.75 * pi / 180],
}


cls = mk_ini_spectra(cut_2_l=True, LMAX=LMAX)

ls = np.arange(2, LMAX)
fac = ls * (ls + 1) / 2 * pi

linestyles = ["solid", "dashed", "dashdot"]
labels = ["Temp", "Pol"]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
which_axs = {"TT": 0, "EE": 1, "BB": 1, "TE": 1}

for c, band in enumerate(charact_Planck.keys()):
    theta_FWHM, s_T, s_pol = charact_Planck[band]
    s = {"TT": s_T, "EE": s_pol, "BB": s_pol, "TE": 0}
    nls = get_nls(s, theta_FWHM, lmax=LMAX)

    for l, key in enumerate(["TT", "EE"]):
        axs[which_axs[key]].plot(
            ls,
            nls[key] * fac,
            color="C" + str(c),
            label=band + "GHz " + labels[l],
            linestyle=linestyles[l],
            alpha=0.8,
        )

for l, key in enumerate(["TT", "EE", "BB"]):
    axs[which_axs[key]].plot(
        ls, cls[key] * fac, color="black", label=key, linestyle=linestyles[l], alpha=1
    )

total_noise = get_total_noise(charact_Planck, lmax=LMAX)
for l, key in enumerate(["TT", "EE"]):
    axs[which_axs[key]].semilogy(
        total_noise[key] * fac,
        color="C3",
        label="All",
        linestyle=linestyles[l],
        linewidth=2.5,
    )
    axs[which_axs[key]].legend()

axs[0].set_title("Temperature")
axs[1].set_title("Polarisation")

plt.tight_layout()
plt.savefig("Figures/Effective_noise.png")

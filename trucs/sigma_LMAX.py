import numpy as np
from matplotlib import pyplot as plt
from functions.simu import *
from functions.fisher_matrix import *
from math import radians, pi

LMAX = 2500

names = ["TT", "EE", "BB", "TE"]

spectra = "total"

cosmo_params = {
    "ombh2": 0.022,
    "omch2": 0.122,
    "cosmomc_theta": 1.04 / 100,
    "tau": 0.0544,
    "As": 2e-9,
    "ns": 0.965,
}

data_cls = mk_ini_spectra(cosmo_params, spectra, LMAX, cut_2_l=True)
# Characteristic of Planck noise [theta beam (arcmin), Temperature noise (muK.deg), Polarisation noise (muK.deg)] from Planck18 p8
charact_Planck = {
    "100": [9.66 / 60 * pi / 180, 1.29 * pi / 180, 1.96 * pi / 180],
    "143": [7.22 / 60 * pi / 180, 0.55 * pi / 180, 1.17 * pi / 180],
    "217": [4.90 / 60 * pi / 180, 0.78 * pi / 180, 1.75 * pi / 180],
}

params_sigmas = {}

nls = get_total_noise(charact_Planck, lmax=LMAX)
cnls = {}
for key in data_cls.keys():
    cnls[key] = data_cls[key] + nls[key]
data = [cnls, data_cls]
linestyle_noise = ["solid", "dashed"]

for n in range(2):
    fisher_matrices = get_fisher_matrix(
        data[n],
        cosmo_params,
        LMAX,
        data_cls_bool=True,
        cut_2_l=True,
        step=0.01,
        plot_noise=None,
        dont_sum=True,
    )

    l_range = list(range(100, LMAX, 100))
    sigmas = np.zeros((len(l_range), len(cosmo_params.keys())), dtype=float)
    for i, l in enumerate(l_range):
        fisher_matrix = np.sum(fisher_matrices[:l], axis=0)
        sigmas[i] = fisher_to_sigmas(fisher_matrix)

    for i, param in enumerate(cosmo_params.keys()):
        plt.plot(
            l_range,
            sigmas.T[i],
            label=param,
            color="C" + str(i),
            linestyle=linestyle_noise[n],
            alpha=0.8,
        )
plt.xlabel("LMAX")
plt.ylabel("sigma(H0)/H0")
plt.legend(list(cosmo_params.keys()), loc="lower right")
plt.xlim(0, LMAX)
plt.semilogy()
plt.savefig("Figures/sigma_LMAX.png")

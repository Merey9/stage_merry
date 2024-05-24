import numpy as np
from functions.simu import *
from functions.fisher_matrix import *
from math import radians, pi
from functions.plot_functions import *

LMAX = 2500
print("LMAX = " + str(LMAX))

names = ["TT", "EE", "BB", "TE"]

spectra = "total"

# Planck resulsts
cosmo_params = {
    "ombh2": 0.022,
    "omch2": 0.122,
    "cosmomc_theta": 1.04 / 100,
    "tau": 0.0544,
    "As": 2e-9,
    "ns": 0.965,
}

planck_var = {
    "ombh2": 0.00015,
    "omch2": 0.0012,
    "cosmomc_theta": 0.00031 / 100,
    "tau": 0.0073,
    "As": (np.exp(3.058) - np.exp(3.044)) * 1e-10 / 2,
    "ns": 0.0042,
}


data_cls = mk_ini_spectra(
    cosmo_params,
    spectra,
    LMAX,
    cut_2_l=True,
    wanted_keys=["TT", "EE", "BB", "TE", "EB"],
)
# Characteristic of Planck noise [theta beam (arcmin), Temperature noise (muK.deg), Polarisation noise (muK.deg)] from Planck18 p8
charact_Planck = {
    "100": [9.66 / 60 * pi / 180, 1.29 * pi / 180, 1.96 * pi / 180],
    "143": [7.22 / 60 * pi / 180, 0.55 * pi / 180, 1.17 * pi / 180],
    "217": [4.90 / 60 * pi / 180, 0.78 * pi / 180, 1.75 * pi / 180],
}

nls = get_total_noise(charact_Planck, lmax=LMAX)

for key in nls.keys():
    data_cls[key] += nls[key]

fisher_matrix = get_fisher_matrix(
    data_cls,
    cosmo_params,
    LMAX,
    data_cls_bool=True,
    cut_2_l=True,
    step=0.01,
    plot_noise=None,
    fsky=0.6,
)

fisher_matrix_inv = np.linalg.inv(fisher_matrix)

sigma_params = {}
for i, key in enumerate(cosmo_params):
    sigma_params[key] = fisher_to_sigmas(fisher_matrix)[i]
    print(key, cosmo_params[key], sigma_params[key])

plot_params_var(
    fisher_matrix,
    planck_var,
    cosmo_params,
    filename="Figures/params_results",
    lmax=LMAX,
)
